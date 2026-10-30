//! Process-private image tickets and completion fencing around shared provider jobs.
//! A provider's terminal Cancelled snapshot does not prove its native worker has returned.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use craft_ai_core::{CapabilityId, Event, InputRole, Output, Payload, Request};

use super::{MASK_MEDIA, OUTPUT_MEDIA, SOURCE_MEDIA};
use crate::{Error, RemovalInput, RemovalOutput, Result};

const MAX_TICKETS: usize = 32;
static NEXT_TICKET: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
pub(crate) struct Bridge {
    tickets: Mutex<HashMap<u64, Arc<Ticket>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Pending,
    Running,
    Finished,
    Stopped,
}
struct State {
    phase: Phase,
    output: Option<RemovalOutput>,
}
pub(super) struct Ticket {
    id: u64,
    pub(super) input: RemovalInput,
    pub(super) available: u64,
    pub(super) component: &'static str,
    request: Request,
    cancelled: AtomicBool,
    state: Mutex<State>,
    finished: Condvar,
}

/// Keeps the registered input and outer native admission alive until the real worker drains.
pub(crate) struct TicketGuard {
    bridge: Arc<Bridge>,
    ticket: Arc<Ticket>,
}

impl Bridge {
    pub(crate) fn request(self: &Arc<Self>, input: &RemovalInput, available: u64, component: &str) -> Result<TicketGuard> {
        input.validate()?;
        let component = crate::catalog::find(component)?.id;
        // compare_exchange works on the supported toolchain as well as newer Rust,
        // where fetch_update was renamed. Exhaustion must never wrap into a live ID.
        let mut candidate = NEXT_TICKET.load(Ordering::Acquire);
        let id = loop {
            let next = candidate.checked_add(1).ok_or_else(|| invalid("the removal ticket sequence is exhausted"))?;
            match NEXT_TICKET.compare_exchange_weak(candidate, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break candidate,
                Err(current) => candidate = current,
            }
        };
        let source_bytes = (input.rgb.len() as u64).checked_mul(4).ok_or_else(|| invalid("removal image size overflow"))?;
        let mask_bytes = (input.mask.len() as u64).checked_mul(4).ok_or_else(|| invalid("removal mask size overflow"))?;
        let request = Request::new(CapabilityId::parse("image.inpaint").map_err(invalid)?)
            .with_input(InputRole::Source, Payload::handle(&uri(id, "source"), SOURCE_MEDIA, Some(source_bytes)).map_err(invalid)?)
            .with_input(InputRole::Mask, Payload::handle(&uri(id, "mask"), MASK_MEDIA, Some(mask_bytes)).map_err(invalid)?)
            .with_param("width", input.width.into())
            .with_param("height", input.height.into())
            .with_param("availableBytes", available.into())
            .with_param("componentModel", component.into());
        let mut tickets = self.tickets.lock().unwrap_or_else(PoisonError::into_inner);
        if tickets.len() >= MAX_TICKETS {
            return Err(Error::Busy);
        }
        let input = RemovalInput { width: input.width, height: input.height, rgb: copy_samples(&input.rgb)?, mask: copy_samples(&input.mask)? };
        let ticket = Arc::new(Ticket {
            id,
            input,
            available,
            component,
            request,
            cancelled: AtomicBool::new(false),
            state: Mutex::new(State { phase: Phase::Pending, output: None }),
            finished: Condvar::new(),
        });
        tickets.insert(id, ticket.clone());
        Ok(TicketGuard { bridge: self.clone(), ticket })
    }

    pub(super) fn enter(&self, request: &Request) -> Result<Flight> {
        let id = request
            .inputs
            .iter()
            .find(|input| input.role == InputRole::Source)
            .and_then(|input| match &input.payload {
                Payload::Handle(handle) => Some(handle.uri.as_str()),
                _ => None,
            })
            .and_then(|text| text.strip_prefix("photocraft-memory:"))
            .and_then(|text| text.strip_suffix("/source"))
            .and_then(|text| text.parse::<u64>().ok())
            .ok_or_else(|| invalid("the removal request has no private image ticket"))?;
        let ticket = self
            .tickets
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("the removal image ticket is no longer active"))?;
        // Exact handles, shape and scalars must match the frozen registration. In particular,
        // request params never grant extra memory or change the admitted component.
        if request != &ticket.request {
            return Err(invalid("the removal request does not match its registered image ticket"));
        }
        let mut state = ticket.lock();
        if state.phase != Phase::Pending || ticket.cancelled() {
            return Err(Error::Cancelled);
        }
        state.phase = Phase::Running;
        drop(state);
        Ok(Flight { ticket })
    }
}

impl Ticket {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
    pub(super) fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        let mut state = self.lock();
        if state.phase == Phase::Pending {
            state.phase = Phase::Stopped;
            self.finished.notify_all();
        }
    }
    fn cancel_and_drain(&self) {
        self.cancel();
        let mut state = self.lock();
        while state.phase == Phase::Running {
            state = self.finished.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// Drop runs after inference, synchronization, output validation and any escaped unwinding.
pub(super) struct Flight {
    pub(super) ticket: Arc<Ticket>,
}
impl Flight {
    pub(super) fn publish(&self, output: RemovalOutput) -> Result<Output> {
        validate_output(&self.ticket.input, &output)?;
        let bytes = ((output.rgb.len() + output.edit_alpha.as_ref().map_or(0, Vec::len)) as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("removal result size overflow"))?;
        let payload = Payload::handle(&uri(self.ticket.id, "output"), OUTPUT_MEDIA, Some(bytes)).map_err(invalid)?;
        let mut state = self.ticket.lock();
        if state.phase != Phase::Running || self.ticket.cancelled() {
            return Err(Error::Cancelled);
        }
        state.output = Some(output);
        Ok(Output { payload, label: None })
    }
}
impl Drop for Flight {
    fn drop(&mut self) {
        let mut state = self.ticket.lock();
        if state.phase == Phase::Running {
            state.phase = Phase::Finished;
        }
        self.ticket.finished.notify_all();
    }
}

impl TicketGuard {
    pub(crate) fn request(&self) -> Request {
        self.ticket.request.clone()
    }
    pub(crate) fn cancel(&self) {
        self.ticket.cancel();
    }
    pub(crate) fn cancel_and_drain(&self) {
        self.ticket.cancel_and_drain();
    }
    pub(crate) fn take_output(&self, events: &[Event]) -> Result<RemovalOutput> {
        let mut outputs = events.iter().filter_map(|event| match event {
            Event::Output(output) => Some(output),
            _ => None,
        });
        let output = outputs.next().ok_or_else(|| invalid("the removal model returned no image"))?;
        if outputs.next().is_some() {
            return Err(invalid("the removal model returned more than one image"));
        }
        let mut state = self.ticket.lock();
        if state.phase != Phase::Finished || self.ticket.cancelled() {
            return Err(Error::Cancelled);
        }
        let stored = state.output.as_ref().ok_or_else(|| invalid("the removal result was already consumed or never produced"))?;
        let bytes = ((stored.rgb.len() + stored.edit_alpha.as_ref().map_or(0, Vec::len)) as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("removal result size overflow"))?;
        let expected = Payload::handle(&uri(self.ticket.id, "output"), OUTPUT_MEDIA, Some(bytes)).map_err(invalid)?;
        if output.payload != expected {
            return Err(invalid("the removal answer does not match this image ticket"));
        }
        state.output.take().ok_or_else(|| invalid("the removal result was already consumed"))
    }
}
impl Drop for TicketGuard {
    fn drop(&mut self) {
        self.cancel_and_drain();
        self.bridge.tickets.lock().unwrap_or_else(PoisonError::into_inner).remove(&self.ticket.id);
    }
}

fn uri(id: u64, part: &str) -> String {
    format!("photocraft-memory:{id}/{part}")
}
fn invalid(error: impl std::fmt::Display) -> Error {
    Error::InvalidInput(error.to_string())
}
fn copy_samples(samples: &[f32]) -> Result<Vec<f32>> {
    let mut result = Vec::new();
    result.try_reserve_exact(samples.len()).map_err(|_| invalid("not enough memory for the admitted removal image"))?;
    result.extend_from_slice(samples);
    Ok(result)
}
fn validate_output(input: &RemovalInput, output: &RemovalOutput) -> Result<()> {
    if output.rgb.len() != input.rgb.len()
        || output.rgb.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || output.edit_alpha.as_ref().is_some_and(|a| a.len() != input.mask.len() || a.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)))
        || !output.inference_scale.is_finite()
        || !(0.0..=1.0).contains(&output.inference_scale)
        || output.inference_scale == 0.0
        || !output.load_ms.is_finite()
        || output.load_ms < 0.0
        || !output.forward_ms.is_finite()
        || output.forward_ms < 0.0
    {
        return Err(invalid("the removal model returned malformed pixels or diagnostics"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use craft_ai_engine::ModelRunner;
    use std::sync::mpsc;
    use std::time::Duration;

    fn input() -> RemovalInput {
        RemovalInput { width: 2, height: 1, rgb: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], mask: vec![0.0, 0.75] }
    }
    fn output() -> RemovalOutput {
        RemovalOutput { rgb: input().rgb, edit_alpha: Some(vec![0.25, 1.0]), inference_scale: 0.5, load_ms: 6.25, forward_ms: 10.5 }
    }
    #[test]
    fn private_handles_preserve_float_pixels_and_take_exact_result_once() {
        let bridge = Arc::new(Bridge::default());
        let guard = bridge.request(&input(), 12345, "lama").unwrap();
        let request = guard.request();
        let flight = bridge.enter(&request).unwrap();
        assert_eq!(flight.ticket.input.rgb, input().rgb);
        assert_eq!(flight.ticket.input.mask, input().mask);
        assert_eq!(flight.ticket.available, 12345);
        let answer = flight.publish(output()).unwrap();
        drop(flight);
        let mut wrong = answer.clone();
        wrong.payload = Payload::handle("photocraft-memory:999/output", OUTPUT_MEDIA, None).unwrap();
        assert!(guard.take_output(&[Event::Output(wrong)]).is_err());
        let events = [Event::Output(answer)];
        let result = guard.take_output(&events).unwrap();
        assert_eq!(result.rgb, output().rgb);
        assert_eq!(result.edit_alpha, output().edit_alpha);
        assert_eq!(result.load_ms, 6.25);
        assert_eq!(result.forward_ms, 10.5);
        assert_eq!(result.inference_scale, 0.5);
        assert!(guard.take_output(&events).is_err());
    }
    #[test]
    fn shared_request_preserves_supported_high_resolution_crops() {
        let pixels = 1500 * 1500;
        let mut input = RemovalInput { width: 1500, height: 1500, rgb: vec![0.25; pixels * 3], mask: vec![0.0; pixels] };
        *input.mask.last_mut().unwrap() = 1.0;
        let bridge = Arc::new(Bridge::default());
        let guard = bridge.request(&input, 1 << 30, "lama").expect("the shared transport must preserve supported high-resolution crops");
        let request = guard.request();
        assert!(request.inputs.iter().all(|input| matches!(input.payload, Payload::Handle(_))));
        assert!(
            request
                .validate(
                    &super::super::Runner::new(Arc::new(crate::store::NativeService::new(None)), crate::catalog::find("lama").unwrap(), bridge.clone())
                        .unwrap()
                        .model()
                        .capabilities[0],
                    &craft_ai_core::payload::Limits::default()
                )
                .is_ok()
        );
        let flight = bridge.enter(&request).unwrap();
        assert_eq!(flight.ticket.input.width, 1500);
        assert_eq!(flight.ticket.input.rgb.len(), pixels * 3);
    }
    #[test]
    fn cancellation_before_entry_refuses_late_work_and_drop_unregisters() {
        let bridge = Arc::new(Bridge::default());
        let guard = bridge.request(&input(), 100, "lama").unwrap();
        let request = guard.request();
        guard.cancel();
        assert!(matches!(bridge.enter(&request), Err(Error::Cancelled)));
        drop(guard);
        assert!(bridge.enter(&request).is_err());
        assert!(bridge.tickets.lock().unwrap().is_empty());
    }
    #[test]
    fn cancellation_does_not_release_a_running_native_worker_early() {
        let bridge = Arc::new(Bridge::default());
        let guard = bridge.request(&input(), 100, "lama").unwrap();
        let flight = bridge.enter(&guard.request()).unwrap();
        let ticket = flight.ticket.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let (drained_tx, drained_rx) = mpsc::channel();
        let wait = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            guard.cancel_and_drain();
            drained_tx.send(()).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(drained_rx.recv_timeout(Duration::from_millis(50)).is_err());
        assert_eq!(ticket.lock().phase, Phase::Running);
        drop(flight);
        drained_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        wait.join().unwrap();
        assert!(bridge.tickets.lock().unwrap().is_empty());
    }
    #[test]
    fn dropping_after_poll_failure_still_drains_the_native_worker() {
        let bridge = Arc::new(Bridge::default());
        let guard = bridge.request(&input(), 100, "lama").unwrap();
        let flight = bridge.enter(&guard.request()).unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        let wait = std::thread::spawn(move || {
            drop(guard);
            done_tx.send(()).unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(50)).is_err());
        drop(flight);
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        wait.join().unwrap();
        assert!(bridge.tickets.lock().unwrap().is_empty());
    }
    #[test]
    fn foreign_tickets_and_changed_budget_or_component_are_refused() {
        let bridge = Arc::new(Bridge::default());
        let guard = bridge.request(&input(), 100, "lama").unwrap();
        let mut changed = guard.request();
        changed.params.insert("availableBytes".into(), u64::MAX.into());
        assert!(bridge.enter(&changed).is_err());
        let mut changed = guard.request();
        changed.params.insert("componentModel".into(), "osor-sdxl".into());
        assert!(bridge.enter(&changed).is_err());
        let other = Arc::new(Bridge::default());
        let other_guard = other.request(&input(), 100, "lama").unwrap();
        assert!(other.enter(&guard.request()).is_err());
        assert!(bridge.enter(&other_guard.request()).is_err());
    }
}
