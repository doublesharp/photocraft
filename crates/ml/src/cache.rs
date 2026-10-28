//! Budgeted warm-model residency, released after five idle minutes.
#[cfg(any(feature = "native-inference", feature = "osor"))]
pub(crate) use enabled::*;

#[cfg(not(any(feature = "native-inference", feature = "osor")))]
pub(crate) fn prepare(_: &str, _: bool, _: u64) {}
#[cfg(not(any(feature = "native-inference", feature = "osor")))]
pub(crate) fn admit(_: bool, _: u64) {}
#[cfg(not(any(feature = "native-inference", feature = "osor")))]
pub(crate) fn clear_model(_: &str) {}
#[cfg(not(any(feature = "native-inference", feature = "osor")))]
pub(crate) fn idle() {
    crate::store::BUSY.store(false, std::sync::atomic::Ordering::Release);
}

#[cfg(any(feature = "native-inference", feature = "osor"))]
mod enabled {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
    use std::time::{Duration, Instant};

    #[cfg(feature = "native-inference")]
    type Onnx = crate::runtime::Cached;
    #[cfg(not(feature = "native-inference"))]
    type Onnx = ();
    #[cfg(feature = "osor")]
    type Osor = crate::osor_runtime::Cached;
    #[cfg(not(feature = "osor"))]
    type Osor = ();

    struct Resident<T> {
        id: &'static str,
        bytes: u64,
        value: T,
    }

    struct State<O, S> {
        onnx: Option<Resident<O>>,
        osor: Option<Resident<S>>,
        idle_since: Option<Instant>,
        running: bool,
    }

    impl<O, S> State<O, S> {
        fn new() -> Self {
            Self { onnx: None, osor: None, idle_since: None, running: false }
        }

        fn empty(&self) -> bool {
            self.onnx.is_none() && self.osor.is_none()
        }

        fn clear(&mut self) {
            self.onnx = None;
            self.osor = None;
            self.idle_since = None;
        }

        fn admit(&mut self, injected: bool, available: u64) {
            self.idle_since = None;
            if injected {
                self.clear();
                return;
            }
            let onnx = self.onnx.as_ref().map_or(0, |entry| entry.bytes);
            let osor = self.osor.as_ref().map_or(0, |entry| entry.bytes);
            if onnx.checked_add(osor).is_none_or(|total| total > available) {
                self.onnx = None;
            }
            if osor > available {
                self.osor = None;
            }
        }

        fn prepare(&mut self, id: &str, required: u64, injected: bool, available: u64) {
            self.idle_since = None;
            if injected || required > available {
                self.clear();
                return;
            }
            if self.onnx.as_ref().is_some_and(|entry| entry.id != id) && id != "osor-sdxl" {
                self.onnx = None;
            }
            if id == "osor-sdxl" {
                if self.onnx.as_ref().is_some_and(|entry| required.checked_add(entry.bytes).is_none_or(|total| total > available)) {
                    self.onnx = None;
                }
            } else if self.osor.as_ref().is_some_and(|entry| required.checked_add(entry.bytes).is_none_or(|total| total > available)) {
                self.osor = None;
            }
        }
    }

    struct Cache<O, S> {
        state: Mutex<State<O, S>>,
        changed: Condvar,
        ttl: Duration,
    }

    impl<O, S> Cache<O, S> {
        fn new(ttl: Duration) -> Self {
            Self { state: Mutex::new(State::new()), changed: Condvar::new(), ttl }
        }

        fn lock(&self) -> MutexGuard<'_, State<O, S>> {
            self.state.lock().unwrap_or_else(PoisonError::into_inner)
        }

        // Release admission under the same mutex as the deadline update. An expiring
        // worker cannot observe a stale deadline immediately after the owner releases.
        fn mark_idle(&self, busy: &AtomicBool) -> bool {
            let mut state = self.lock();
            state.idle_since = (!state.empty()).then(Instant::now);
            let start = !state.empty() && !state.running;
            if start {
                state.running = true;
            }
            busy.store(false, Ordering::Release);
            self.changed.notify_all();
            start
        }

        fn failed_start(&self, busy: &AtomicBool) {
            let mut state = self.lock();
            state.running = false;
            // Another owner may have acquired admission after mark_idle returned.
            if busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                state.clear();
                busy.store(false, Ordering::Release);
            }
            self.changed.notify_all();
        }

        fn reap(&self, busy: &AtomicBool) {
            let mut state = self.lock();
            loop {
                if state.empty() {
                    state.running = false;
                    return;
                }
                let Some(since) = state.idle_since else {
                    state = self.changed.wait(state).unwrap_or_else(PoisonError::into_inner);
                    continue;
                };
                let remaining = self.ttl.saturating_sub(since.elapsed());
                if !remaining.is_zero() {
                    (state, _) = self.changed.wait_timeout(state, remaining).unwrap_or_else(PoisonError::into_inner);
                    continue;
                }
                // A real forward, preprocessing, or retained preview owns BUSY. Never
                // free a model beneath it, including the interval before prepare().
                if busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
                    state = self.changed.wait(state).unwrap_or_else(PoisonError::into_inner);
                    continue;
                }
                state.clear();
                state.running = false;
                busy.store(false, Ordering::Release);
                self.changed.notify_all();
                return;
            }
        }
    }

    static CACHE: LazyLock<Arc<Cache<Onnx, Osor>>> = LazyLock::new(|| Arc::new(Cache::new(Duration::from_secs(5 * 60))));

    fn estimate(id: &str) -> u64 {
        crate::catalog::find(id).map_or(u64::MAX, |model| u64::from(model.memory_mib) * 1024 * 1024)
    }

    pub(crate) fn prepare(id: &str, injected: bool, available_bytes: u64) {
        CACHE.lock().prepare(id, estimate(id), injected, available_bytes);
        CACHE.changed.notify_all();
    }

    pub(crate) fn admit(injected: bool, available_bytes: u64) {
        CACHE.lock().admit(injected, available_bytes);
        CACHE.changed.notify_all();
    }

    pub(crate) fn clear_model(id: &str) {
        let mut state = CACHE.lock();
        if state.onnx.as_ref().is_some_and(|entry| entry.id == id) {
            state.onnx = None;
        }
        if state.osor.as_ref().is_some_and(|entry| entry.id == id) {
            state.osor = None;
        }
        CACHE.changed.notify_all();
    }

    #[cfg(feature = "native-inference")]
    pub(crate) fn take_onnx() -> Option<crate::runtime::Cached> {
        let mut state = CACHE.lock();
        state.idle_since = None;
        state.onnx.take().map(|entry| entry.value)
    }

    #[cfg(feature = "native-inference")]
    pub(crate) fn put_onnx(value: crate::runtime::Cached) {
        let mut state = CACHE.lock();
        state.idle_since = None;
        state.onnx = Some(Resident { id: value.model_id, bytes: estimate(value.model_id), value });
    }

    #[cfg(feature = "osor")]
    pub(crate) fn take_osor() -> Option<crate::osor_runtime::Cached> {
        let mut state = CACHE.lock();
        state.idle_since = None;
        state.osor.take().map(|entry| entry.value)
    }

    #[cfg(feature = "osor")]
    pub(crate) fn put_osor(value: crate::osor_runtime::Cached) {
        let mut state = CACHE.lock();
        state.idle_since = None;
        state.osor = Some(Resident { id: "osor-sdxl", bytes: estimate("osor-sdxl"), value });
    }

    pub(crate) fn idle() {
        if CACHE.mark_idle(&crate::store::BUSY) {
            let cache = Arc::clone(&CACHE);
            if std::thread::Builder::new().name("photocraft-model-idle".into()).spawn(move || cache.reap(&crate::store::BUSY)).is_err() {
                // Retention is optional; failure to create its sole worker must not
                // turn a completed edit into an error or retain an idle model forever.
                CACHE.failed_start(&crate::store::BUSY);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::mpsc;

        fn state() -> State<(), ()> {
            let mut state = State::new();
            state.onnx = Some(Resident { id: "lama", bytes: 3, value: () });
            state.osor = Some(Resident { id: "osor-sdxl", bytes: 40, value: () });
            state
        }

        #[test]
        fn active_model_and_warm_companion_share_the_frozen_budget() {
            for id in ["lama", "osor-sdxl"] {
                let required = if id == "lama" { 3 } else { 40 };
                let mut cached = state();
                cached.prepare(id, required, false, 43);
                assert!(cached.onnx.is_some() && cached.osor.is_some());
                cached.prepare(id, required, false, 42);
                assert_eq!(cached.onnx.is_some(), id == "lama");
                assert_eq!(cached.osor.is_some(), id == "osor-sdxl");
            }
            let mut cached = state();
            cached.prepare("migan-v1", 2, false, 42);
            assert!(cached.onnx.is_none(), "a different ONNX plan must be released before loading");
            assert!(cached.osor.is_some());
        }

        #[test]
        fn batch_admission_preserves_osor_until_the_actual_forward() {
            let mut cached = state();
            cached.admit(false, 43);
            assert!(cached.onnx.is_some() && cached.osor.is_some());
            cached.admit(false, 40);
            assert!(cached.onnx.is_none() && cached.osor.is_some());
            cached.admit(false, 39);
            assert!(cached.empty());
            let mut cached = state();
            cached.admit(true, u64::MAX);
            assert!(cached.empty());
            let mut cached = state();
            cached.prepare("lama", 3, true, u64::MAX);
            assert!(cached.empty());
        }

        #[test]
        fn residency_estimates_fail_closed_on_overflow() {
            let mut cached = state();
            if let Some(onnx) = &mut cached.onnx {
                onnx.bytes = u64::MAX;
            }
            cached.admit(false, u64::MAX);
            assert!(cached.onnx.is_none() && cached.osor.is_some());
            let mut cached = state();
            cached.prepare("lama", u64::MAX, false, u64::MAX);
            assert!(cached.osor.is_none());
        }

        struct Spy(mpsc::Sender<()>);
        impl Drop for Spy {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }

        #[test]
        fn timer_releases_models_without_another_inference_and_waits_for_the_owner() {
            let cache = Arc::new(Cache::<Spy, ()>::new(Duration::from_millis(10)));
            let busy = Arc::new(AtomicBool::new(true));
            let (tx, rx) = mpsc::channel();
            cache.lock().onnx = Some(Resident { id: "lama", bytes: 3, value: Spy(tx) });
            assert!(cache.mark_idle(&busy));
            // Acquire immediately after release but before prepare has its mutex.
            // The expired worker must leave this owner's model intact.
            assert!(busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok());
            let worker_cache = Arc::clone(&cache);
            let worker_busy = Arc::clone(&busy);
            let worker = std::thread::spawn(move || worker_cache.reap(&worker_busy));
            assert!(matches!(rx.recv_timeout(Duration::from_millis(30)), Err(mpsc::RecvTimeoutError::Timeout)));
            assert!(!cache.mark_idle(&busy), "an existing worker must be reused");
            rx.recv_timeout(Duration::from_secs(3)).expect("idle cleanup must happen without another inference");
            worker.join().unwrap();
            assert!(cache.lock().empty());
            assert!(!cache.lock().running);
            assert!(!busy.load(Ordering::Acquire));
        }

        #[test]
        fn releasing_a_new_owner_refreshes_the_idle_deadline() {
            let cache = Cache::<(), ()>::new(Duration::from_secs(300));
            let busy = AtomicBool::new(true);
            cache.lock().onnx = Some(Resident { id: "lama", bytes: 3, value: () });
            let old = Instant::now().checked_sub(Duration::from_secs(600)).unwrap();
            {
                let mut state = cache.lock();
                state.idle_since = Some(old);
                state.running = true;
            }
            assert!(!cache.mark_idle(&busy));
            assert!(cache.lock().idle_since.is_some_and(|since| since > old));
            assert!(!busy.load(Ordering::Acquire));
        }

        #[test]
        fn failed_worker_start_releases_idle_models_but_not_an_active_owner() {
            for active in [false, true] {
                let cache = Cache::<Spy, ()>::new(Duration::from_millis(10));
                let busy = AtomicBool::new(true);
                let (tx, rx) = mpsc::channel();
                cache.lock().onnx = Some(Resident { id: "lama", bytes: 3, value: Spy(tx) });
                assert!(cache.mark_idle(&busy));
                if active {
                    busy.store(true, Ordering::Release);
                }
                cache.failed_start(&busy);
                assert_eq!(cache.lock().empty(), !active);
                assert_eq!(busy.load(Ordering::Acquire), active);
                if active {
                    assert!(cache.mark_idle(&busy), "the next release must retry worker creation");
                    cache.failed_start(&busy);
                }
                rx.recv_timeout(Duration::from_secs(3)).unwrap();
            }
        }
    }
}
