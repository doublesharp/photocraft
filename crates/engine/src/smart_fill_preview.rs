//! Reviewed LaMa fill candidates. Pixel ownership precedes the model reservation at drop.
use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};
use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};
use std::sync::Arc;

pub(crate) struct Capture {
    context: crate::removal_preview::Capture,
    sequence: u64,
}
pub(crate) struct Candidate {
    pub doc: Arc<Document>,
    pub active_layer: Option<LayerId>,
    pub crop: Rect,
    pub params: Value,
    pub result: Value,
    pub _reservation: photocraft_ml::RemovalReservation,
}
pub(crate) struct Preview {
    capture: Capture,
    candidate: Candidate,
}
fn bad(cmd: &str, msg: &str) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}
impl Capture {
    pub(crate) fn new(s: &mut Session) -> Result<Self> {
        let context = crate::removal_preview::Capture::new(s)?;
        s.fill_preview_sequence = s.fill_preview_sequence.checked_add(1).ok_or_else(|| bad("edit.contentAwareFill", "fill preview id limit reached"))?;
        Ok(Self { context, sequence: s.fill_preview_sequence })
    }
    fn current(&self, s: &Session) -> bool {
        self.sequence == s.fill_preview_sequence && self.context.current(s)
    }
}
pub(crate) fn publish(s: &mut Session, capture: Capture, candidate: Candidate) -> Result<Value> {
    if !capture.current(s) {
        return Err(bad("edit.contentAwareFill", "the Smart Fill preview context changed; generate a new preview"));
    }
    let mut result = candidate.result.clone();
    result["preview"] = json!(true);
    result["previewId"] = json!(capture.sequence);
    s.fill_preview = Some(Preview { capture, candidate });
    Ok(result)
}
pub(crate) fn apply_candidate(s: &mut Session, capture: Capture, candidate: Candidate) -> Result<Value> {
    if !capture.current(s) {
        return Err(bad("edit.contentAwareFill", "the Smart Fill context changed; generate a new result"));
    }
    s.edit("Content-Aware Fill", |doc, active| {
        *doc = (*candidate.doc).clone();
        *active = candidate.active_layer;
        Ok(())
    })?;
    if let Some(st) = s.active_mut() {
        st.last_damage = Some(candidate.crop);
    }
    Ok(candidate.result.clone())
}
impl Session {
    /// The exact accepted-preview source for display. Release snapshots after discard or replacement.
    pub fn fill_preview_document(&self) -> Option<(Arc<Document>, u64)> {
        self.fill_preview.as_ref().filter(|p| p.capture.current(self)).map(|p| (Arc::clone(&p.candidate.doc), p.capture.sequence))
    }
    /// Read-only status. No inference, files, or model loading occurs here.
    pub fn fill_preview_status(&self) -> Value {
        let Some(p) = self.fill_preview.as_ref() else { return json!({"ready":false}) };
        let mut result = p.candidate.result.clone();
        result["ready"] = json!(p.capture.current(self));
        result["id"] = json!(p.capture.sequence);
        result
    }
    pub(crate) fn clear_stale_fill_preview(&mut self) {
        if self.fill_preview.as_ref().is_some_and(|p| !p.capture.current(self)) {
            self.fill_preview = None;
        }
    }
}
fn validate_id(p: &Value, cmd: &str) -> Result<Option<u64>> {
    let object = p.as_object().ok_or_else(|| bad(cmd, "params must be an object"))?;
    if object.keys().any(|k| k != "id") {
        return Err(bad(cmd, "only an optional preview id is accepted"));
    }
    p.get("id").map(|id| id.as_u64().ok_or_else(|| bad(cmd, "id must be a nonnegative preview id"))).transpose()
}
fn can_apply(s: &Session) -> std::result::Result<(), String> {
    if s.fill_preview.as_ref().is_none_or(|p| !p.capture.current(s)) {
        return Err("no current Smart Fill preview is available".into());
    }
    s.job_conflict("edit.contentAwareFill", true).map_or(Ok(()), Err)
}
fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "fill.preview.apply";
    if s.actions.playing > 0 || s.actions.script_depth > 0 {
        return Err(bad(CMD, "Smart Fill previews require explicit review outside action or script playback"));
    }
    let id = validate_id(p, CMD)?;
    let preview = s.fill_preview.as_ref().ok_or_else(|| bad(CMD, "no current Smart Fill preview is available"))?;
    if id.is_some_and(|id| id != preview.capture.sequence) {
        return Err(bad(CMD, "the preview id does not match the current candidate"));
    }
    if !preview.capture.current(s) {
        s.fill_preview = None;
        return Err(bad(CMD, "the Smart Fill preview is stale"));
    }
    if let Some(gate) = s.authorize {
        gate("edit.contentAwareFill", &preview.candidate.params)?;
    }
    crate::jobs::authorize_render_command("edit.contentAwareFill", &preview.candidate.params)?;
    let preview = s.fill_preview.take().ok_or_else(|| bad(CMD, "no current Smart Fill preview is available"))?;
    let params = preview.candidate.params.clone();
    let mut result = apply_candidate(s, preview.capture, preview.candidate)?;
    result["applied"] = json!(true);
    result["__fillParams"] = params;
    Ok(result)
}
fn discard(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "fill.preview.discard";
    let id = validate_id(p, CMD)?;
    if id.is_some_and(|id| s.fill_preview.as_ref().is_none_or(|p| p.capture.sequence != id)) {
        return Err(bad(CMD, "the preview id does not match the current candidate"));
    }
    s.fill_preview_sequence = s.fill_preview_sequence.checked_add(1).ok_or_else(|| bad(CMD, "fill preview id limit reached"))?;
    let discarded = s.fill_preview.take().is_some();
    // An outstanding fill forward still owns admission until its worker has drained.
    if let Some(job) = s.active_job().filter(|j| j.command == "edit.contentAwareFill") {
        s.cancel_job(job.id);
    }
    Ok(json!({"discarded":discarded}))
}
pub(crate) fn specs() -> Vec<CommandSpec> {
    let spec = |id, label, params, run| CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: |_| Ok(()), run, journal: false };
    vec![
        spec("fill.preview.status", "Smart Fill Preview Status", "{}", |s, p| {
            if !p.as_object().is_some_and(|o| o.is_empty()) {
                return Err(bad("fill.preview.status", "no options are accepted"));
            }
            Ok(s.fill_preview_status())
        }),
        CommandSpec { enabled: can_apply, ..spec("fill.preview.apply", "Apply Smart Fill Preview", r#"{"id":u64?}"#, apply) },
        spec("fill.preview.discard", "Discard Smart Fill Preview", r#"{"id":u64?}"#, discard),
    ]
}
