//! Completed Remove candidates. Inference output and its admission remain owned until the
//! exact candidate is accepted or discarded. Transient preview commands never enter actions.
use std::sync::Arc;

use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

pub(crate) struct Capture {
    source: Arc<Document>,
    revision: u64,
    active_generation: u64,
    context_generation: u64,
    removal_prefs: crate::prefs::GenerativeRemove,
    active_layer: Option<LayerId>,
    target: crate::channel_cmds::ChannelTarget,
    working_rgb: String,
}

impl Capture {
    pub(crate) fn new(s: &Session) -> Result<Self> {
        let st = s.active().ok_or(EngineError::NoDocument)?;
        Ok(Self {
            source: Arc::clone(&st.doc),
            revision: st.revision,
            active_generation: s.active_generation(),
            context_generation: s.remove_context_generation(),
            removal_prefs: s.prefs().generative_remove.clone(),
            active_layer: st.active_layer,
            target: st.channel_view.target,
            working_rgb: s.color.settings.working_rgb.clone(),
        })
    }

    fn current(&self, s: &Session) -> bool {
        s.active().is_some_and(|st| {
            Arc::ptr_eq(&st.doc, &self.source)
                && st.revision == self.revision
                && st.active_layer == self.active_layer
                && st.channel_view.target == self.target
                && crate::float_cmds::floating(st).is_none()
        }) && s.active_generation() == self.active_generation
            && s.remove_context_generation() == self.context_generation
            // Brush, tool and view preferences are persisted after stroke submission. Only
            // removal settings affect this candidate; generation also catches change-and-restore.
            && s.prefs().generative_remove == self.removal_prefs
            && s.color.settings.working_rgb == self.working_rgb
    }
}

/// Fields drop in order: candidate pixels and raw output precede their owned reservation.
pub(crate) struct Candidate {
    pub doc: Arc<Document>,
    pub _content: Box<dyn Send>,
    pub active_layer: Option<LayerId>,
    pub crop: Rect,
    pub params: Value,
    pub result: Value,
}

pub(crate) struct Preview {
    id: u64,
    capture: Capture,
    candidate: Candidate,
}

fn bad(cmd: &str, message: &str) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: message.into() }
}

fn validate_id(p: &Value, cmd: &str) -> Result<Option<u64>> {
    let object = p.as_object().ok_or_else(|| bad(cmd, "params must be an object"))?;
    if object.keys().any(|key| key != "id") {
        return Err(bad(cmd, "only an optional preview `id` is accepted"));
    }
    p.get("id").map(|value| value.as_u64().ok_or_else(|| bad(cmd, "`id` must be a nonnegative preview id"))).transpose()
}

fn refuse_playback(s: &Session, cmd: &str) -> Result<()> {
    if s.actions.playing > 0 || s.actions.script_depth > 0 {
        return Err(bad(cmd, "Remove previews require explicit review and are unavailable during action or script playback"));
    }
    Ok(())
}

pub(crate) fn publish(s: &mut Session, capture: Capture, candidate: Candidate) -> Result<Value> {
    if !capture.current(s) {
        return Err(bad("paint.remove", "the Remove preview context changed while inference was running; the result was discarded"));
    }
    let id = s.removal_preview_sequence.checked_add(1).ok_or_else(|| bad("paint.remove", "preview id limit reached"))?;
    s.removal_preview_sequence = id;
    let mut result = candidate.result.clone();
    result["preview"] = json!(true);
    result["previewId"] = json!(id);
    s.removal_preview = Some(Preview { id, capture, candidate });
    Ok(result)
}

impl Session {
    /// The exact, still-applicable candidate and its session-local display generation.
    /// Callers release their cached snapshot when this returns `None` or a different id.
    pub fn removal_preview_document(&self) -> Option<(Arc<Document>, u64)> {
        self.removal_preview.as_ref().filter(|preview| preview.capture.current(self)).map(|preview| (Arc::clone(&preview.candidate.doc), preview.id))
    }

    /// Read-only preview metadata for frontends and inspection. This does not run a command.
    pub fn removal_preview_status(&self) -> Value {
        let Some(preview) = self.removal_preview.as_ref() else { return json!({"ready":false}) };
        let capture = &preview.capture;
        json!({"ready":capture.current(self), "id":preview.id, "doc":capture.source.id.0,
            "revision":capture.revision, "sourceActiveLayer":capture.active_layer.map(|id| id.0),
            "target":capture.target, "targetLayer":preview.candidate.params.get("layer").and_then(Value::as_u64).or(capture.active_layer.map(|id| id.0)),
            "activeGeneration":capture.active_generation,
            "contextGeneration":capture.context_generation, "workingRgb":capture.working_rgb,
            "hasSelection":capture.source.selection.is_some(),
            "modelId":preview.candidate.result.get("modelId"), "modelSha256":preview.candidate.result.get("modelSha256"),
            "effectiveMethod":preview.candidate.result.get("effectiveMethod"), "regionalPlan":preview.candidate.result.get("regionalPlan"),
            "modelRevisions":preview.candidate.result.get("modelRevisions"),
            "crop":[preview.candidate.crop.x0,preview.candidate.crop.y0,preview.candidate.crop.x1,preview.candidate.crop.y1],
            "removeEffects":preview.candidate.params.get("removeEffects").and_then(Value::as_bool).unwrap_or(false)})
    }

    pub(crate) fn clear_stale_removal_preview(&mut self) {
        if self.removal_preview.as_ref().is_some_and(|preview| !preview.capture.current(self)) {
            self.removal_preview = None;
        }
    }
}

fn can_apply(s: &Session) -> std::result::Result<(), String> {
    if s.removal_preview.as_ref().is_none_or(|preview| !preview.capture.current(s)) {
        return Err("no current Remove preview is available".into());
    }
    if let Some(reason) = s.job_conflict("paint.remove", true) {
        return Err(reason);
    }
    Ok(())
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "remove.preview.apply";
    refuse_playback(s, CMD)?;
    let id = validate_id(p, CMD)?;
    let preview = s.removal_preview.as_ref().ok_or_else(|| bad(CMD, "no current Remove preview is available"))?;
    if id.is_some_and(|id| id != preview.id) {
        return Err(bad(CMD, "the preview id does not match the current candidate"));
    }
    if !preview.capture.current(s) {
        s.removal_preview = None;
        return Err(bad(CMD, "the Remove preview is stale; generate a new preview"));
    }
    // An apply command cannot grant the underlying pixel edit permissions it did not have.
    if let Some(gate) = s.authorize {
        gate("paint.remove", &preview.candidate.params)?;
    }
    crate::jobs::authorize_render_command("paint.remove", &preview.candidate.params)?;
    let preview = s.removal_preview.take().ok_or_else(|| bad(CMD, "no current Remove preview is available"))?;
    let candidate = &preview.candidate;
    s.edit("Remove", |doc, active| {
        *doc = (*candidate.doc).clone();
        *active = candidate.active_layer;
        Ok(())
    })?;
    if let Some(st) = s.active_mut() {
        st.last_damage = Some(candidate.crop);
    }
    let mut result = candidate.result.clone();
    result["applied"] = json!(true);
    // Dispatch records the accepted result as paint.remove, never a transient preview id.
    result["__removeParams"] = candidate.params.clone();
    Ok(result)
}

fn discard(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "remove.preview.discard";
    let id = validate_id(p, CMD)?;
    if id.is_some_and(|id| s.removal_preview.as_ref().is_none_or(|preview| preview.id != id)) {
        return Err(bad(CMD, "the preview id does not match the current candidate"));
    }
    let discarded = s.removal_preview.take().is_some();
    // Also invalidate any in-flight preview captured before this discard.
    s.note_remove_context_change();
    Ok(json!({"discarded":discarded}))
}

pub(crate) fn specs() -> Vec<CommandSpec> {
    let spec = |id, label, params, run| CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: |_| Ok(()), run, journal: false };
    vec![
        spec("remove.preview.status", "Remove Preview Status", "{} → current candidate metadata", |s, p| {
            let object = p.as_object().ok_or_else(|| bad("remove.preview.status", "params must be an object"))?;
            if !object.is_empty() {
                return Err(bad("remove.preview.status", "no options are accepted"));
            }
            Ok(s.removal_preview_status())
        }),
        CommandSpec {
            enabled: can_apply,
            ..spec("remove.preview.apply", "Apply Remove Preview", r#"{"id":u64?} → commit the shown candidate as one undo step"#, apply)
        },
        spec("remove.preview.discard", "Discard Remove Preview", r#"{"id":u64?} → discard candidate without editing"#, discard),
    ]
}
