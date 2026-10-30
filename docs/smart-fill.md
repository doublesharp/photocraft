# Smart Fill with LaMa

Open Edit → Content-Aware Fill with a selection on a pixel layer. Choose LaMa (AI)
for local filling from the surrounding image, or Classical for the sampling brush,
custom source areas, colour adaptation, rotation, scaling and mirroring controls.
The fill-area lasso and output choices remain available with either method.

LaMa shares Remove's installed model, explicit local-processing opt-in, model
manager, memory allowance and five-minute idle cache. Opening the workspace or
painting a selection never downloads a model. Models… opens the existing setup
and management controls. A ready, enabled LaMa installation is selected for each new
workspace. Classical remains an explicit choice within that workspace; its sampling
and adaptation settings are remembered. Unsupported contexts use Classical.

AI fill supports RGB 8/16-bit sRGB raster contexts with opaque known surroundings.
Sample All Layers can supply opaque surroundings for an empty target. The image
context is bounded and may be scaled to the model's resolution; result metadata
reports the crop and inference scale. HDR, other profiles/modes and unsupported
targets produce an actionable error. Memory checks precede model preprocessing.

The preview leaves the document and history unchanged. Apply or OK commits the
exact reviewed result without running LaMa again, as one Undo step. Current Layer,
New Layer and Duplicate Layer output preserve selection coverage and the source
outside the fill area. Cancel releases the candidate; a noninterruptible LaMa
forward drains before its memory reservation is released. Changing the fill area
supersedes the old preview. Overlay, zoom and unrelated brush/view changes reuse
the same result.

## Compare repairs

The two Method choices, **Classical** and **LaMa (AI)**, are directly visible.
The original stays on the left and the preview title identifies the method on
the right. Switch between them to inspect texture and edges at the same zoom.

A completed LaMa result is retained while you inspect Classical. Returning to
LaMa restores those exact pixels without another inference run. A Classical
sampling error does not discard the retained LaMa result. Classical-only sampling
and adaptation settings are remembered without becoming AI inputs.

Changing the fill area, output, sampled layers, source or relevant model settings
invalidates the AI candidate. Switching away during an unfinished AI operation
cancels it instead of retaining a partial result. Apply uses the currently chosen
method. Cancel, closing the workspace or opening Models releases the retained
candidate. Comparison itself changes neither the document nor its history.

## Automation

The existing command keeps Classical as its default for compatible scripts:

```json
{"method":"ai","preview":true,"output":"new","sampleAllLayers":true}
```

Pass this to `edit.contentAwareFill`. It uses LaMa; another model ID is refused.
Read `fill.preview.status {}` and apply or discard the returned candidate with
`fill.preview.apply {"id":42}` or `fill.preview.discard {"id":42}`. Use the actual
returned ID. Preview IDs are transient; accepted actions record the underlying
fill parameters. `preview:false` runs and commits the explicit AI command directly.

The Content-Aware Fill workspace exposes `method:"ai"|"classical"` through its
existing `ui` control object. Classical-only sampling inputs are rejected in AI
mode. The shared model-management commands remain explicit setup actions. The workspace
reports `previewId` only when its AI preview is selected, and
`retainedAiPreviewId` also while viewing Classical. `previewReady` describes the
currently displayed method, including a computed image awaiting texture upload.

LaMa fills from image context. Prompted Generative Fill/Expand, broad visual-quality
qualification, other native platforms and learned browser execution remain
separate work.

## Qualification

On 2026-10-28, the independent Smart Fill branch passed 12 focused engine and
12 UI tests. These cover output choices and RGB depths, fractional coverage,
source transparency locks, preview ownership, cancellation/drain, unsupported
contexts, memory refusal and Apply/OK lifecycle.

An aggregate macOS development build used the existing LaMa weights and an
8 GiB configured allowance on a procedural 256×192 document. Preview and
unrelated preferences/overlay changes preserved the source and candidate; Apply
used that candidate once, OK after Apply added no fill, and Undo restored the
exact saved PNG. Warm reopening did not reload LaMa. In-flight cancellation
drained admission, preserved the source and allowed another fill. Classical
controls and the shared Models manager were also exercised. Native-resolution
images at 1280×900 and 1000×700 were inspected with all controls visible.

These checks used retained weights and a private profile. They do not establish
fresh-install download/preparation, broad image quality, other native platforms,
signed packaging or full-size performance budgets.


On 2026-10-29, the comparison workspace passed 35 focused UI checks. The complete
UI crate passed 4,292 library and 134 integration tests serially (22 ignored).
An initial parallel run had one Window-menu structure-key failure; that test
passed in isolation, and the retained pre-change executable passed its parallel
suite. No menu-code correction is claimed.

An isolated native macOS build compared both methods on three fixed public-domain
photo crops with synthetic holes: forest texture, a dam wall and a thin line.
LaMa ran on Metal. Method switches preserved the source and the exact LaMa
candidate; Apply preserved every outside-mask pixel and Undo restored each input
for both methods. Large and small native screenshots were inspected. These
three reconstruction cases establish workflow and preservation behavior, not
captured object-removal quality, a universal winning method or a speed budget.
