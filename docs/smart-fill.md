# Smart Fill with LaMa

Open Edit → Content-Aware Fill with a selection on a pixel layer. Choose LaMa (AI)
for local filling from the surrounding image, or Classical for the sampling brush,
custom source areas, colour adaptation, rotation, scaling and mirroring controls.
The fill-area lasso and output choices remain available with either method.

LaMa shares Remove's installed model, explicit local-processing opt-in, model
manager, memory allowance and five-minute idle cache. Opening the workspace or
painting a selection never downloads a model. Models… opens the existing setup
and management controls. A ready, enabled LaMa installation is proposed for a new
workspace; saved method choices are preserved where the document supports them.

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
mode. The shared model-management commands remain explicit setup actions.

LaMa fills from image context. Prompted Generative Fill/Expand, broad visual-quality
qualification, other native platforms and learned browser execution remain
separate work.
