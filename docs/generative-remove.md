# Local AI Remove

PhotoCraft's desktop and CLI builds include native Rust LaMa, MI-GAN and OSOR support. Weights are separate, optional downloads. Inference stays local after installation; no Python runtime or model server is used. The browser and builds without learned backends retain Classical Remove.

## Choose a setup

The first use of Remove proposes the **Recommended** LaMa + OSOR combination when its backends are available and measured physical memory is at least 48 GiB. Otherwise it proposes LaMa, MI-GAN or Classical in that order, according to backend availability. If physical memory cannot be measured, Recommended remains selectable with its memory guidance. Nothing downloads until the user accepts the linked model licenses and confirms setup. Saved choices are preserved and can be changed through **Models…** in the Remove options bar (also Edit → Preferences → Generative Remove Models).

| Setup | Download if no models are installed | Processing |
| --- | ---: | --- |
| Recommended | 8,623,145,822 bytes (about 8.6 GB) | LaMa for narrow marks; OSOR for objects, with an explicit preview |
| Smaller: LaMa | 208,044,816 bytes (208 MB) | LaMa for every AI repair |
| Compact: MI-GAN v1 | 29,546,882 bytes (30 MB) | MI-GAN v1 for every AI repair |
| Classical | None | Existing content-aware algorithm |
| Custom | User-selected components | Per-model enable and selection controls |

The dialog shows installed model counts and the actual remaining download, excluding models already present from the download total. Preparing a missing OSOR bundle needs about 19.1 GB of temporary disk space and a conservative 40 GiB memory allowance. Recommended explicitly proposes a PhotoCraft memory allowance of at least 48 GiB for preparation and inference headroom, with the current and proposed values shown before confirmation. Raw MiB controls are under **Advanced memory settings**. Machines with less than 48 GiB of measured physical memory should use a smaller setup. OSOR's prepared bundle occupies about 10.6 GB; downloaded inputs are removed after preparation. An installed bundle requires no preparation download or temporary preparation workspace. These are estimates, not operating-system RSS limits.

Setup installs only missing components, sequentially, and continues if the dialog closes. **Models…** shows the active model, stage and cancellation control when reopened. Failure or cancellation keeps completed models and stops the remaining queue; **Finish setup…** opens a proposal for the remaining work and still requires confirmation. Ready requires installed, usable and enabled models and a memory allowance that clears the setup check; individual repairs also need room for their image buffers. A smaller setup with too little allowance directs the user to Preferences → Performance → Memory Usage or another setup before confirmation. Selecting Classical remembers the choice without downloads. Closing first-use setup leaves existing settings intact. A brush stroke never downloads a model, and a missing component produces an actionable error. After restart, selecting Remove offers recovery once for an incomplete saved preset; existing Custom choices remain untouched.

Normal **Download and use…** and **Use this setup…** model actions open the shared setup proposal, including licenses and any memory change, before changing settings or starting transfers. **Advanced** holds per-model enable controls, imports, removal and manual **Use this model** selection. Manual selection enables local AI and switches the Remove method to AI; it does not download weights. Removing a model requires confirmation explaining whether managed files or only an import receipt will be removed.

## Use the tool

Content-Aware Fill can also use LaMa through the same installed model, processing opt-in, memory allowance and idle cache. Its Classical method retains custom sampling controls. See [Smart Fill with LaMa](smart-fill.md). Prompted Generative Fill remains separate work.

Use **Auto** with the selected setup. Recommended groups touching stroke footprints, then routes each independent region using its original source pixels. The automatic rule sends narrow regions (footprint aspect ratio at least 3:1 and minor extent at most 64 pixels) to LaMa and other regions to OSOR. **Repair type** can explicitly select small marks or objects for a stroke. This is a geometry rule informed by the prior manual workflow; broad image-quality qualification remains open.

The result combines local repairs into one candidate. Review it on the canvas and choose **Apply** (Enter) or **Cancel** (Esc). Applying does not repeat inference and creates one undo step. **Shadows and reflections** allows OSOR to repair its bounded context around the painted object, within the current selection; LaMa remains restricted to painted marks. Object cores take precedence where repaired contexts overlap. The predicted OSOR alpha is not applied a second time to its already blended RGB.

**Remove after each stroke** starts processing on release. Turn it off to paint several regions and press Enter to process them together. Each queued stroke retains its brush and repair type. **Create a new layer** preserves the source layer. Esc cancels queued strokes, a ready preview, or cancellable work. A ready preview changes neither the document nor its history.

**Classical** explicitly selects the existing algorithm. Commands without `method` retain their historical Classical behavior. In Custom setup, Auto keeps the legacy Classical fallback; **AI** uses the selected enabled model explicitly. Explicit AI failures never silently retry Classical. Existing saved model configurations remain Custom until the user chooses a setup.

## Supported editing contexts

The learned bridge accepts full-colour RGB U8/U16 raster targets in sRGB, with opaque known context. It preserves the document depth and uses fractional selection coverage once at final write. It can fill a transparent hole from opaque context and can write a new layer. HDR/32f, other colour models/profiles, channel/mask targets and complex masked/clipped/effect layer arrangements receive a compatibility error for explicit AI. Sample All Layers currently accepts only simple raster composites. Classical remains available for other contexts.

The operation captures the source, target, selection, brush geometry, model and settings before work begins. The active gesture freezes its effective brush at pointer-down, so later brush changes cannot alter its tinted footprint. Queued UI strokes are discarded when their context changes, including a change and restoration within one action. Oversized rendered footprints are rejected before coverage allocation; the bound includes soft-tip tails, sampled-tip rotation, and scattering. Busy gestures are refused before capture, so a press during processing cannot become a later accidental stroke.

Small repairs read bounded source regions rather than resize the document. The model input pads the crop while preserving aspect ratio; oversized crops are explicitly scaled and the command result reports `inferenceScale` and the crop bounds. OSOR downsamples only above a 512 px long side, pads to multiples of eight (at least 32 px), and restores RGB and predicted edit alpha to the source crop dimensions. Its RGB already contains latent-space alpha blending; the predicted alpha is not multiplied into that RGB again. The service permits one model operation at a time. It keeps loaded models for five idle minutes after processing and any pending preview finish, so consecutive removals can reuse them. LaMa and OSOR stay loaded together when their combined memory estimates and the image buffers fit the configured memory allowance; a tighter allowance releases the companion before loading the next model. Preflight also caps the combined generated brush work at 262,144 estimated dabs, including scattering and dual tips. Live and queued tint use the same check before rendering. Admission includes spare capacity for generated dabs, walker steps and smoothing buffers as well as the model working-set estimate and a conservative pixel-buffer bound against the captured Preferences → Performance → Memory Usage value before allocating crop buffers. OSOR admission uses a conservative 40 GiB model working-set estimate plus transient buffers. Bundle headers are bounded to 12 GiB of decoded F32 weights, and the native loader enforces that bound on the actual hashed input before tensor allocation. For native evaluation, set Memory Usage to 49,152 MiB (`performance.memoryUsageMb`) to allow headroom for transients; the default budget can reject OSOR before allocation. This estimate is not an operating-system RSS cap.

A preview retains the exact completed candidate and its resource admission until Apply or Cancel; applying it does not run inference again. Source, target, selection, document, working-profile or relevant settings changes invalidate it. A stale or mismatched preview cannot be applied. Recommended hybrid and effects previews require explicit review and are unavailable during action or script playback; ordinary LaMa/MI-GAN action playback is unchanged.

## Downloads and local OSOR bundles

The catalog pins immutable source revisions, exact sizes and SHA-256 values. ONNX files can also be imported through **Install from file…**; incompatible or renamed files are rejected. OSOR setup downloads four safetensors inputs: SDXL Inpainting UNet and VAE, Phase II adapters, and cached prompt embeddings. It merges them in Rust into a new managed bundle. Pickle checkpoints are not accepted. The installer bounds input and decoded sizes before allocation, verifies hashes, supports cancellation before publication, and publishes the manifest last through held directory capabilities. Interrupted preparation does not become an installed model.

OSOR source inputs are pinned to `diffusers/stable-diffusion-xl-1.0-inpainting-0.1@115134f363124c53c7d878647567d04daf26e41e` and `eerie-road/osor-sdxl@a14effb4aabab39cbcbefd7517d2af147ff4d283`. See the catalog and [attribution](../ATTRIBUTION.md) for every artifact and license. CreativeML Open RAIL++-M includes use restrictions; the Rust library's MIT license does not replace model terms. PhotoCraft ships no weights.

**Install from folder…** remains available for an existing OSOR bundle containing `manifest.json`, `unet.safetensors`, `vae.safetensors` and `prompt.safetensors`. Import hashes all four files and stores a small reference receipt without copying multi-GB weights. Keep that directory at its imported path without symlinks. Moving or changing it makes the reference unavailable. Import verifies identity, not neural-graph validity or image quality. **Remove model** removes a downloaded managed bundle, or only the receipt for an externally imported bundle. It never removes the external source directory.

The service rehashes model files before inference, including warm runs. It admits one active operation. The hybrid batch still accepts the maximum sequential family working set, and uses extra allowance to retain both warm models only when their combined estimates plus all transient buffers fit. Idle cleanup cannot run during processing or while a preview owns admission. Integrity verification still runs on warm models; retaining them avoids rebuilding their loaded runtime, and does not remove hashing time or inference work. OSOR cancellation reaches cooperative checkpoints inside model blocks; already queued device work still drains before its resource permit is released. ONNX forwards remain noninterruptible and retain the permit until completion. See [the ML contract](../crates/ml/README.md) for capability and same-user filesystem-race limitations.

## Storage, builds and commands

Models use machine-local app data: Application Support on macOS, LOCALAPPDATA on Windows, and XDG_DATA_HOME (or `~/.local/share`) on Linux. `PHOTOCRAFT_CONFIG_DIR` keeps models beneath that override. Portable mode uses `PhotoCraftData/models` beside the executable and does not redirect an unwritable model directory. `PHOTOCRAFT_MODEL_DIR` explicitly selects a different real directory, including Scratch for development.

Default desktop and CLI builds enable `local-ai` and `osor`. LaMa prefers supported Metal operations on macOS, with the same model on CPU when GPU preparation or its memory requirement is unavailable. Some graphs combine GPU and CPU operations. MI-GAN remains on CPU pending separate GPU qualification. OSOR uses Metal/F32 on macOS and CPU/F32 on other native targets. `--no-default-features` omits learned backends; `--features local-ai` selects the smaller ONNX-only runtime. `photocraft-ml` itself has no default inference feature. Builds and ordinary tests require no model files and never download them. Follow Scratch output rules in `AGENTS.md`.

Trusted local CLI run/batch/droplet entry points restore saved removal choices and memory budget. Remote automation receives no implicit file-read or processing opt-in. Engine clients use `remove.setup {workflow,memoryUsageMb?}` to configure only, followed by explicit `ai.models.install {modelId}` commands for downloads. `remove.capabilities` and `ai.models.list` expose the selected workflow, components, preparation estimates, license links and readiness. `ai.models.list.lastExecution` separately reports the completed runtime/device, graph placement when available, and CPU fallback reason. Models shows the last run as GPU, mixed GPU/CPU, or CPU; compiled support alone does not establish GPU execution.

`paint.remove` accepts `method:classical|auto|ai`, optional `modelId`, and `repairType:auto|detail|object` globally or per stroke. Recommended Auto always creates a preview. Explicit `removeEffects:true` requires review. Use `remove.preview.status {}`, then `remove.preview.apply {id?}` or `remove.preview.discard {id?}`; supplying the returned ID binds the action to that exact candidate. Results retain the regional plan and model revisions. Untrusted automation and action/script playback cannot administer models or apply unreviewed hybrid/effects repairs.

## Qualification and further work

Focused engine and UI tests cover regional routing, original-source sampling, selection coverage, precedence, one-step undo, configuration, cancellation and no implicit downloads. Current setup/manager layouts were checked with five offscreen fixtures, including fresh, partial, ready and small-window states. Installed LaMa/MI-GAN retain their separate numerical and macOS application checks.

On 2026-10-28, osor-rs v0.2.0 (`1b299a1d`) passed the retained managed-bundle Metal/F32 service smoke: cold and warm 101×47 input restored finite, source-sized RGB/alpha with zero repeat difference and retained-output admission. An isolated, preconfigured default native macOS build on a procedural 256×192 image passed unrelated preference/brush/view writes during OSOR, unchanged source before Apply, fullscreen preview, Escape discard and main-view recovery, Enter Apply, exact PNG restoration after Undo, warm in-flight cancellation, admission drain, a successful next removal and clean exit. Both native screenshots were visually inspected. These native checks used already installed models and did not exercise a new download or first-use installation.

The published v0.2.0 CLI separately replayed one 704×368 photo against the retained independent PyTorch/MPS reference, preserving the original mask, both captured noise tensors, managed bundle, F32 Metal execution and frozen tolerances. All 18 stages passed; decoded RGB maximum absolute error was 0.0009236335754394531 and RMS error was 0.000008959910624121942. No new reference was generated. This is a bounded numerical replay, distinct from the application's 512 px long-side bridge and broad visual-quality acceptance. Stage-synchronized CLI timings do not qualify normal application performance. Scratch receipts retain source/binary hashes, artifact identity, commands and results.

Earlier checks with the previous `f9d4b2f9` pin remain historical. Full managed preparation from the four pinned local streams reproduced the UNet/VAE hashes, cleaned up inputs and took 66.14 seconds; this did not exercise an entire 8.4 GB HTTPS transfer. The actual Rust HTTPS transport separately matched the pinned prompt bytes, and trusted CLI installation downloaded the 30 MB MI-GAN artifact. That earlier native build completed first-use Classical and Compact configuration and mixed LaMa/OSOR preview/apply/Undo/discard. Full preparation and HTTPS transfer were not rerun for v0.2.0.

Historical `f9d4b2f9` release measurements on a dense procedural 6000×4000 image processed two independent regions with LaMa and OSOR in 22.98 seconds on the first run and 23.19–25.76 seconds on repetitions, restoring the exact document with Undo. Sampled whole-process RSS peaked at about 12.0 GiB for these small crops; it does not qualify larger crops or establish a physical-memory cap. No current-pin 24 MP release measurement was performed.

The historical `1b299a1d` quick performance gate exited successfully, but marked the shared-host run as contended. Regression comparison was skipped because the published baseline is from an M4 Pro and this host is an M2 Max. Quick mode uses smaller documents and does not apply full-size performance budgets. No baseline or budget was changed; this run does not establish a performance improvement.

New setup strings use English fallback where translations are missing. These checks do not establish Photoshop parity, broad removal quality, signed packaging, or Linux/Windows runtime acceptance. Moebius refinement has no adapter; the prior broad pilot did not justify adding it unconditionally. Prompted Generative Fill/Expand and learned browser execution remain separate work.

The current Cargo package/library is `osor`, pinned to published commit `94584f64`
in the unchanged `doublesharp/osor-rs` Git repository. PhotoCraft keeps this
immutable dependency pin when upstream main changes.
Initial offline validation seeded the existing Cargo Git cache from the supplied
local checkout; that remains historical provenance, separate from remote resolution.

On 2026-10-28, the `94584f64` CLI separately passed all 18 frozen Metal/F32 replay
stages for the retained 704×368 photo, using the unchanged independent reference,
mask, noise tensors, managed bundle and tolerances. Decoded RGB maximum absolute
error was 0.0006973743438720703 and RMS error was 0.000008269988458171247. This is
bounded numerical evidence, distinct from the historical application lifecycle and
installer checks above; it makes no broad quality or throughput claim.

The `94584f64` retained-bundle Metal/F32 service smoke also passed cold and warm
101×47 input: finite source-sized RGB/alpha, zero repeat difference, no warm reload
and retained-output admission. It used existing weights and exercised no download
or model preparation. These bridge checks remain separate from broad removal
quality and from the historical native application lifecycle checks.

LaMa's 2026-10-29 Metal check passed the retained 512×512 oracle and direct CPU
comparison within the unchanged one-byte threshold. Its graph mixes GPU and CPU
operations; warm hardware timestamps confirmed GPU dispatch. Cache switching,
retained-output ownership and cancellation/drain passed. Timings are shared-host
diagnostics, and this scene does not establish broad quality or a performance
budget. MI-GAN remains on CPU pending separate GPU qualification.
