# Local AI Remove

PhotoCraft's desktop and CLI builds include native Rust LaMa, MI-GAN and OSOR support. Weights are separate, optional downloads. Inference stays local after installation; no Python runtime or model server is used. The browser and builds without learned backends retain Classical Remove.

## Choose a setup

The first use of Remove offers the **Recommended** LaMa + OSOR combination. Nothing downloads until the user accepts the linked model licenses and confirms setup. The choice is remembered and can be changed through **Models…** in the Remove options bar (also Edit → Preferences → Generative Remove Models).

| Setup | Download if no models are installed | Processing |
| --- | ---: | --- |
| Recommended | 8,623,145,822 bytes (about 8.6 GB) | LaMa for narrow marks; OSOR for objects, with an explicit preview |
| Smaller: LaMa | 208,044,816 bytes (208 MB) | LaMa for every AI repair |
| Compact: MI-GAN v1 | 29,546,882 bytes (30 MB) | MI-GAN v1 for every AI repair |
| Classical | None | Existing content-aware algorithm |
| Custom | User-selected components | Per-model enable and selection controls |

Full OSOR preparation needs about 19.1 GB of temporary disk space and a conservative 40 GiB memory allowance. The recommended setup explicitly proposes a 49,152 MiB PhotoCraft memory budget for preparation and inference headroom. It cannot increase that setting without confirmation. Machines with less than 48 GiB of measured physical memory should use a smaller setup. OSOR's prepared bundle occupies about 10.6 GB; downloaded inputs are removed after preparation. These are estimates, not operating-system RSS limits.

Setup installs only missing components, sequentially, and continues if the dialog closes. Failure or cancellation stops the remaining queue; retry is explicit. Selecting Classical remembers the choice without downloads. Closing first-use setup leaves existing settings intact. A brush stroke never downloads a model, and a missing component produces an actionable error.

## Use the tool

Use **Auto** with the selected setup. Recommended groups touching stroke footprints, then routes each independent region using its original source pixels. The automatic rule sends narrow regions (footprint aspect ratio at least 3:1 and minor extent at most 64 pixels) to LaMa and other regions to OSOR. **Repair type** can explicitly select small marks or objects for a stroke. This is a geometry rule informed by the prior manual workflow; broad image-quality qualification remains open.

The result combines local repairs into one candidate. Review it on the canvas and choose **Apply** (Enter) or **Cancel** (Esc). Applying does not repeat inference and creates one undo step. **Shadows and reflections** allows OSOR to repair its bounded context around the painted object, within the current selection; LaMa remains restricted to painted marks. Object cores take precedence where repaired contexts overlap. The predicted OSOR alpha is not applied a second time to its already blended RGB.

**Remove after each stroke** starts processing on release. Turn it off to paint several regions and press Enter to process them together. Each queued stroke retains its brush and repair type. **Create a new layer** preserves the source layer. Esc cancels queued strokes, a ready preview, or cancellable work. A ready preview changes neither the document nor its history.

**Classical** explicitly selects the existing algorithm. Commands without `method` retain their historical Classical behavior. In Custom setup, Auto keeps the legacy Classical fallback; **AI** uses the selected enabled model explicitly. Explicit AI failures never silently retry Classical. Existing saved model configurations remain Custom until the user chooses a setup.

## Supported editing contexts

The learned bridge accepts full-colour RGB U8/U16 raster targets in sRGB, with opaque known context. It preserves the document depth and uses fractional selection coverage once at final write. It can fill a transparent hole from opaque context and can write a new layer. HDR/32f, other colour models/profiles, channel/mask targets and complex masked/clipped/effect layer arrangements receive a compatibility error for explicit AI. Sample All Layers currently accepts only simple raster composites. Classical remains available for other contexts.

The operation captures the source, target, selection, brush geometry, model and settings before work begins. The active gesture freezes its effective brush at pointer-down, so later brush changes cannot alter its tinted footprint. Queued UI strokes are discarded when their context changes, including a change and restoration within one action. Oversized rendered footprints are rejected before coverage allocation; the bound includes soft-tip tails, sampled-tip rotation, and scattering. Busy gestures are refused before capture, so a press during processing cannot become a later accidental stroke.

Small repairs read bounded source regions rather than resize the document. The model input pads the crop while preserving aspect ratio; oversized crops are explicitly scaled and the command result reports `inferenceScale` and the crop bounds. OSOR downsamples only above a 512 px long side, pads to multiples of eight (at least 32 px), and restores RGB and predicted edit alpha to the source crop dimensions. Its RGB already contains latent-space alpha blending; the predicted alpha is not multiplied into that RGB again. The service permits one model operation at a time and caches one model family. Preflight also caps the combined generated brush work at 262,144 estimated dabs, including scattering and dual tips. Live and queued tint use the same check before rendering. Admission includes spare capacity for generated dabs, walker steps and smoothing buffers as well as the model working-set estimate and a conservative pixel-buffer bound against the captured Preferences → Performance → Memory Usage value before allocating crop buffers. OSOR admission uses a conservative 40 GiB model working-set estimate plus transient buffers. Bundle headers are bounded to 12 GiB of decoded F32 weights, and the native loader enforces that bound on the actual hashed input before tensor allocation. For native evaluation, set Memory Usage to 49,152 MiB (`performance.memoryUsageMb`) to allow headroom for transients; the default budget can reject OSOR before allocation. This estimate is not an operating-system RSS cap.

A preview retains the exact completed candidate and its resource admission until Apply or Cancel; applying it does not run inference again. Source, target, selection, document, working-profile or relevant settings changes invalidate it. A stale or mismatched preview cannot be applied. Recommended hybrid and effects previews require explicit review and are unavailable during action or script playback; ordinary LaMa/MI-GAN action playback is unchanged.

## Downloads and local OSOR bundles

The catalog pins immutable source revisions, exact sizes and SHA-256 values. ONNX files can also be imported through **Install from file…**; incompatible or renamed files are rejected. OSOR setup downloads four safetensors inputs: SDXL Inpainting UNet and VAE, Phase II adapters, and cached prompt embeddings. It merges them in Rust into a new managed bundle. Pickle checkpoints are not accepted. The installer bounds input and decoded sizes before allocation, verifies hashes, supports cancellation before publication, and publishes the manifest last through held directory capabilities. Interrupted preparation does not become an installed model.

OSOR source inputs are pinned to `diffusers/stable-diffusion-xl-1.0-inpainting-0.1@115134f363124c53c7d878647567d04daf26e41e` and `eerie-road/osor-sdxl@a14effb4aabab39cbcbefd7517d2af147ff4d283`. See the catalog and [attribution](../ATTRIBUTION.md) for every artifact and license. CreativeML Open RAIL++-M includes use restrictions; the Rust library's MIT license does not replace model terms. PhotoCraft ships no weights.

**Install from folder…** remains available for an existing osor-rs bundle containing `manifest.json`, `unet.safetensors`, `vae.safetensors` and `prompt.safetensors`. Import hashes all four files and stores a small reference receipt without copying multi-GB weights. Keep that directory at its imported path without symlinks. Moving or changing it makes the reference unavailable. Import verifies identity, not neural-graph validity or image quality. **Remove model** removes a downloaded managed bundle, or only the receipt for an externally imported bundle. It never removes the external source directory.

The service rehashes model files before inference, including warm runs. It admits one active operation and one warm model family; the hybrid batch uses the maximum sequential family working set rather than retaining both models. A noninterruptible forward may continue draining after logical cancellation, retaining its resource permit until completion. See [the ML contract](../crates/ml/README.md) for capability and same-user filesystem-race limitations.

## Storage, builds and commands

Models use machine-local app data: Application Support on macOS, LOCALAPPDATA on Windows, and XDG_DATA_HOME (or `~/.local/share`) on Linux. `PHOTOCRAFT_CONFIG_DIR` keeps models beneath that override. Portable mode uses `PhotoCraftData/models` beside the executable and does not redirect an unwritable model directory. `PHOTOCRAFT_MODEL_DIR` explicitly selects a different real directory, including Scratch for development.

Default desktop and CLI builds enable `local-ai` and `osor`. OSOR uses Metal/F32 on macOS and CPU/F32 on other native targets. `--no-default-features` omits learned backends; `--features local-ai` selects the smaller ONNX-only runtime. `photocraft-ml` itself has no default inference feature. Builds and ordinary tests require no model files and never download them. Follow Scratch output rules in `AGENTS.md`.

Trusted local CLI run/batch/droplet entry points restore saved removal choices and memory budget. Remote automation receives no implicit file-read or processing opt-in. Engine clients use `remove.setup {workflow,memoryUsageMb?}` to configure only, followed by explicit `ai.models.install {modelId}` commands for downloads. `remove.capabilities` and `ai.models.list` expose the selected workflow, components, preparation estimates, license links and readiness.

`paint.remove` accepts `method:classical|auto|ai`, optional `modelId`, and `repairType:auto|detail|object` globally or per stroke. Recommended Auto always creates a preview. Explicit `removeEffects:true` requires review. Use `remove.preview.status {}`, then `remove.preview.apply {id?}` or `remove.preview.discard {id?}`; supplying the returned ID binds the action to that exact candidate. Results retain the regional plan and model revisions. Untrusted automation and action/script playback cannot administer models or apply unreviewed hybrid/effects repairs.

## Qualification and further work

Focused engine and UI tests cover regional routing, original-source sampling, selection coverage, precedence, one-step undo, configuration, cancellation and no implicit downloads. Installed LaMa/MI-GAN and OSOR have separate numerical and macOS application checks. The pinned download recipe's new prompt passed all 18 captured stages against a fresh independent PyTorch/MPS reference on one 704×368 photo, using the prior fixed tolerances. Decoded RGB maximum absolute error was 0.0009242 and RMS error was 0.000008952. Full managed preparation from the four pinned local streams reproduced the UNet/VAE hashes, cleaned up inputs and took 66.14 seconds; this does not exercise an entire 8.4 GB HTTPS transfer. The actual Rust HTTPS transport separately matched the pinned prompt bytes, and trusted CLI installation downloaded the 30 MB MI-GAN artifact. A default native macOS build completed first-use Classical and Compact configuration, mixed LaMa/OSOR preview, unchanged source before Apply, exact PNG restoration after Undo, discard and clean exit. Both setup sizes and the native preview were visually inspected. Scratch receipts record the exact source, artifact and run boundaries.

A release check on a dense procedural 6000×4000 image processed two independent regions with LaMa and OSOR in 22.98 seconds on its first run and 23.19–25.76 seconds on repetitions. Sampled whole-process RSS peaked at about 12.0 GiB for these small crops; it does not qualify larger crops or establish a physical-memory cap. Every run restored the exact original document with Undo. The quick performance gate exited successfully; comparison with the published regression baseline was skipped because the baseline machine is an M4 Pro and this host is an M2 Max. No baseline or budget was changed.

New setup strings use English fallback where translations are missing. These checks do not establish Photoshop parity, broad removal quality, signed packaging, or Linux/Windows runtime acceptance. Moebius refinement has no adapter; the prior broad pilot did not justify adding it unconditionally. Prompted Generative Fill/Expand and learned browser execution remain separate work.
