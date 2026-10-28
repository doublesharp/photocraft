# Local AI Remove

PhotoCraft extends the existing Remove tool with optional local CPU inference. The desktop and CLI builds include the Rust runtime by default; model weights are downloaded separately. A build without `local-ai`, and the browser build, keep Classical Remove.

## Use the tool

1. Select Remove from the J-tool flyout and open **Models…** in its options bar (also Edit → Preferences → Generative Remove Models).
2. Download a model or choose **Install from file** for its exact catalog artifact. The catalog pins its source revision, expected size and SHA-256. A renamed or incompatible ONNX file is rejected.
3. Enable local AI removal, enable the desired process, and select its model. Installation and enabling processing are separate choices.
4. Choose **AI** in the Method control and paint over the object. Processing starts on release, or collect strokes with Remove after each stroke off and press Enter. Esc cancels pending strokes or cancellable work. Create a new layer leaves the source layer unchanged; the finished repair is one undo step.

**Classical** preserves the existing algorithm. Existing commands without `method` also retain that behavior. **Auto** reports its effective method and fallback reason; it currently uses Classical because no learned model has passed the separate broad quality and performance gates for automatic selection. Explicit AI errors never silently retry Classical. A brush stroke never downloads a model.

| Model | Optional download | Runtime |
| --- | ---: | --- |
| LaMa | 208,044,816 bytes (208 MB) | Native Rust tract CPU, fixed 512×512 model context |
| MI-GAN v1 | 29,546,882 bytes (30 MB) | Native Rust tract CPU, exact older MI-GAN pipeline |

The manager exposes the model source, license, expected hash and installation/readiness state. It offers download, cancellation, import and removal. Downloads use HTTPS, bounded streams, staged writes and hash verification. Models remain offline after installation. A running noninterruptible forward can continue draining after logical cancellation, while its resource permit remains held.

## Supported editing contexts

The learned bridge accepts full-colour RGB U8/U16 raster targets in sRGB, with opaque known context. It preserves the document depth and uses fractional selection coverage once at final write. It can fill a transparent hole from opaque context and can write a new layer. HDR/32f, other colour models/profiles, channel/mask targets and complex masked/clipped/effect layer arrangements receive a compatibility error for explicit AI. Sample All Layers currently accepts only simple raster composites. Classical remains available for other contexts.

The operation captures the source, target, selection, brush geometry, model and settings before work begins. The active gesture freezes its effective brush at pointer-down, so later brush changes cannot alter its tinted footprint. Queued UI strokes are discarded when their context changes, including a change and restoration within one action. Oversized rendered footprints are rejected before coverage allocation; the bound includes soft-tip tails, sampled-tip rotation, and scattering. Busy gestures are refused before capture, so a press during processing cannot become a later accidental stroke.

Small repairs read bounded source regions rather than resize the document. The model input pads the crop while preserving aspect ratio; oversized crops are explicitly scaled and the command result reports `inferenceScale` and the crop bounds. The service permits one model operation at a time and caches one model family. Preflight also caps the combined generated brush work at 262,144 estimated dabs, including scattering and dual tips. Live and queued tint use the same check before rendering. Admission includes spare capacity for generated dabs, walker steps and smoothing buffers as well as the model working-set estimate and a conservative pixel-buffer bound against the captured Preferences → Performance → Memory Usage value before allocating crop buffers. This estimate is not an operating-system RSS cap.

## Storage and builds

Models use machine-local app data: Application Support on macOS, LOCALAPPDATA on Windows, and XDG_DATA_HOME (or `~/.local/share`) on Linux. `PHOTOCRAFT_CONFIG_DIR` keeps models beneath that override. Portable mode uses `PhotoCraftData/models` beside the executable and does not silently redirect an unwritable model folder. `PHOTOCRAFT_MODEL_DIR` explicitly selects another real, writable directory, including Scratch during development. Weights and download partials are never embedded in the binary or committed to this repository.

`cargo build -p photocraft` and `cargo build -p photocraft-cli` enable `local-ai`; use `--no-default-features` to omit the native inference backend. `photocraft-ml` itself has no default inference feature. Default compilation and tests need no model files, and normal tests never download weights.

Trusted local CLI run/batch/droplet entry points restore only the saved Generative Remove choices and memory budget from the normal preferences file, using the same portable-directory writability and fallback policy as the desktop; remote automation receives no such file read or implicit processing opt-in. Batch scratch sessions retain that same budget. Corrupt owned model files remain removable in the manager before reinstalling.

Engine clients share `paint.remove` with `method: classical|auto|ai` and optional `modelId`, `remove.capabilities`, and `ai.models.list`. Explicit local model administration uses `ai.models.install`, `ai.models.import` and `ai.models.remove`. Untrusted automation and action/script playback cannot install, import or remove models. The command registry documents parameters and background-job results.

## Qualification and further work

Native LaMa and MI-GAN have matching-artifact numerical checks on a supplied 512×512 scene within one decoded byte value, and real installed-model apply/undo coverage on macOS. The native development app was also exercised through its control channel: the model manager and queued tint were visually inspected, a queued stroke completed real MI-GAN inference, and Undo restored the exact original PNG export. Its isolated model was imported and removed through the native CLI. Those checks do not establish Photoshop parity or broad image quality. Linux/Windows runtime behavior, signed packaging, native download-button/file-dialog journeys, wider quality categories, effects/preview, accelerated inference and learned browser execution require their own qualification.

The release `generative_remove_bench` validates a small repair in a dense 6000×4000 procedural cloud/noise image on an Apple M2 Max (12 logical CPUs, 96 GiB RAM). It reports cold and warm runs, the captured context crop, loading/forward time and sampled whole-process RSS. Runs during the shared compilation window are co-located validation, not calibrated performance budgets or broad image-quality qualification. The example uses already installed catalog artifacts and never downloads weights.

OSOR-SDXL Phase II and Moebius are not available adapters in this delivery. The reusable OSOR inference boundary and the additional output/conditioning requirements are recorded in [the ML README](../crates/ml/README.md). No external Python/model server is used by the shipped Remove path. Prompted Generative Fill/Expand is separate work.
