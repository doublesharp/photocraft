# Local removal runtime

`photocraft-ml` accepts a bounded encoded-sRGB crop and a white-is-hole mask. The engine owns crop selection, colour conversion, document edits and history. The service accepts only catalog model IDs, pins model bytes and SHA-256, and never downloads during inference. The app injects a storage directory and an optional downloader. Native inference uses `tract-onnx` 0.23.9 behind `native-inference`; wasm reports it unavailable.

The model store opens each directory component without following links and retains directory capabilities for the entire operation. Opens, staging, verification, no-clobber publication, removal and cleanup use those handles; a replaced pathname cannot redirect them outside the held store. File opens reject symlinks and nonregular entries. Directory and file identity checks detect the tested replacement points and interrupted installs clean up their own staged bytes through the held directory. Failed verification, cancellation and stream errors preserve an existing install. Removal moves the selected entry into a randomly named staging directory (mode 0700 on Unix), checks the moved handle, and restores a raced replacement without overwriting another entry; if restoration is impossible, the error names the retained recovery directory. Linux and Windows race behavior still needs native qualification. `management_available` reports whether a directory is configured; it does not probe writability, so a read-only directory is reported when a management operation is attempted.

A writable model directory is not a security boundary against an actively hostile process with the same filesystem authority. Portable capability-relative hard-link and unlink operations still select a leaf by name, not atomically by open-file identity; replacement between a check and a syscall, or while cleanup is running, can change the affected leaf. The bounded cleanup search also cannot find a partial moved outside the held directory or beyond 4,096 entries. Use a directory writable only by the trusted user, and do not claim race-free conditional publication or deletion against that user. Subsequent model loads always rehash the opened file.

The process-wide permit admits at most one active operation. The warm cache retains at most one ONNX plan and one OSOR model together when their catalog working-set estimates fit the captured memory budget after subtracting all admitted transient buffers. A single background worker releases both after five idle minutes. The idle interval starts again when the operation or retained-preview reservation releases admission; the worker cannot evict models while that ownership is held. Insufficient headroom evicts a companion before loading the requested model. A failed cleanup-thread start evicts idle models or retries after the active owner releases. Model bytes are rehashed before each inference, even when a compiled plan is warm, because file metadata can be restored after a rewrite. Cold loads parse the same bounded byte snapshot that was hashed, with a transient allocation up to the catalog artifact size. Cancellation cannot interrupt a tract forward; admission stays held until the worker returns. The engine also retains it while an output waits to be applied or discarded. Model installation calls `RemovalControl::begin_irreversible` immediately before no-clobber publication and ignores cancellation after that point. Hosts with asynchronous cancellation must override the default hook with an atomic gate that rejects new cancellation once commit begins; the engine `JobCtx` connects this hook to its commit gate. The crop has a 16,777,216-pixel input limit.

## Native ONNX compute

ONNX inference defaults to `OnnxBackend::Auto`: prefer Metal for LaMa on macOS when the
pinned graph has supported GPU operations and its working set fits the captured
budget. Unsupported operations can remain on CPU within the same graph. Other
native platforms use CPU. MI-GAN Auto remains on CPU pending separate GPU
qualification. GPU initialization, lowering or drained-forward failure
falls back to the **same model** on CPU; cancellation never starts a retry.
`with_onnx_backend(OnnxBackend::Cpu)` forces CPU, while `Metal` requires GPU support
and reports an error for qualification rather than hiding a fallback.

`last_execution()` reports only a completed result: runtime/device, actual prepared
graph placement where available, and a CPU fallback reason. `ai.models.list` exposes
it as `lastExecution`; Models shows GPU, mixed CPU/GPU or CPU processing for the last
run. Node counts describe placement, not percentages of work or elapsed time.
OSOR diagnostics use its actual compute adapter and returned tensor devices.

The ONNX cache distinguishes backend requests and fingerprints. The catalog
estimate plus the prepared Metal arena must fit budget minus transients; a warm
companion is evicted before allocating that arena. Tight budgets use CPU, and a
later allowance large enough for the arena retries GPU preparation. Other CPU
fallback plans remain warm until eviction, backend change or five idle minutes.
Every model read still verifies its pinned hash. GPU synchronization finishes
before cancellation releases ownership, output decoding or a CPU retry. The Metal
context's kernel cache is separate from retained model weights. Wasm has no native
inference dependency or learned GPU execution.

## Early resource admission API

`RemovalService::admit(&self, model_id: &str, transient_bytes: u64, budget_bytes: u64) -> Result<RemovalPermit<'_>>` checks the catalog model's `memory_mib` working-set estimate (MiB × 1,048,576) plus the caller's conservative bound for all overlapping transient buffers with checked arithmetic against the frozen byte budget. An overflow or excess returns an error before the caller allocates those buffers; an already-active process-wide operation returns `Error::Busy`. The estimate is an admission policy, not an operating-system-enforced allocation cap. An incompatible ONNX plan is evicted before admission returns. OSOR and an ONNX companion can remain resident only within the same frozen budget; each inference rechecks the estimates before loading.

Hold the returned permit across selection/hole/region and crop construction, inference, and any noninterruptible worker drain. Bind it mutably and call `permit.infer(&input, control) -> Result<RemovalOutput>` once. When output must outlive the worker, consume the permit with `permit.into_reservation()` and keep the owned `RemovalReservation` with the output until it is applied or discarded. The reservation can outlive the service; dropping it releases process-wide admission. Arrange ownership so the output buffers are dropped before the reservation, including error, cancellation, and discard paths. Dropping an untransferred permit also releases admission. The model ID is fixed at admission.

`RemovalService::infer` remains available for callers that have already constructed an input and delegates with an unbounded budget; it releases its permit before returning, so callers retaining output under admission must use the permit and reservation API. Account for all live buffers and the returned output in `transient_bytes`; inference rejects a bound smaller than the actual f32 input RGB, mask and output RGB (28 bytes per pixel for ONNX; 32 including learned alpha for OSOR). WebAssembly admission and inference report `Error::Unavailable`.

`admit_models(&[model_ids], transient_bytes, budget_bytes)` returns a
`RemovalBatchPermit` for at most 32 sequential forwards. It admits the largest
model working set plus the caller's complete retained-buffer estimate. Each
`permit.infer(model_id, input, control)` checks membership and prepares only the
model actually requested. Batch admission preserves eligible warm models even
when the first listed model is not needed. Companion models remain resident only
when their combined estimates fit budget minus transients; tighter budgets retain
the previous sequential-eviction behavior. A failed forward ends the batch; the lease
still remains held. Capture replay identity with `permit.model_sha256(model_id)`
and transfer ownership with `permit.into_reservation()` through preview apply or
discard. Retained buffers must be dropped before that reservation.

## Explicit OSOR preparation

The compiled OSOR recipe downloads four digest-pinned safetensors files totaling
8,415,101,006 bytes. `ModelSpec.artifacts` and `model_artifact(id)` expose their
immutable source URLs, sizes, digests and license links. The application must
provide `ModelDownloader::open_artifact`; its existing single-file `open` method
continues to serve ONNX models. Model metadata separately discloses estimated
installed bytes, preparation workspace and preparation memory.

The explicit model-management command calls
`install_with_memory_budget("osor-sdxl", frozen_budget_bytes, control)`.
Preparation requires an estimated 40 GiB of memory and 19.1 GB of temporary disk
space, including the roughly 10.61 GB installed bundle. The unbudgeted `install`
method rejects OSOR before network or storage access. Inference never downloads.
The service retains its global lease through all transfer, conversion and cleanup
work, and evicts warm inference models before preparation.

Inputs and output files are exclusively created through held directory
capabilities. Every input stream and the staged file are checked against compiled
lengths and SHA-256 values. Pure Rust conversion consumes held files and an
output sink, never Python, pickle or an ambient output path. Its manifest is
complete before a managed reference receipt is published without clobbering an
existing installation. Publication also checks the held directory identity and
the manifest returned by the completed conversion. The converter's 24 GiB decoded
input cap is separate from the 12 GiB resident-bundle cap and the 40 GiB
preparation estimate. Cancellation is cooperative between conversion stages;
a synchronous stage keeps admission until it returns. Failed preparation cleans
its owned files while preserving an existing valid receipt. Successful
preparation removes raw inputs and keeps a copy of the model license. Removing a
downloaded model removes its fixed owned bundle files and receipt; imported
external bundles retain their original weights.

The mirror's fixed prompt is F32 and differs numerically from the earlier local
F16-export qualification. On 2026-10-28, osor-rs development revision `1b299a1d`
replayed the retained managed bundle on Metal/F32 against the frozen independent
PyTorch/MPS reference for one 704×368 photo. The original mask and both captured
noise tensors were reused; all 18 stages passed the unchanged frozen limits.
Decoded RGB maximum absolute error was 0.0009236335754394531 and RMS error was
0.000008959910624121942. The reference was retained from the previous qualification;
no new PyTorch reference run was performed. This qualifies that bounded CLI scene,
separately from the application's 512 px long-side bridge and broader removal quality.
Stage-synchronized CLI timings are diagnostic, not normal application performance.

That historical `1b299a1d` pin also passed the retained-bundle Metal/F32 service smoke: cold and
warm 101×47 input restored finite, source-sized RGB/alpha with zero repeat difference
and retained-output admission. A separate, preconfigured native macOS app run on a
procedural 256×192 image passed unrelated preference/brush/view writes during OSOR,
preview source preservation, Escape discard and main-view recovery, Enter Apply,
exact PNG Undo, warm in-flight cancellation, admission drain, a successful next
removal and clean private-process exit. Full model preparation and application HTTPS
transfer were not rerun for `1b299a1d`; the 66.14-second preparation and 24 MP release
measurements in the application documentation remain historical `f9d4b2f9` results.

The ignored
`real_osor_installer_converts_pinned_local_streams_into_managed_bundle` test
requires explicit `PHOTOCRAFT_ML_OSOR_INSTALL_ROOT`,
`PHOTOCRAFT_ML_OSOR_INSTALL_CACHE` and `PHOTOCRAFT_ML_OSOR_INSTALL_PROMPT` paths.
It exercises actual conversion from pinned local streams and preserves its managed
bundle for subsequent qualification. It does not test application HTTPS transport
or infer image-quality acceptance.

## Optional real-model check

The ignored `real_recommended_batches_reuse_both_models_within_budget` test requires
`PHOTOCRAFT_ML_LAMA` and `PHOTOCRAFT_ML_OSOR_BUNDLE`. It runs two sequential
LaMa-to-OSOR batches through separate service instances at a 48 GiB allowance and
checks that neither model reloads in the second batch. The OSOR service smoke
also checks warm reuse when the next batch lists LaMa first but runs only OSOR.
These are cache-lifecycle checks, not image-quality or throughput comparisons.

The ignored `real_models_match_reference_and_reuse_one_warm_family` test requires `PHOTOCRAFT_ML_LAMA`, `PHOTOCRAFT_ML_MIGAN` and `PHOTOCRAFT_ML_ORACLE_DIR`. The oracle directory supplies a 512×512 interleaved RGB byte image (`image.rgb`), a 512×512 binary white-is-hole mask (`mask.bin`), and interleaved RGB byte outputs from an independent reference (`lama.rgb`, `migan.rgb`). Both artifacts are imported and verified by their catalog hashes. The test compares decoded samples within one byte value and checks cold/warm loads. Fixtures and test adapters do not qualify a model or platform; these comparisons cover only the supplied scene.

The ignored `real_onnx_cpu_metal_outputs_timings_and_drain` and
`real_lama_gpu_budget_falls_back_and_recovers` checks use the same explicit model
and oracle paths on macOS. They preserve the existing one-code-value output gate,
compare matched LaMa CPU/GPU runs, collect GPU timestamps separately from normal timing,
and exercise backend switching, cancellation, budget recovery and warm-file hash
rejection. They require a caller-owned model execution window; ordinary tests do
not load weights.

## OSOR-SDXL Phase II bridge

The `osor-sdxl` catalog entry is always discoverable. Native `osor` enables the
`osor` adapter and automatically compiles its Metal backend on macOS; other
native platforms compile the CPU backend. `osor-metal` remains a compatibility
feature alias. Desktop and CLI default features include `local-ai` and `osor`.
The library is pinned to [osor v0.1.0, revision cd99c92464ff36f7026ed80c4f2bec1f10073f84](https://github.com/doublesharp/osor-rs/tree/v0.1.0),
using Candle 0.11.0. Updating it requires renewing the affected loading, conversion
and inference checks.
The desktop application's native setup selects Metal on macOS and CPU elsewhere;
the bare library service defaults to CPU/F32 for OSOR. Applications
can explicitly construct `RemovalService::new(directory).with_osor_backend(OsorBackend::Metal { ordinal: 0 })`;
ML and engine code never read an environment variable to choose a backend. Unsupported
backend selections report unavailable. `model_runtime_available(id)` reports support
for each model independently; ONNX support alone does not enable OSOR.

`service.import("osor-sdxl", bundle_directory, control)` accepts an explicitly
selected, offline-converted Phase II bundle containing `manifest.json`,
`unet.safetensors`, `vae.safetensors` and `prompt.safetensors`. Import validates the
library manifest contract and hashes every capability-opened regular file, then
publishes only `osor-sdxl-reference.json` in the managed model directory. It never
copies the multi-GB weights. The source directory and all of its parents must remain at the imported
path without symlinks. Removing the model removes its managed reference and warm
cache, leaving the source weights intact. A different reference requires removing
the current reference first. Bundle files are bounded to 16 GiB in total. The exact tensor headers fed into the
SHA-256 stream must describe at most 12 GiB of decoded F32 weights; the native
loader independently enforces the same decoded-byte limit before allocation.
The 40 GiB model estimate includes headroom for activations and warm allocator
retention observed in repeated Metal execution; it is not a hard RSS limit.

The receipt pins exact source file lengths and SHA-256 values, including the raw
manifest bytes. Before every inference, including warm forwards, the store reopens
all source files through no-follow capabilities and rehashes them. The cold loader
receives the trusted manifest SHA-256 and validates the exact tensor bytes parsed
against its pinned manifest; a post-load manifest comparison provides an additional
check. Import/hash verification does not claim that the neural graph has successfully
loaded or that a scene has been qualified: tensor names/shapes are fully checked by
the library on cold load. Status distinguishes imported presence from verification
in this service instance; after restart it reports that bytes need reverification.
A failed source verification marks the current reference unavailable until reimport.
The shared-store same-user race limitations described above still apply.

The initial context limit preserves aspect ratio, downsamples only above a 512 px
long side, and reflection-pads each axis to a multiple of eight (minimum 32 px).
Padding is known mask context. RGB and the library's learned edit alpha are restored
to the source crop dimensions. The bridge supplies explicit Gaussian posterior and
diffusion noise from two frozen internal seeds, making repeated input reproducible
without extending `RemovalInput`. Its decoded RGB already includes the library's
latent alpha blend. `RemovalOutput.edit_alpha: Some(Vec<f32>)` discloses learned edit
support; consumers must not multiply this alpha into that RGB again. LaMa and MI-GAN
return `edit_alpha: None`.

OSOR shares the existing process-wide operation permit and single warm-family rule
with the ONNX adapters. The bridge calls `infer_with_cancellation`, connecting
`RemovalControl` to the library token with one scoped monitor per forward. The
monitor checks cancellation every 10 ms, including between stage-observer calls;
the library checks that token at cooperative checkpoints inside model blocks.
Cancellation is not an immediate device interrupt: device work is synchronized
before admission is released, and a completed output reservation remains owned
until apply/discard. The existing explicit noise streams and input geometry are
unchanged by the library update. Admission currently
uses a conservative **40 GiB** model working-set estimate plus caller-owned transient
buffers. The minimum retained input/output bound is **32 bytes per source pixel**
for OSOR (including learned alpha), versus 28 for the current ONNX adapters. This is
an admission estimate, not an operating-system allocation cap. `model_sha256(id)`
returns the imported raw manifest digest for replay identity; callers must capture or
revalidate it while holding the removal permit.

The [OSOR model card](https://huggingface.co/QinmingZhou/OSOR) identifies the SDXL
weights as CreativeML Open RAIL++-M with use restrictions. Explicit preparation
uses the pinned safetensors mirror and includes its license text; weights are
never embedded in the application. Code and weight licenses remain separate.

The ignored `real_osor_service_smoke_restores_rgb_alpha_and_reuses_warm_family`
test requires macOS, `--features osor-metal`, and an explicit local
`PHOTOCRAFT_ML_OSOR_BUNDLE` path. It imports the local reference, runs one synthetic
101×47 crop cold and warm, checks finite source-sized RGB/alpha, deterministic
repetition and retained-output admission, then removes only its receipt. Reserve a
single real-model memory window before running it. It prints a JSON result with
the pinned manifest digest and shared-host timings; this smoke does not establish
independent-reference accuracy or visual removal quality.

The Cargo package and Rust library are named `osor`; the Git repository remains
`doublesharp/osor-rs`. PhotoCraft uses the `v0.1.0` Git tag, with revision
`cd99c924` fixed in `Cargo.lock`. Its Rust source and dependency versions are
unchanged from the previously qualified `94584f64` pin; only the package version
and documentation changed. Earlier receipts below retain their original revisions.
Initial offline validation seeded the existing Cargo Git cache from the supplied local checkout.
That cache provenance is separate from fresh remote resolution.

On 2026-10-28, the `94584f64` CLI replayed the retained 704×368 scene on Metal/F32
against the same independent reference, original mask/noises, managed model bundle
and frozen tolerances. All 18 stages passed; decoded RGB maximum absolute error was
0.0006973743438720703 and RMS error was 0.000008269988458171247. No reference or limit
was changed. The historical application and installer results above retain their
actual revisions; this bounded CLI replay does not qualify broad image quality,
model preparation, other platforms or normal application performance.

The `94584f64` retained-bundle Metal/F32 service smoke also passed cold and warm
101×47 input: finite source-sized RGB/alpha, zero repeat difference, no warm reload
and retained-output admission. It used existing weights and exercised no download
or model preparation. These bridge checks remain separate from broad removal
quality and from the historical native application lifecycle checks.

## LaMa Metal qualification, 2026-10-29

On this macOS host, the retained 512×512 independent oracle scene passed the
unchanged one-code-value gate for CPU, Metal and their direct comparison. The
prepared LaMa graph placed 2,305 computational nodes on GPU and 308 on CPU, with
235 transfers and a 153,190,400-byte GPU arena. A separate warm profile recorded
2,415 GPU timestamp dispatches with positive aggregate device time; node counts
are not work percentages. Cold/warm execution, backend changes, retained-output ownership and
cancellation/drain followed by a successful warm fill passed. Matched timings
remain shared-host diagnostics; this is one-scene numerical evidence, not broad
removal-quality or performance acceptance. MI-GAN Auto remains CPU-only pending
separate GPU qualification. Scratch receipts retain all samples and hashes.
