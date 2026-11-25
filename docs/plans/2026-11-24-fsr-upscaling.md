# FSR 3.1 temporal upscaling plan

Proposed November 24, 2026. Adds AMD FidelityFX Super Resolution 3.1 as an
optional replacement for the current bilinear Render Scale stretch, so the
world can draw at fewer pixels and be reconstructed with a temporal
upscaler on any GPU vendor's hardware.

## Why only FSR

The three candidate upscalers were checked against the requirements (cross-GPU,
works in this wgpu engine, open source):

- **AMD FSR 3.1** — MIT license, full shader and C++ source in the
  [FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
  (v2.4, October 2026 ships FSR 3.1.6). Shader-based temporal upscaler; runs on
  any GPU with typed UAV loads and R16G16B16A16_UNORM support. Qualifies.
- **Intel XeSS** — the [intel/xess](https://github.com/intel/xess) repository
  (SDK 3.0.2, July 2026) ships prebuilt binaries and samples under Intel's SDK
  terms; the implementation is not open source despite Intel's knowledge base
  claiming otherwise ([Phoronix](https://www.phoronix.com/news/Intel-XeSS-SDK-2.0.1),
  [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/intel-finally-releases-the-xess-2-0-sdk-for-developers-technology-still-gated-by-closed-source-barriers)).
  Excluded: not open source, and it would mean shipping closed vendor binaries.
- **NVIDIA DLSS** — proprietary, requires an NVIDIA license, NVIDIA hardware
  only. Excluded.

AMD's newer ML upscaler ("FSR Upscaling", formerly FSR 4) is AMD-hardware-only
(RX 6000/7000/9000, Ryzen AI, Xbox), so it cannot be the cross-vendor path.
FSR 3.1 is the newest cross-GPU upscaler in the current SDK and remains its
recommended fallback baseline.

## Scope

In:

- FSR 3.1 upscaling (the temporal super-resolution passes only) as an option
  alongside the existing bilinear stretch, driven by a new graphics setting.
- The renderer inputs FSR requires that do not exist yet: motion vectors at
  render resolution and a single-sample depth copy when MSAA is on, plus a
  small exposure pass.

Out (explicitly):

- Frame generation (FSR FG or otherwise). It needs optical flow and previous-
  frame reprojection; revisit only after upscaling is stable in playtests.
- The FSR reactive mask pass (keeps UI/text edges crisp). No open wgpu port
  implements it yet; add later if playtests show edge artifacts on HUD or text.
- XeSS/DLSS integration of any kind, and the AMD ML upscaler.

## Current state the plan builds on

- `bri_render::upscale` (`crates/render/src/upscale.rs`) already implements
  Render Scale: the world draws into a smaller texture at
  `graphics.render_scale` percent (25–100, `$pref::Video::RenderScale`,
  choices 100/85/70/50 in `crates/ui/src/screens/options.rs`) and is stretched
  over the window with bilinear filtering. The interface draws at full size
  after. FSR slots into exactly this seam: same small world texture, a
  different stretch pass.
- Pass order today (`App::render_scene`, `crates/client/src/app/mod.rs`):
  world → (bloom copy) → bilinear upscale → UI. The plan keeps that order with
  FSR replacing the final stretch: world → resolve → bloom → **FSR** → UI.
- World color is already resolved from MSAA into the render-scale/bloom target;
  `gpu.depth` holds the world-pass depth, multisampled when
  `graphics.samples > 1` (`create_depth_samples`, `crates/client/src/app/render.rs`).
- No motion vectors exist anywhere in the codebase (verified by search).
- Headless frame capture tests already exercise this seam
  (`render_scale_draws_the_world_smaller_and_fills_the_window`,
  `crates/client/src/app/tests.rs`), and GPU pass timing exists
  (`gpu_passes`/`timing.rs`) so the new passes show up in the performance overlay.

## Design decisions

1. **Port FSR 3.1 into `bri-render`; do not depend on an external crate.**
   The engine already vendors what it needs (winit is patched from
   `vendor/winit` with its LICENSE kept) and every post-processing effect is a
   native `.rs` + `.wgsl` module in `crates/render`. A new
   `crates/render/src/fsr.rs` plus vendored shader sources under
   `crates/render/src/fsr/` follows that pattern, keeps the exact-pinned wgpu
   dependency graph untouched, and lets us add the reactive mask later without
   an upstream release. The shaders come from AMD's MIT-licensed FidelityFX SDK
   (FSR 3.1.6), converted to WGSL; a `LICENSE` file with AMD's copyright notice
   and a version stamp stay beside them, matching the winit precedent.
2. **Use [wgpu-ffx](https://github.com/cwfitzgerald/wgpu-ffx) as a reference
   implementation, not a dependency.** It is an MIT Rust port of the FSR 3
   upscaler for wgpu (context/view/dispatch API, HDR and depth flags, dynamic
   resolution). It targets wgpu 29 against our pinned `=30.0.1` and is marked
   early/experimental (no reactive pass; always picks full-precision shader
   variants), so depending on it couples core rendering to an experimental
   crate. Its WGSL conversions and pipeline orchestration are the fastest
   correct starting point for our own port, and Phase 0 below validates that
   assumption before we commit to the port.
3. **Keep the setting simple in v1.** Add one graphics option, Upscaler:
   Bilinear (default; today's behavior) or FSR 3.1. Render Scale keeps meaning
   "what percent of the window the world draws at"; when FSR is selected it
   becomes the downscale ratio fed to the upscaler. AMD's named presets
   (Quality 1.5x / Balanced 1.7x / Performance 2.0x / Ultra Performance 3.0x)
   can map onto Render Scale choices later without changing the setting shape.
   This is not a v20 setting, matching how other non-v20 options are marked in
   `options.rs`.
4. **Motion vectors: camera-only baseline first, then per-object.** A complete
   MV texture (camera + object motion) needs scene-shader changes across the
   shape/terrain/water/sprite passes and previous-frame transform plumbing. To
   get FSR working end-to-end quickly, Phase 1 computes MVs in a post pass from
   the depth buffer alone: each pixel's world position is reprojected through
   the previous frame's view-projection to yield its screen-space delta. That
   is exact for camera motion and zero for static geometry; moving objects will
   ghost until Phase 2 replaces it with per-object MVs written by the world
   passes (world-space motion from object matrix deltas plus camera inverse,
   converted to pixel offsets in the fragment shader).

## Phases

### Phase 0 — spike (de-risk before porting)

Vendor `wgpu-ffx` into a throwaway branch/crate against our wgpu pin; render
the existing synthetic test content headless at 50% scale with FSR on and off.
Confirms: wgpu 30 compatibility, shader conversion quality, pass cost from the
timing overlay, and image quality versus bilinear in an offscreen A/B capture.
Outcome is a go/no-go note in `docs/progress/` with the captures attached.

### Phase 1 — upscaler module, camera-only MVs

- `crates/render/src/fsr.rs`: context (pipelines, bind group layouts), per-size
  view (history/reconstruction textures at render and window resolution),
  dispatch entry point taking color, depth, exposure and an MV texture.
- Vendored FSR 3.1 WGSL shaders with LICENSE and version stamp.
- Exposure pass: small compute over the resolved world color producing the
  single-pixel exposure FSR expects (AMD's sample provides this utility).
- Depth copy pass for `samples > 1`: MSAA depth read into a single-sample
  R32Float texture at render resolution, allocated only while FSR is on.
- Camera-only MV post pass as described in decision 4; previous-frame
  view-projection stored per frame (the client already tracks camera state).
- Client wiring: `Graphics.upscaler` field, setting plumbing through
  `graphics.rs`, and the `render_scene` branch that dispatches FSR instead of
  the bilinear stretch when selected. Upscaler resets on map load/resize, as
  Render Scale targets already rebuild.

### Phase 2 — per-object motion vectors

- MV texture (RG16F at render resolution) cleared to zero each frame; world
  passes that draw moving geometry write their per-pixel screen-space delta as
  an extra color attachment via pipeline variants, so the FSR-off path is
  untouched.
- Previous-frame transform snapshot for dynamic objects (players, bricks/props,
  vehicles, projectiles, water surface, dynamic sprites); static chunks need no
  MV output because the texture is zero-cleared.
- Replace the camera-only post pass; keep it as a fallback only if a pass class
  cannot produce MVs.

### Phase 3 — settings UI, validation, playtest handoff

- Options screen entry (Upscaler menu beside Render Scale) in
  `crates/ui/src/screens/options.rs` with the non-v20 marking, plus unit tests
  mirroring the existing `render_scale_defaults_to_every_pixel_and_saves_a_choice`.
- Offscreen A/B evidence: same map and seed, camera pan plus a moving object,
  bilinear vs FSR captures; MV correctness checks (static scene yields zero MVs;
  known camera rotation yields the expected delta at a sampled pixel).
- GPU pass timings recorded for the new passes in the performance overlay;
  results noted against the existing render budgets.
- Playtest handoff to Maxwell with concrete instructions and the setting's
  default position (default stays Bilinear until playtests accept FSR as an
  equal-or-better image at a real frame-rate gain).

## Testing boundary

Per AGENTS.md, Maxwell performs all interactive playtests. Everything before
handoff is headless: synthetic-content frame captures, unit tests for the new
setting and size math, `bri-client --check` (no content changes expected), and
scoped clippy/tests on `bri-render`, `bri-client` and `bri-ui`. No GPU suites or
benchmarks run while Maxwell is playtesting.

## Risks and open questions

- **MSAA interplay**: FSR consumes resolved color, which the pipeline already
  produces; the only new requirement is single-sample depth (decision: copy
  pass). Confirm the bloom→FSR order keeps glow unblurred at render resolution.
- **Color space / HDR**: the world target uses the surface format today. If a
  surface is 10-bit or HDR, FSR's float intermediates handle it; verify against
  `color.rs` handling during Phase 1 rather than assuming.
- **Reactive mask gap**: without it, HUD/text edges can shimmer at aggressive
  scales. Acceptable for v1 (default scale stays modest); tracked as follow-up.
- **WebGPU/wasm compatibility** if a web target is ever built: the compute
  passes should be portable within wgpu's feature set; flag during Phase 0.
- **Trademark**: "FSR" and "FidelityFX" are AMD trademarks; using them
  descriptively in settings text follows common practice, but keep naming close
  to AMD's guidelines if the option ships publicly.

## Implementation status (November 24, 2026)

Implementation is complete on branch `bionic/upscaling` as uncommitted WIP,
pending GPU validation on Maxwell's laptop — this machine cannot run GPU tests
(VRAM occupied). Headless checks all pass: `cargo check -p bri-client
--all-targets` clean, clippy clean on `bri-client` and `bri-ui`, 6/6 FSR math
unit tests pass headless, the graphics setting test passes (2/2), and the
options upscaler test passes. The only failure in the full `bri-ui` suite is a
pre-existing environmental one (`avatar_pack_and_palette_render_check_synthetic`
needs a real adapter).

First GPU validation (Maxwell's laptop, NVIDIA) failed at device creation:
the vendored FSR3 passes declare 16-bit float storage images, which the
Vulkan spec does not expose as STORAGE_IMAGE for any conformant driver —
NVIDIA rejects the bind group layout outright. Fixed by widening every
FSR-written texture (and the engine's MV pass) from 16-bit to 32-bit float:
GLSL qualifiers in the callbacks header, `FsrResourceName::format()` and
texture creation in wgpu-ffx, the engine's `fsr.rs` MV/output formats, and a
recompile of all 37 affected SPIR-V files (documented as the second patch in
`vendor/wgpu-ffx/BRI-PATCH.md`).

Second GPU validation (same machine, Bionic disconnected to free VRAM) got
past Prepare Inputs and failed at "FSR3 Prepare Reactivity Bind Group Layout"
binding 10: ReadWrite storage access to R8Unorm is not supported. Different
root cause from the first: wgpu-core validates ReadWrite storage bind group
layout entries against a static WebGPU-guaranteed feature table whenever the
driver covers all guaranteed format features (every conformant driver does),
and that table grants R8Unorm no storage usage at all — so it fails on every
backend, not just NVIDIA. Fixed by widening the three remaining 8-bit FSR
state textures (accumulation current/previous, shading change, new locks)
from r8 / R8Unorm to r32f / R32Float in the callbacks header and wgpu-ffx's
format table plus texture creation; all 40 SPIR-V files recompiled,
spirv-val clean (third patch in `vendor/wgpu-ffx/BRI-PATCH.md`). Headless
checks pass again; awaiting another laptop run. That failing run also
segfaulted on process exit after the panic — if a passing run still does,
include that output.

Third and fourth GPU validations (same machine) got past every bind group layout and failed at shader module creation for "FSR3 Luma Pyramid" — twice, on two independent naga rules about memory decorations. Rule one (third run): `@coherent`/`@volatile` are only valid on variables in the storage address space; AMD's GLSL puts `coherent` on the SPD mip5 storage image and glslc emits all storage images with `UniformConstant` pointer types (no `Image` storage class in these SPIR-V 1.5 modules), which naga treats as non-storage — illegal regardless of driver or format support. Rule two (fourth run, after fixing rule one): any `@coherent` additionally requires the MEMORY_DECORATION_COHERENT capability at validation time, which wgpu only enables when the device requests `wgpu::Features::MEMORY_DECORATION_COHERENT` — our engine requests no such feature; that error pointed at the sibling SPD counter SSBO. Only two pass modules are affected (luma_pyramid, shading_change_pyramid); earlier runs died at earlier passes' bind group layouts before construction reached them (`FsrPass::new` builds each pass's shader module before its layout; lib.rs order: PrepareInputs → … → LumaPyramid 7th). Fixed by removing `coherent` from both SPD declarations in the callbacks header — safe because each invocation only claims a unique index via `atomicAdd` and touches its own slots (cross-invocation flow uses groupshared memory with explicit barriers; per-dispatch counter resets rely on Vulkan queue ordering), and glslc compiles the atomics identically without the qualifier. Only the two affected SPIR-V files were rebuilt (−40 B each across both edits); a scan of all 40 modules covering variable- and member-level decorations confirms zero Coherent/Volatile decorations remain and spirv-val is clean (fourth patch in `vendor/wgpu-ffx/BRI-PATCH.md`). Headless checks pass again; awaiting another laptop run. Note: the six passes constructed before LumaPyramid are now proven naga-clean on this machine, so any further module failure would be a new issue in one of the later passes — send that output too.

Fifth GPU validation (same machine) got past every shader module — Luma Pyramid included, confirming the coherent-removal fix — and failed at its bind group layout: binding 3 (`FrameInfo`, ReadWrite storage on Rgba32Float) is not supported. Root cause is a hole in the third patch's assumption: wgpu-core validates storage-texture access against the static WebGPU-guaranteed format table unless the device requests `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`, and that table grants combined read-write storage to R32Float only — Rg32f/Rgba32f get separate read-only and write-only storage. FSR's pyramid passes genuinely both load and store their SPD mip textures (Rg32f) and frame-info texture (Rgba32f) within one module, so no per-binding downgrade or format swap can satisfy the static table. Fixed engine-side in `crates/client/src/platform.rs`: the device now also requests `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` (intersected with what the adapter advertises), which makes wgpu validate formats against real driver capabilities — where every conformant Vulkan driver exposes full read-write 32-bit float storage per spec. This is the third patch's "considered alternative", now vindicated by evidence; the widened 32-bit formats stand as-is. The identical error recurred on the next run because this test's device comes from `bri_ui::gpu::Headless` (a feature-less descriptor), not platform.rs — that headless path now makes the same intersected request, so both device paths are covered.

Sixth GPU validation (same machine) got past every shader module and format check and failed at the luma pyramid's bind group layout with a new error class: "Too many bindings of type StorageTextures in Stage COMPUTE, limit is 4, count was 8". Root cause: wgpu's default device limits cap storage textures per compute stage at four (`Limits::default()` in wgpu-types), while FSR's luma pyramid binds eight — FrameInfo plus six SPD mips plus FarthestDepthMip1 (`SpdAtomicCount` is an SSBO). No FSR layout fits under four: ShadingChangePyramid needs six, PrepareInputs five. Fixed engine-side in both device paths (game device and the headless test device): the request now also sets `required_limits.max_storage_textures_per_shader_stage` to the adapter's own value — never more than that, so validation can never fail and weak GPUs still create their devices — and `render_scene` gates FSR creation on the device reporting at least eight (`fsr::MIN_STORAGE_TEXTURES_PER_STAGE`), falling back to bilinear otherwise: a GPU that cannot hold eight storage textures per stage would have wgpu panic building the pass. Headless checks pass again; awaiting another laptop run.

Seventh GPU validation (same machine) got past every FSR shader module and bind group layout — all of AMD's passes now build on NVIDIA — and failed in our own engine-side WGSL: `Device::create_shader_module` for "fsr passes" with a parse error (`expected ';' found '{'`). Root cause: wgpu 30's naga rewrote its WGSL frontend, and the four engine-side modules in `fsr.rs` had never been GPU-compiled — every earlier run died inside `FsrContext::new` first. Fixed five syntax/semantics gaps against naga 30's strictness (seventh change in `vendor/wgpu-ffx/BRI-PATCH.md`): call-form struct construction instead of brace literals, explicit `-> @location(0) vec4<f32>` on the two fragment entry points, full spec storage format names that name the real texture — `r32float` for the depth copies, `rg32float` for MV — plus the renamed `texture_depth_multisampled_2d` with scalar depth loads, and — per the current WGSL spec — textureStore values that are always `vec4<CF>`, so the MV pass stores vec4s into its RG image (only .xy lands; Vulkan's OpImageWrite rule sets a floor on component count, not an exact match). Verified headless with a CPU naga parse+validate harness mirroring wgpu-core's exact frontend path plus spirv-val on the emitted SPIR-V; cargo check and clippy clean. Awaiting another laptop run.

Eighth GPU validation (same machine) compiled every shader module — AMD's and ours — but failed at `Device::create_compute_pipeline` for "fsr depth copy": the shader's storage binding declared Rgba32Float while the pipeline layout offers R32Float. The original WGSL (`rgba32f`) was wrong twice over: abbreviated (naga 30 maps only full spec names) *and* naming an RGBA image where the scratch texture is R32Float — latent because these modules had never been GPU-compiled until now, and naga's validator cannot see the Rust-side formats. Both depth-copy modules now declare `r32float` to match `depth_single` (the vec4 store rule stands; only .r lands). Headless checks pass again; awaiting another laptop run.

Ninth GPU validation (same machine) built every FSR shader module and pipeline — the depth-copy format fix held — and failed at texture creation inside wgpu-ffx: `Device::create_texture` for "FSR3 Lanczos2 LUT" rejected R16Snorm, which wgpu gates behind `Features::TEXTURE_FORMAT_16BIT_NORM`, a feature neither device path requested. The LUT can't be re-formatted like the storage images (patches 2–3): FSR's GLSL samples it expecting snorm-normalized Lanczos coefficients. Fixed engine-side: both device paths now request `TEXTURE_FORMAT_16BIT_NORM` (intersected with adapter features), and the FSR capability gate in `render_scene` checks it, falling back to bilinear where absent — R16_Snorm is core Vulkan, so conformant drivers all expose it. An audit of every texture wgpu-ffx creates found this LUT the only 16-bit norm format; headless checks pass again (eighth change in `vendor/wgpu-ffx/BRI-PATCH.md`). Awaiting another laptop run.

Tenth GPU validation (same machine) got past every FSR shader module, pipeline, and texture — all of AMD's passes build on NVIDIA — and failed at the first bind group *creation* during frame dispatch: `Device::create_bind_group` for "FSR3 Prepare Inputs Bind Group" rejected binding 0 with "Texture binding 0 expects sample type Float { filterable: true }, but was given a view with format Rg32Float (sample type Float { filterable: false })". Root cause is the flip side of patches 2–3: those widened FSR's internal signals into 32-bit float formats because they are all the conformant read/write storage supports — and the Vulkan spec guarantees no linear filtering for R32F/RG32F/RGBA32F (wgpu marks such views non-filterable unless the driver advertises FLOAT32_FILTERABLE, which NVIDIA does not). Upstream wgpu-ffx declares every SRV bind group layout entry as filterable float except input depth — a *requirement* in wgpu-core's bind validation, so every 32f signal bind fails; and its GLSL linearly samples those same textures through one shared `s_LinearClamp` sampler (fourteen "Sample*" helpers over 32-bit float textures), which would be non-conformant at dispatch time even if the binds succeeded. Fixed with the ninth patch (`vendor/wgpu-ffx/BRI-PATCH.md`): the SRV entries now declare `Float { filterable: false }` (wgpu-core accepts any float view against such an entry; the Lanczos LUT keeps a filterable declaration, since RCAS linear-samples it and wgpu classifies R16Snorm as filterable float), and those fourteen helpers now use the existing point clamp sampler — exact, because every call passes an explicit LOD; only internal analysis signals lose within-level bilinear smoothing, never the color path (masks, input color, and the Lanczos LUT keep linear sampling). All 40 FSR3 SPIR-V files recompiled in place with group-consistency asserted; spirv-val clean; headless checks pass again. Awaiting another laptop run — this patch should clear bind group creation for every pass at once, since no SRV entry over-declares filterability anymore.

Eleventh GPU validation (same machine) got past every bind group layout and shader module but failed at compute pipeline *creation* for "FSR3 Prepare Reactivity": `Unable to filter the texture (group: 0, binding: 4) by the sampler (group: 0, binding: 14)` — non-filterable float textures can't be sampled with a filtering sampler. That is wgpu-core's *pipeline*-level cross-check (validation.rs): for every sampling pair in an entry point it rejects the pipeline when our own BGL declares the sampler `Filtering` while declaring the texture `Float { filterable: false }` — independent of the actual views and of driver capability, so no device feature request could have fixed it. The ninth patch's blanket non-filterable declaration satisfied the bind group check but broke this one for the five textures FSR still linearly samples through s_LinearClamp (input color Rgba16F; reactive-mask and transparency/composition slots — both bound to the crate's 1×1 R8Unorm default when the engine passes None; dilated reactive masks Rgba8U; Lanczos LUT R16Snorm). Binding 4 of PrepareReactivity is InputTransparencyAndComposition, binding 14 s_LinearClamp. wgpu-types computes only R32F/RG32F/RGBA32F views as non-filterable (gated on FLOAT32_FILTERABLE, which NVIDIA does not expose); all five views above are unconditionally filterable formats, so the fix — tenth patch (`vendor/wgpu-ffx/BRI-PATCH.md`) — declares exactly those five SRV entries `Float { filterable: true }` and leaves every 32f signal (all point-sampled) non-filterable. No shader changes, no .spv rebuild; headless checks pass again. Awaiting another laptop run — this should clear pipeline creation for every pass at once, since every filtering sampler now pairs with a filterable-declared texture.

Twelfth GPU validation (same machine) got past everything in wgpu-ffx — all of AMD's passes validated and dispatched on NVIDIA, confirming the tenth patch — and failed at the engine's own final step: `Device::create_bind_group` for "fsr blit" rejected binding 0 with "expects sample type Float { filterable: true }, but was given a view with format Rgba32Float (sample type Float { filterable: false })". The convert/blit full-screen copies share one source BGL entry that declared `filterable: true`, but both fragment shaders use only textureLoad (exact texel fetches, no sampler), so filtering is not required; the blit's source is FSR's Rgba32Float output, whose views are non-filterable on NVIDIA. Fixed engine-side (`crates/client/src/app/fsr.rs`; ninth change in `vendor/wgpu-ffx/BRI-PATCH.md`): the entry now declares `filterable: false`, which accepts any float view and trips no pipeline-level check since nothing samples with a sampler. An audit of every `filterable` declaration across the engine found this the only over-declaration on the FSR path — all others sit on long-standing render paths (bloom, sprites, scene color, UI) that read 8/16-bit formats and have always worked here. Headless checks pass again. Awaiting another laptop run.

Thirteenth GPU validation (same machine) got past every object creation — all FSR pipelines and bind groups, plus the engine's own blit layout, validated on NVIDIA — but failed at command *encoding*: "In a CommandEncoder — Features { CLEAR_TEXTURE } are required but not enabled on the device". The crate clears its SPD mips with `CommandEncoder::clear_texture` every frame (plus AccumulationPrevious/SpdMips on history reset), and wgpu gates that command behind a feature neither device path requested. wgpu-hal grants `CLEAR_TEXTURE` unconditionally on Vulkan (base feature set, no extension gate), so this was purely a missing request: both device paths now ask for it intersected with adapter features, the FSR capability gate checks it too (bilinear fallback where absent), and an audit of every encoder/queue call in the crate found no other feature-gated API. Headless checks pass again. Awaiting another laptop run — after this the frame should encode and submit cleanly; what remains is dispatch-time behavior on real hardware, the test's image assertions, and the known MV sign-convention risk if motion looks trailing.

### Approach change (supersedes decision 1)

FSR is implemented **on top of the vendored [wgpu-ffx](https://github.com/cwfitzgerald/wgpu-ffx) crate** rather than hand-ported into `bri-render`. The crate lives in `vendor/wgpu-ffx/` (path dependency, pinned wgpu 30.0.1), with its MIT license and AMD shader sources kept in-tree under `shaders/include/fsr3upscaler/`; a `BRI-PATCH.md` documents the local patches (the inverted-depth validation threshold `<0.075` → `<0.04`, the 16→32-bit float storage widening for Vulkan-conformant drivers, and the R8Unorm→R32Float widening for wgpu's WebGPU-guaranteed storage validation, plus removing the `coherent` qualifier from the SPD mip5 storage image so naga accepts the modules). Decision 2's "reference only" stance is superseded: it is now a direct dependency, but vendored so we can patch in place if real hardware exposes bugs.

### What exists on disk

- `crates/client/src/app/fsr.rs` — the FSR module: lazy lifecycle mirroring bloom (created on first use; cleared on gpu_ready/gpu_stopped/map change), per-frame sizing, depth copy to R32Float (two compute pipelines, single-sample and MSAA variants), camera-only MV pass (main + clear variants sharing one pipeline layout), jitter applied per AMD's manual (`jitterX = 2*jx/renderWidth`, `jitterY = -2*jy/renderHeight`), jump-cut detection that resets history for one frame (eye moved >50 units or backward axes differ by >45°), dt clamped to [1.0, 250.0] ms.
- Settings: `Upscaler { Bilinear, Fsr }` in `graphics.rs` (pref `$pref::Video::Upscaler`; out-of-range values clamp like other video settings) and an "Upscaler:" row after "Render Scale:" on the Options screen with unit tests mirroring the render-scale test.
- `render_scene` wiring: FSR runs only when selected, the render size differs from the frame size (and is at least 2×2), and the device supports at least eight storage textures per compute stage (`fsr::MIN_STORAGE_TEXTURES_PER_STAGE`; wgpu's default limit of four cannot hold the luma pyramid's layout) plus 16-bit norm texture formats (the Lanczos LUT is R16Snorm); otherwise a silent bilinear fallback for that frame. The depth-copy guard requires `gpu.depth` to exactly match the render size, since the WGSL does not bounds-check its source; any miss falls back to bilinear.
- Camera hook in `render.rs`: when FSR is active, a jittered per-frame view-projection is applied to the world pass via `FrameCamera`; the sub-half-pixel shift into culling/mirrors is harmless.

### Corrections found during implementation (record for Phase 2)

1. **Depth format**: FSR binds InputDepth as a *non-filterable float*, not a depth format — so engine depth must always be copied into an R32Float scratch texture even at `samples == 1`.
2. **MV sign convention**: stored motion vector = previous_jittered_pos − current_jittered_pos (`px_prev − px_curr`). Algebraically derived, never GPU-proven; see the trailing check in the laptop checklist below.
3. **textureStore is always vec4** (current WGSL spec, enforced by naga 30): storage texture stores take `vec4<CF>` for every format, including R and RG images — naga's validator hardcodes the quad size, matching spec; Vulkan discards the extra components (VUID-RuntimeSpirv-OpImageWrite-07112 requires *at least* as many components as the image format). The MV pass stores `vec4(mv, 0.0, 0.0)` into its Rg32Float image; the depth copies store `vec4(d)` into their R32Float images the same way.

### wgpu 30 API adaptations (vs the crate's upstream wgpu-29 target)

`RenderPassColorAttachment` uses `ops: Operations<Color>` (no load/store/clear fields); render pass descriptors require `occlusion_query_set: None, multiview_mask: None`; pipeline descriptors require `cache: None` and explicit `PipelineLayout`s built from the bind group layouts; buffers bind via `as_entire_buffer_binding()`; scratch textures/buffers are 1×1/4-byte placeholders until first sizing (wgpu 30 types have no `Default`).

### Laptop validation checklist (Maxwell)

**Before running anything:** disconnect Bionic first — its local LLM server
holds ~22 GiB of VRAM and the GPU tests need it.
- **Never sync `target/` between machines** (found 2026-11-24): Syncthing was syncing this repo's build directory to the testing laptop, renaming/deleting cargo rlibs mid-build and corrupting fingerprints — the symptom is `crate X required to be available in rlib format, but was not found` plus `.sync-conflict-*` files under `target/`. Fixed with a `.stignore` containing `**/target/` at `/run/media/surdavow/sub/Github` (note: v2.x calls it `.stignore`, not `.syncthingignore`). If the rlib error ever recurs, `cargo clean` and rebuild before chasing anything else.

- Options → Graphics → **Upscaler = "FSR 3.1"** at Render Scale 50% or 70%; toggle back and forth to confirm the setting persists and bilinear is unchanged when off.
- Expect a one-time hitch on first enabling FSR (pipeline compile); subsequent toggles should be clean.
- Compare image quality against Bilinear at the same scale: edges, HUD/text crispness, overall sharpness.
- **Ghosting of moving objects is expected** in v1 — motion vectors are camera-only; per-object MVs are Phase 2.
- Watch for disocclusion artifacts near the 0.05 near plane and general brightness sanity (exposure pass output).
- **If everything trails in a consistent direction when the camera moves, the MV sign convention is suspect** — flip `px_prev − px_curr` to its negation in `fsr.rs` and re-test before chasing anything else.
- Known residual risk from the 32-bit widening: linear filtering of R32F / RG32F / RGBA32F images is spec-optional but universally supported. If a validation error names `SamplerLinearClamp` or one of those formats, that — not the storage formats — is the cause.
- Then run the GPU-dependent `bri-client` tests on that machine (they fail by design headless where VRAM is occupied): at minimum `cargo test -p bri-client --lib fsr_upscaler_reconstructs_the_scaled_world_at_window_size` — a smoke test that hosts synthetic content, draws full/bilinear/FSR frames at 50% scale and checks the FSR context survives, the output is a plausible world distinct from bilinear, and toggling back and forth keeps drawing — plus its neighbor `render_scale_draws_the_world_smaller_and_fills_the_window` (the bilinear path it replaced), then the wider suite. Finally do the Phase 3 A/B capture: same map/seed, camera pan plus a moving object, bilinear vs FSR.
- If the test process segfaults on exit (SIGSEGV) even with a green result, send that output too — the second validation run showed one and it may be a side effect of panicking mid device creation rather than an FSR bug.

### Remaining work (Phase 2+)

Per-object motion vectors (plan Phase 2); RCAS sharpening is off in v1 (`enable_sharpening = false`); the reactive mask pass if playtests show HUD/text shimmer at aggressive scales.

## 2026-11-25: FSR 3.1 "heat shimmer" at still camera

**Symptom (user playtest):** with FSR 3.1 selected, aliasing wobbles like heat haze even when the camera is perfectly still; bilinear upscale looks fine.

### Root cause

The engine's MV contract and the crate's jitter cancellation are correct: for a still scene the stored MV is exactly `(j_prev − j_curr)` px and FSR cancels it to zero, so colour reprojection is stable. The shimmer comes from every *other* temporal read that adds a jitter-shifted UV but samples a **non-filterable 32-bit float texture** — core Vulkan exposes no filterable f32 formats:

- the shading-change pyramid's 5-tap crosses (`GetCurrentLumaBilinearSamples` / `GetPreviousLumaBilinearSamples`, `floor(uv*size)` + integer texelFetch) in `ffx_fsr3upscaler_shading_change_pyramid.h`;
- the luma-instability history read into point-sampled `r_luma_history` in `ffx_fsr3upscaler_luma_instability.h`;
- dilated-reactive-mask / farthest-depth-mip1 samples (point-sampled R32F) in accumulate & prepare_reactivity;
- the BRI-PATCH Ninth point-sampling sites in `ffx_fsr3upscaler_callbacks_glsl.h`.

Each read lands on an integer texel whose *content* differs frame-to-frame by exactly the jitter delta — which changes every Halton step even for perfectly still content. Shading change has no threshold (any nonzero luma diff decays accumulation), so reactivity/accumulation weights flicker sub-pixel every frame and the temporal filter keeps restarting → heat shimmer. In AMD's D3D12 reference these textures are filterable (R32F is filterable there) so the shifts cancel smoothly; the artifact is specific to Vulkan point sampling. No engine-side MV trick can fix it: the needed correction is fractional and varies per frame, which integer-indexed reads cannot track.

### Mitigation implemented 2026-11-25 (uncommitted)

`crates/client/src/app/fsr.rs`: `nearest_neighbor_jitter_tour(phase_count)` — a greedy nearest-neighbour ordering of all Halton phases; `Fsr::begin_frame` now reads the current jitter from this tour instead of plain index order. Per-frame jitter delta shrinks from ~0.25–0.5 px to well under 0.1 px typical, while full-cycle TAA coverage stays identical (every phase visited once). Two new CPU unit tests cover tour properties; all `app::fsr` pure-math tests pass. **This is a mitigation, not a cure** — the quantization error at point-sampled reads remains, just smaller.

### Definitive fix implemented 2026-11-25 (uncommitted, awaiting playtest confirmation)

GLSL surgery as planned — manual 4-tap bilinear at the jitter-shifted read sites: exact `texelFetch`es mixed by the fractional offset, clamped to the border like AMD's clamp sampler. No filtering sampler is involved, so conformance and the Ninth/Tenth patches' sample-type declarations are unchanged; at an exact texel centre this returns exactly the old point-sampled value.

- **Converted (jitter-shifted / reprojected UV sites):** five "Sample*" helpers in `ffx_fsr3upscaler_callbacks_glsl.h` — `SampleSPDMipLevel`, `SampleShadingChange`, `SampleFarthestDepthMip1`, `SampleCurrentLuma`, `SampleLumaHistory` — via new shared infrastructure (`FfxBilinearCoords` / `FfxBilinearResolve` / `FfxBilinearNeighbor`); and the shading-change pyramid's two 5-tap crosses in `ffx_fsr3upscaler_shading_change_pyramid.h`, which now keep a *continuous* base position (the +Jitter term cancels the luma buffer's own jitter shift, so a still scene lands on identical world points in current and previous frames) with each tap bilinear at base+offset. A deliberate deviation from AMD's reference, which floors these crosses.
- **Deliberately left point-sampled** (verified stable for still scenes — no jitter term or exact integer positions): `SampleAccumulation`, `SampleLumaInstability`, the dilated MV/depth reads, and the exact-position `LoadCurrentLuma` users (`ComputeThinFeatureConfidence`, luma pyramid source load).
- 28 `.spv` files rebuilt in place (accumulate ×24 permutation variants + one each of luma_instability, prepare_reactivity, shading_change, shading_change_pyramid); the other 12 byte-identical; all 40 pass spirv-val. Rebuild/regression tooling is now in-tree: `vendor/wgpu-ffx/shaders/rebuild_spv.py` (`--dry-run` byte-compares every committed `.spv` against a fresh compile — one command shows an edit's blast radius).
- Gotcha found on first compile: desktop GLSL for Vulkan has no `saturate` builtin (ESSL-only) — the infrastructure uses `clamp(x, 0.0, 1.0)`.

Verified CPU-side: cargo check/clippy clean, spirv-val 40/40, dry-run shows exactly the expected 28 files changed and nothing else, all `app::fsr` pure-math tests pass. **Awaiting playtest:** FSR on/off at 50% scale with a still camera — shimmer should be gone or near-gone; also check motion handling for regressions (the crosses are bilinear where AMD's reference floors them). If shimmer persists: remaining candidates are bilinearizing the rest of patch-nine's point-sampled helpers (`SampleAccumulation`, `SampleLumaInstability`, dilated MV/depth UV reads), or a DittoSR port as a second upscaler option.

### Playtest 2026-11-25 (after the GLSL fix): shimmer persists; name tags implicated

The user's playtest of the rebuilt release binary (verified to contain the new SPIR-V) found the world shimmer **unchanged**, plus a new clue: player name tags jitter too, "for some reason" — they're UI.

**Name-tag root cause (engine-side, two parts):** `name_tags()` projected labels through FSR's *jittered* view-projection (`render_frame` replaces `camera.view_projection` before anything else runs), and its NDC offset is amplified by the upscale ratio when mapped to window pixels — up to ~1-2 px of wobble at 50% scale. And like all UI, tags draw into the render-scale buffer *before* upscaling, so FSR temporally filters them like scene content (thin high-frequency text being exactly what temporal filters struggle with).

**Fix applied:** `render_frame` now captures `ui_view_projection` before FSR replaces the camera's and passes it to `name_tags()`, making label positions temporally stable so the filter can lock onto them. The proper fix — drawing name tags *after* upscaling at window resolution, crisp text never entering temporal history (standard practice: BeamNG, the Bloodborne/Uncharted ports all composite HUD post-upscale) — is a follow-up; bri-ui's `Ui::render` already takes an arbitrary target + DrawList, so a post-FSR name-tag pass is feasible.

**Research on the residual world shimmer:** it matches documented FSR 3.1 behavior rather than a port delta now that our sampling matches AMD's reference semantics — Digital Foundry's FSR 3.1 review calls sub-pixel flicker "inherent to FSR" and notes it persists in 3.1 (reduced vs 3.0); [AMD SDK issue #134](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/issues/134) reports "a constant shimmering on almost every edge with FSR3 as if the algorithm is unable to converge, this does not happen with FSR2" against AMD's *own* Unreal plugin; [OptiScaler #55](https://github.com/optiscaler/OptiScaler/issues/55) sees excessive shimmer in shipped games (Cyberpunk 2077, Alan Wake 2). Our content — flat colours and hard edges on every surface — is the worst case for edge convergence. Options: accept it with bilinear as the stable fallback; trade render scale against aliasing load; port **FSR 2** as an alternative upscaler option (better temporal stability per those reports; wgpu-ffx ships FSR 3 only); or DittoSR later (user's suggestion).
