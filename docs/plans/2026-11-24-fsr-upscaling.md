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

### Approach change (supersedes decision 1)

FSR is implemented **on top of the vendored [wgpu-ffx](https://github.com/cwfitzgerald/wgpu-ffx) crate** rather than hand-ported into `bri-render`. The crate lives in `vendor/wgpu-ffx/` (path dependency, pinned wgpu 30.0.1), with its MIT license and AMD shader sources kept in-tree under `shaders/include/fsr3upscaler/`; a `BRI-PATCH.md` documents the one local patch (inverted-depth validation threshold `<0.075` → `<0.04`). Decision 2's "reference only" stance is superseded: it is now a direct dependency, but vendored so we can patch in place if real hardware exposes bugs.

### What exists on disk

- `crates/client/src/app/fsr.rs` — the FSR module: lazy lifecycle mirroring bloom (created on first use; cleared on gpu_ready/gpu_stopped/map change), per-frame sizing, depth copy to R32Float (two compute pipelines, single-sample and MSAA variants), camera-only MV pass (main + clear variants sharing one pipeline layout), jitter applied per AMD's manual (`jitterX = 2*jx/renderWidth`, `jitterY = -2*jy/renderHeight`), jump-cut detection that resets history for one frame (eye moved >50 units or backward axes differ by >45°), dt clamped to [1.0, 250.0] ms.
- Settings: `Upscaler { Bilinear, Fsr }` in `graphics.rs` (pref `$pref::Video::Upscaler`; out-of-range values clamp like other video settings) and an "Upscaler:" row after "Render Scale:" on the Options screen with unit tests mirroring the render-scale test.
- `render_scene` wiring: FSR runs only when selected and the render size differs from the frame size (and is at least 2×2); otherwise a silent bilinear fallback for that frame. The depth-copy guard requires `gpu.depth` to exactly match the render size, since the WGSL does not bounds-check its source; any miss falls back to bilinear.
- Camera hook in `render.rs`: when FSR is active, a jittered per-frame view-projection is applied to the world pass via `FrameCamera`; the sub-half-pixel shift into culling/mirrors is harmless.

### Corrections found during implementation (record for Phase 2)

1. **Depth format**: FSR binds InputDepth as a *non-filterable float*, not a depth format — so engine depth must always be copied into an R32Float scratch texture even at `samples == 1`.
2. **MV sign convention**: stored motion vector = previous_jittered_pos − current_jittered_pos (`px_prev − px_curr`). Algebraically derived, never GPU-proven; see the trailing check in the laptop checklist below.

### wgpu 30 API adaptations (vs the crate's upstream wgpu-29 target)

`RenderPassColorAttachment` uses `ops: Operations<Color>` (no load/store/clear fields); render pass descriptors require `occlusion_query_set: None, multiview_mask: None`; pipeline descriptors require `cache: None` and explicit `PipelineLayout`s built from the bind group layouts; buffers bind via `as_entire_buffer_binding()`; scratch textures/buffers are 1×1/4-byte placeholders until first sizing (wgpu 30 types have no `Default`).

### Laptop validation checklist (Maxwell)

- Options → Graphics → **Upscaler = "FSR 3.1"** at Render Scale 50% or 70%; toggle back and forth to confirm the setting persists and bilinear is unchanged when off.
- Expect a one-time hitch on first enabling FSR (pipeline compile); subsequent toggles should be clean.
- Compare image quality against Bilinear at the same scale: edges, HUD/text crispness, overall sharpness.
- **Ghosting of moving objects is expected** in v1 — motion vectors are camera-only; per-object MVs are Phase 2.
- Watch for disocclusion artifacts near the 0.05 near plane and general brightness sanity (exposure pass output).
- **If everything trails in a consistent direction when the camera moves, the MV sign convention is suspect** — flip `px_prev − px_curr` to its negation in `fsr.rs` and re-test before chasing anything else.
- Then run the GPU-dependent `bri-client` test suite on that machine (it fails by design headless here) and do the Phase 3 A/B capture: same map/seed, camera pan plus a moving object, bilinear vs FSR.

### Remaining work (Phase 2+)

Per-object motion vectors (plan Phase 2); RCAS sharpening is off in v1 (`enable_sharpening = false`); the reactive mask pass if playtests show HUD/text shimmer at aggressive scales.
