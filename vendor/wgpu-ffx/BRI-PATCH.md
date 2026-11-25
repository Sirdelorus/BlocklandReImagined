# Vendored: wgpu-ffx

Source: https://github.com/cwfitzgerald/wgpu-ffx, main at
`6fe7064064b396b30df97341c2ca957ca1b00884`, vendored 2026-11-24. License: MIT
(`LICENSE.MIT`). Upstream also ships the AMD FidelityFX shader sources under
`shaders/` (MIT, Advanced Micro Devices, Inc.); those stay as reference for
the in-tree FSR port planned in `docs/plans/2026-11-24-fsr-upscaling.md`.

Local changes from upstream:

- The upstream Cargo workspace was flattened so this folder holds two plain
  crates (`wgpu-ffx`, `wgpu-ffx-shaders-spv`) that the main workspace can path-
  depend on, like `vendor/winit`. Removed: root `[workspace]` manifest,
  `Cargo.lock`, `xtask/`, `renovate.json`, `rust-toolchain.toml`, `deny.toml`,
  `RELEASE.md`.
- `wgpu-ffx/Cargo.toml`: workspace-inherited fields inlined; the wgpu
  requirement changed from upstream's `29` to `=30.0.1` so it unifies with the
  workspace pin and shares one wgpu build; dev-dependencies dropped (their
  sample/test backends are not used here).
- `wgpu-ffx/src/pass.rs`: adapted the `create_shader_module_passthrough`
  call to wgpu 30's `ShaderModuleDescriptorPassthrough` — the old
  `num_workgroups: (0, 0, 0)` hint became an empty `entry_points` list and
  the `spirv` field now takes u32 words (the existing
  `bytemuck::cast_slice` infers that from the field type).
- `wgpu-ffx/src/validation.rs`: the DEPTH_INVERTED sanity guard rejected a
  "far" plane below 0.075; under the inverted convention our engine's true
  near plane (0.05, `FAR_PLANE`/near in `crates/client`) is what gets passed
  as `camera_far`, so it tripped `InvertedDepthFarTooLow`. Lowered that floor
  to 0.04 — the check still catches accidental zero/near-zero planes while
  allowing our near plane.

Build note: on cargo/rustc 1.93.1, selecting the vendored crate directly
(`cargo check -p wgpu-ffx`) panics in cargo's feature resolver; build or
check it through a consuming crate instead (e.g.
`cargo check -p bri-client --all-targets`).

## Second patch — NVIDIA-compatible storage formats (16→32-bit float)

Vendored as-is, FSR 3.1 fails device validation on conformant Vulkan drivers
(NVIDIA): `Device::create_bind_group_layout` for the "FSR3 Prepare Inputs Bind
Group Layout" rejects binding 3 with "WriteOnly access to storage textures
with format Rg16Float is not supported". Root cause: the Vulkan spec exposes
no STORAGE_IMAGE feature for any 16-bit float format (R16F, RG16F,
RGBA16F), and NVIDIA's driver conforms. Upstream declares 14 explicit-format
storage image sites with `r16f`/`rg16f`/`rgba16f` GLSL qualifiers and matching
Rust formats, so FSR3 cannot run unpatched on any conformant GPU (it may only
have worked on non-conformant drivers).

Changes:

- `shaders/include/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h`: all 14
  explicit-format storage declarations widened to 32-bit (`r16f`→`r32f`,
  `rg16f`→`rg32f` — the mip5 site keeps its `coherent` qualifier — and
  `rgba16f`→`rgba32f`). These are the only storage sites with explicit format
  qualifiers in the callbacks header.
- `wgpu-ffx/src/resources.rs`: `FsrResourceName::format()` returns R32Float /
  Rg32Float / Rgba32Float for every resource FSR writes (output color,
  dilated motion vectors, luma family, farthest depth and its mip1, internal
  upscaled colors, SPD mips, luma history), with the texture creation in
  `FsrResources::new` updated to match. `InputColor` stays Rgba16Float — it is
  only attached/sampled, never stored.
- `wgpu-ffx/src/lib.rs`: the VRAM estimator's byte math and comments follow
  the new formats; `FsrDispatchInfo`'s doc comment notes dilated motion
  vectors are Rg32Float; the dummy textures in the `fsr_dispatch_smoke` test
  use the widened formats.
- All 37 affected `.spv` files under `wgpu-ffx-shaders-spv/src/fsr3upscaler/`
  recompiled in place (the three passes with no 16-bit float storage —
  autogen_reactive, prepare_reactivity, shading_change — are untouched).
  Filenames keep upstream's content hashes and are stale by design; the
  generated `mod.rs` is unchanged.

Engine-side follow-on (`crates/client/src/app/fsr.rs`, outside this crate):
the camera-only MV pass writes its output as a storage image, so its WGSL
qualifier, bind group layout entry, and `reallocate()` moved from rg16f /
Rg16Float to rg32f / Rg32Float; the FSR output texture is now Rgba32Float.

Recompile recipe (from upstream's deleted `xtask/src/compile_shaders.rs`):

    glslc --target-env=vulkan1.2 -Ishaders/include <base+perm defines> \
        -fshader-stage=comp -O <pass.glsl> -o out.spv

Compiler drift note: the local glslc (Debian 1:1.4.363.0) is newer than
upstream's and folds redundant double-bitcasts, so the luma_pyramid and
shading_change_pyramid variants are ~128 B smaller and not byte-identical to
upstream even pre-patch; functionally identical (spirv-val clean). The eight
prepare_inputs permutation variants compiled byte-identical to upstream with
this recipe, confirming it.

Verified: zero 16-bit float writeonly storage images in any of the 40 `.spv`
files and all pass `spirv-val`; `cargo check -p bri-client --all-targets` is
clean.

## Third patch — R8Unorm→R32Float for WebGPU-guaranteed storage validation

Even after the second patch, FSR 3.1 still failed device creation on a
conformant driver (observed on NVIDIA/Blackwell): `Device::create_bind_group_layout`
for "FSR3 Prepare Reactivity Bind Group Layout" rejected binding 10 with
"ReadWrite access to storage textures with format R8Unorm is not supported".
Root cause — different from the second patch's Vulkan-spec gap: wgpu-core
validates ReadWrite storage bind group layout entries against a static
WebGPU-guaranteed feature table (`guaranteed_format_features` in wgpu-types)
whenever the device's downlevel flags include WEBGPU_TEXTURE_FORMAT_SUPPORT,
which wgpu-hal sets on Vulkan for any driver that covers all WebGPU-guaranteed
format features — i.e. every conformant driver, on every backend. In that
table R8Unorm has no storage usage at all (Rgba8Unorm is write-only), while
the 32-bit float formats have full read/write storage. So this failure is not
NVIDIA-specific; it would hit any conformant driver.

Changes:

- `shaders/include/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h`: three
  explicit-format storage declarations widened r8→r32f (`rw_accumulation`,
  `rw_shading_change`, `rw_new_locks`). Full audit of all 19 explicit-format
  sites in the header: these were the only remaining sub-32-bit ones. The one
  rgba8 site (DilatedReactiveMasks) is bound WriteOnly and is
  WebGPU-guaranteed as-is; the two r11f_g11f_b10f sites belong to the
  GenerateReactive pass, which upstream never creates (commented out in
  `lib.rs`, `todo!()` in `pass.rs`) and are inert.
- `wgpu-ffx/src/resources.rs`: `FsrResourceName::format()` returns R32Float
  for AccumulationCurrent/AccumulationPrevious, ShadingChange and NewLocks;
  the matching texture creation sites follow.
- `wgpu-ffx/src/lib.rs`: VRAM estimator byte math ×4 for those three textures.
- All 40 `.spv` files under `wgpu-ffx-shaders-spv/src/fsr3upscaler/`
  recompiled in place (only the accumulate family, prepare_reactivity and
  shading_change changed bytes; glslc -O strips unused declarations).

Behavior note: r8→r32f on accumulation/shading-change/new-locks removes the
1/255 truncation of internal FSR state — strictly more precise, no regression
expected.

Considered alternative: requesting `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
on device creation would make wgpu query real driver capabilities (NVIDIA
does support R8 storage per spec) and let the original formats stand. Widening
to 32f was chosen instead because it is WebGPU-guaranteed on every backend
and leaves the engine's device features untouched — a premise the fifth GPU run
partly overturned: combined read-write storage of Rg32f/Rgba32f is not in that
guaranteed table, so the engine now requests this feature after all (see the
fifth change below).

Verified: explicit image formats across all 40 modules are now only
R32f/Rg32f/Rgba32f plus one Rgba8 (DilatedReactiveMasks, bound write-only in
`to_bgl_entry` and WebGPU-guaranteed); no 8-bit or 16-bit float storage images
remain; all pass `spirv-val`; `cargo check -p bri-client --all-targets` and
clippy are clean.

## Fourth patch — removed `coherent` from both SPD declarations

Even after patches 2–3, FSR 3.1 still failed on a conformant driver (observed on
NVIDIA/Blackwell) — this time not at bind group layout creation: 
`Device::create_shader_module` for "FSR3 Luma Pyramid" was rejected by naga with
"Global variable [5] '' is invalid … Memory decorations (`@coherent`, `@volatile`) are only valid for variables in the `storage` address space".

Root cause — a third, distinct issue: AMD's callbacks header declares the topmost SPD mip as
`layout(set = 0, binding = FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_5, rg32f) coherent uniform image2D rw_spd_mip5;`. glslc emits every `uniform image2D` global in these SPIR-V 1.5 modules as an `OpVariable` with a `UniformConstant` pointer type (it never uses the `Image` storage class here), and naga treats `UniformConstant`-class globals as outside the `storage` address space — so the `Coherent` memory decoration is illegal there and the module is rejected on any driver, independent of format support. The declaration lives in a shared header but only compiles into two pass modules: luma_pyramid (binding 9) and shading_change_pyramid (binding 10). Earlier runs never reached it because `FsrPass::new` creates each pass's shader module before its bind group layout, and lib.rs constructs the passes PrepareInputs → PrepareReactivity → … → LumaPyramid (7th) / ShadingChangePyramid (8th): patches 2–3's format bugs aborted device setup at earlier passes' layouts first.

After removing the mip5 decoration, a fourth GPU run got further into the same module and naga rejected it for a different reason: "Global variable [6] '' is invalid … `@coherent` requires the MEMORY_DECORATION_COHERENT capability". That global is the sibling SPD counter SSBO. naga enforces two independent rules on memory decorations: (1) they are only valid in the storage address space — which kills the mip5 image decoration above no matter what; and (2) any `@coherent`/`@volatile` additionally requires the corresponding device feature (`wgpu::Features::MEMORY_DECORATION_COHERENT`) to be enabled when wgpu validates the module — wgpu-naga-bridge maps device features onto naga's capability set, and our engine requests no such feature. glslc emits two decorations for a `coherent buffer` block (an `OpDecorate` on the variable plus an `OpMemberDecorate` on its member), so both had to go.

Changes:

- `shaders/include/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h`: removed the
  `coherent` qualifier from both SPD declarations — the mip5 storage image
  (line 855) and the `RwSpdGlobalAtomic` counter buffer block (line 896). glslc
  still compiles the `atomicAdd`/reset pair identically without the qualifier:
  Vulkan GLSL allows atomic builtins on plain integer lvalues, emitting relaxed
  atomics (`OpAtomicIAdd` on a plain uint pointee, no `OpTypeAtomic`) which naga
  handles via its scalar-to-atomic type upgrade in `front/atomic_upgrade.rs`
  (verified with a standalone glslc compile of the non-coherent pattern).

Why removing it is safe: the counter's only job is handing each invocation a unique index (`SPD_IncreaseAtomicCounter`); atomicity of that claim does not depend on `coherent`. Each invocation then touches only its own workgroup-shared slots and its own image pixels — cross-invocation data flow goes through groupshared memory with explicit barriers, not global-memory coherence. The counter is reset once per dispatch (`SPD_ResetAtomicCounter`) for the next dispatch, whose visibility of that write Vulkan guarantees by queue ordering regardless of decorations.

Considered alternative: request `wgpu::Features::MEMORY_DECORATION_COHERENT` at device creation (all four wgpu backends advertise it unconditionally) and keep AMD's SSBO decoration. Rejected to avoid coupling FSR to a device feature — the source-level removal stays self-contained in this crate. The mip5 image's `coherent` could not be kept under any option: rule (1), the storage-address-space restriction, is independent of capabilities.

Verified: rebuilt with the same tooling — of 40 files only luma_pyramid and shading_change_pyramid changed bytes (the mip5 edit removed one `OpDecorate`, −12 B each; the SSBO edit removed an `OpDecorate` plus an `OpMemberDecorate`, −28 B each; all 32 permutation combinations per group remain byte-identical). A disassembly scan covering both decoration forms (`OpDecorate`, `OpMemberDecorate`) across all 40 modules finds zero remaining Coherent/Volatile decorations anywhere. All 40 pass spirv-val; `cargo check -p bri-client --all-targets` is clean.

Upstream note: the .spv files shipped upstream have the identical structure (the decoration is present in the pre-patch modules), so unpatched wgpu-ffx FSR3 should fail on any wgpu/naga consumer — worth reporting to cwfitzgerald/wgpu-ffx alongside patches 2–3.

## Fifth change (engine-side) — `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` on the device

Not a patch to this crate. After the fourth patch, GPU validation got past every FSR shader module and failed at "FSR3 Luma Pyramid Bind Group Layout" binding 3 (`FrameInfo`, ReadWrite storage, Rgba32Float). Root cause is a hole in the third patch's assumption: wgpu-core validates storage-texture access against the static WebGPU-guaranteed format table unless the device requests `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` (see `describe_format_features` in wgpu-core), and that table grants combined read-write storage to R32Float only — Rg32f/Rgba32f get separate `STORAGE_READ_ONLY | STORAGE_WRITE_ONLY`. FSR's pyramid passes genuinely both load and store their SPD mip textures (Rg32f) and frame-info texture (Rgba32f) within one module, so no per-binding downgrade or format swap can satisfy the static table.

Change: device creation now requests `wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` — always intersected with what the adapter advertises — in two places: `crates/client/src/platform.rs` for the game device (alongside the timestamp features) and `bri_ui::gpu::Headless::new()` in crates/ui, which creates the feature-less device this project's GPU tests use; a run after only platform.rs was fixed failed identically because the test device never received the feature. With it set, wgpu validates all format usage against real driver capabilities; every conformant Vulkan driver exposes full read-write storage for 32-bit float formats per spec, so this is safe on NVIDIA/AMD/Intel alike and can only relax — never tighten — validation relative to the guaranteed table. This was the third patch's "considered alternative"; the fifth GPU run vindicated it. The widened 32-bit formats from patches 2–3 stand unchanged.

## Sixth change (engine-side) — `max_storage_textures_per_shader_stage` limit on the device

Not a patch to this crate. After the fifth change, GPU validation got past every FSR shader module and format check and failed at "FSR3 Luma Pyramid Bind Group Layout" with "Too many bindings of type StorageTextures in Stage COMPUTE, limit is 4, count was 8". Root cause: wgpu's `Limits::default()` caps storage textures per compute stage at four (wgpu-types), while the luma pyramid pass binds eight — FrameInfo plus six SPD mips plus FarthestDepthMip1 (`SpdAtomicCount` is an SSBO, not a texture). No FSR layout fits under four: ShadingChangePyramid needs six and PrepareInputs five.

Change: device creation now also requests `required_limits.max_storage_textures_per_shader_stage` set to the adapter's own value — in the same two places as the fifth change (`crates/client/src/platform.rs` for the game device, `bri_ui::gpu::Headless::new()` in crates/ui for the test device). Requesting exactly what the adapter reports can never fail validation (wgpu-core rejects only strictly-greater requests), so weak GPUs still create their devices; on a GPU that supports fewer than eight storage textures per stage FSR simply cannot build its passes, and `render_scene` now checks `device.limits()` against `fsr::MIN_STORAGE_TEXTURES_PER_STAGE` (= 8) before creating the upscaler, falling back to bilinear like every other FSR miss instead of letting wgpu panic.

## Seventh change (engine-side) — naga 30 WGSL frontend fixes in `fsr.rs`

Not a patch to this crate. After the sixth change, every AMD pass built on NVIDIA and the failure moved into our own engine-side shaders: `Device::create_shader_module` for "fsr passes" rejected with a parse error — `expected ';' found '{'` at `return VsOut { pos: ... }`. Root cause: wgpu 30's naga (30.0.1) rewrote its WGSL frontend, and the four engine-side modules in `crates/client/src/app/fsr.rs` were written to an older frontend's leniency — they had never been GPU-compiled, because every earlier run died inside `FsrContext::new` before `Fsr::new` reached them.

Five fixes against naga 30's strictness (all in the WGSL strings of `fsr.rs`; no Rust-side changes):

1. **Struct literals**: the new frontend has no brace form (`Type { field: value }`); construction is call-form, as elsewhere in this codebase — `return VsOut(vec4(p[i], 0.0, 1.0));`.
2. **Fragment entry points** need explicit return annotations: both identity-copy fragments are now `-> @location(0) vec4<f32>` (without it the validator rejects the entry point with InvalidReturnType).
3. **Storage format names must be full spec names, naming the real texture**: naga's frontend maps only `r32float` / `rg32float`, not the `rgba32f` / `rg32f` abbreviations — and the depth-copy modules' original `rgba32f` was doubly wrong: abbreviated *and* naming an RGBA image where the scratch texture is R32Float. The pipeline-layout match caught it on first GPU run ("Texture class Storage { format: R32Float, ... } doesn't match the shader Storage { format: Rgba32Float, ... }"): both depth-copy modules now declare `r32float` (matching `depth_single`) and the MV module declares `rg32float` (matching `motion_vectors`). The Rust-side formats were already R32Float/Rg32Float from the second patch.
4. **Multisample depth type renamed**: `texture_multisample_depth_2d` → `texture_depth_multisampled_2d`; a depth-texture `textureLoad` returns scalar f32, so the `.r` in both depth-copy modules went away (the MV module's `.r` on its plain float depth read stays).
5. **textureStore always takes vec4** — a WGSL spec change naga now enforces: the stored value is `vec4<CF>` for every storage format, so the MV pass's three stores into its rg32float image became `vec4<f32>(0.0)` (twice) and `vec4(px_prev - (px + 0.5), 0.0, 0.0)`. Only .xy lands in the RG image — that is what FSR reads. naga's SPIR-V backend emits the full 4-component value as the OpImageWrite operand without truncation; that is valid because Vulkan sets a floor on component count rather than an exact match: VUID-RuntimeSpirv-OpImageWrite-07112 requires the Texel to have *at least* as many components as the image format.

Verified headless (no GPU): a CPU harness running naga 30.0.1's `parse_str` plus full validation — wgpu-core's exact frontend path, which its device code notes must stay in sync with `naga::front::wgsl::parse_str` — accepts all four modules; the SPIR-V naga lowers for each passes spirv-val (the disassembly confirms the vec4 OpImageWrite into the Rg32f image); `cargo check -p bri-client --all-targets` and clippy are clean. Awaiting a laptop run.

## Eighth change (engine-side) — `TEXTURE_FORMAT_16BIT_NORM` on the device

Not a patch to this crate. After the seventh change, every FSR shader module and pipeline built and the failure moved into texture creation: `Device::create_texture` for "FSR3 Lanczos2 LUT" rejected R16Snorm with "Texture format R16Snorm can't be used due to missing features … TEXTURE_FORMAT_16BIT_NORM". Root cause: wgpu gates all six 16-bit *norm* formats (R/Rg/Rgba × Unorm/Snorm) behind that feature (`TextureFormat::required_features` in wgpu-types), and the engine's device paths requested no such feature. The LUT cannot be re-formatted like patches 2–3's storage images: FSR's GLSL samples it expecting snorm-normalized Lanczos coefficients, so the value encoding is part of the shader contract.

Change: both device paths (the game device in `crates/client/src/platform.rs`, the headless test device in crates/ui) now also request `wgpu::Features::TEXTURE_FORMAT_16BIT_NORM` — intersected with what the adapter advertises, like the fifth change's format feature — and `render_scene`'s FSR capability gate checks it too, falling back to bilinear on a device that lacks it. R16_Snorm is a core Vulkan format, so every conformant Vulkan driver exposes the feature; where a backend does not, the intersect means we simply ask for nothing extra. An audit of every texture this crate creates found the LUT the only 16-bit norm format in use — everything else is 32-bit float or 8-bit unorm, and the engine's own Rgba16Float input needs no feature (wgpu gates only the norm variants). `cargo check -p bri-client --all-targets` and clippy on `bri-client` + `bri-ui` are clean. Awaiting a laptop run.

## Ninth patch — non-filterable sample types, point sampling of 32-bit float signals

Even after patches 2–4 and engine changes 5–8 (device features/limits), FSR 3.1 still failed on a conformant driver (observed on NVIDIA/Blackwell) — this time at bind group *creation* during the first frame dispatch: `Device::create_bind_group` for "FSR3 Prepare Inputs Bind Group" rejected binding 0 with "Texture binding 0 expects sample type Float { filterable: true }, but was given a view with format Rg32Float (sample type Float { filterable: false })".

Root cause — two distinct issues in one failure mode:

1. **The BGL entries over-declared.** Upstream's `to_bgl_entry` declares every SRV texture entry as `TextureSampleType::Float { filterable: true }` except InputDepth (`filterable = !matches!(self, FsrResourceName::InputDepth)` in resources.rs). wgpu-core's bind group validation treats that as a *requirement*: the view must actually be filterable — it accepts any float view only when the entry says `filterable: false` (the sample-type match in its create_bind_group path). Meanwhile wgpu-types computes R32F/RG32F/RGBA32F views as non-filterable unless the device advertises `Features::FLOAT32_FILTERABLE`, a capability wgpu-hal derives from the driver's own format properties — NVIDIA does not expose linear filtering for those formats, so on this machine every 32f view is `Float { filterable: false }` (visible in the error message itself). Patches 2–3 moved FSR's internal signals into exactly those 32-bit formats, so from that point *every* SRV bind of an internal signal was doomed; PrepareInputs' binding 0 (the engine's Rg32Float motion vectors) is simply the first one reached. The same arm also declared the Lanczos LUT as a plain float — wgpu classifies snorm formats as `Float { filterable: true }` in its sample-type table (wgpu 30 has no snorm variant), so that bind happened to match, but only by accident of classification.

2. **The shaders genuinely linear-sample non-filterable textures.** AMD's callbacks header routes every "Sample*" helper through `s_LinearClamp`. Fourteen helpers sample a 32-bit float texture: eleven over the crate's internal signals (SPD mips Rg32f; accumulation, shading change, luma family and farthest depth + mip1 R32f; luma history and internal upscaled color Rgba32f) and three over engine-provided textures (input depth and dilated depth R32f; dilated motion vectors Rg32f). Linear filtering of those formats is not guaranteed by the Vulkan spec's format table for any driver, so on a conformant driver these dispatches are non-conformant even if the binds succeed. This is the flip side of patches 2–3: they forced FSR's internal signals into 32-bit float because that is all the conformant read/write storage supports — and those formats have no linear filtering guarantee, so making storage conformant made AMD's sampling helpers non-conformant on exactly the drivers where the storage fix is needed. (Pre-patch-2 several of these signals were 16-bit float or R8Unorm — filterable formats — so upstream's linear sampling was conformant in principle; but those storage formats are themselves non-conformant, which is why unpatched FSR3 could never have run on a conformant driver at all.)

Changes:

- `wgpu-ffx/src/resources.rs`: the SRV arm of `to_bgl_entry` now declares every float texture entry as `Float { filterable: false }` — wgpu-core accepts any float view against such an entry, so this fixes binds for the 32f signals *and* keeps working for the genuinely filterable ones (R8Unorm masks, Rgba16Float input color) — and declares the Lanczos LUT as `Float { filterable: true }`, which records that RCAS linear-samples it.
- `shaders/include/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h`: those fourteen "Sample*" helpers now use `s_PointClamp` instead of `s_LinearClamp`. Every call site passes an explicit LOD to textureLod, so point sampling is exact — it only drops bilinear smoothing within a level, and only on internal analysis signals (masks, weights, history), never the final color path. The helpers over genuinely filterable formats keep linear sampling: reactive/transparency masks (R8Unorm defaults), input color (Rgba16Float), dilated reactive masks (Rgba8Unorm), and the Lanczos LUT (R16Snorm — RCAS weight quality depends on it).
- All 40 `.spv` files under `wgpu-ffx-shaders-spv/src/fsr3upscaler/` recompiled in place with the recipe above. Changed bytes: the accumulate family (22 files, +52 B each), luma_instability (+52 B), debug_view (+52 B), prepare_reactivity (+52 B) and shading_change (same size, different bytes — its linear sampler became unused and was dropped from codegen); prepare_inputs ×8, both pyramids, rcas and autogen_reactive are byte-identical (they compile none of the edited helpers). All 32 permutation groups per file remain byte-consistent; all 40 pass spirv-val.

Behavior note: on drivers that *do* advertise FLOAT32_FILTERABLE this patch still point-samples those internal signals — a deliberate, portable choice rather than a capability branch (the .spv is one artifact). Quality impact is limited to sub-pixel smoothing of masks/weights/history lookups; the color path (RCAS with its Lanczos LUT) is untouched. Applications supplying reactive or transparency/composition masks should use a filterable format such as R8Unorm so those helpers' linear sampling stays conformant.

Verified: `cargo check -p bri-client --all-targets` and clippy on `bri-client` + `bri-ui` clean; spirv-val clean on all 40 modules; disassembly of the rebuilt shading_change module confirms its only sampler is now binding 3 (the point clamp) with an explicit-LOD sample of the Rg32f SPD mips. Awaiting a laptop run — this patch should clear bind group creation for every pass at once, since no SRV entry over-declares filterability anymore.

Upstream note: worth reporting to cwfitzgerald/wgpu-ffx alongside patches 2–4 — the sample-type declarations assume an adapter-specific capability that conformant drivers (NVIDIA) do not have, and the linear sampling of 32-bit float signals is non-conformant on them regardless.

## Tenth patch — per-format sample types for the linearly-sampled textures

The eleventh GPU validation (NVIDIA/Blackwell) got past every bind group layout and shader module but failed at compute pipeline *creation*: `Device::create_compute_pipeline` for "FSR3 Prepare Reactivity" rejected with "Unable to filter the texture (ResourceBinding { group: 0, binding: 4 }) by the sampler (ResourceBinding { group: 0, binding: 14 }) — Non-filterable float textures can't be sampled with a filtering sampler".

Root cause: wgpu-core's pipeline validation carries a *second* sample-type check the Ninth patch did not account for. For every sampling pair in an entry point (the texture/sampler handle pairs naga records from the SPIR-V), it rejects the pipeline when the BGL declares the sampler as `SamplerBindingType::Filtering` while declaring the texture as `Float { filterable: false }` — a pure cross-check between our two BGL entries, independent of the actual views and of what any driver supports. The Ninth patch's blanket non-filterable declaration satisfied the bind group check (any float view is accepted against such an entry) but broke this one for every texture FSR still linearly samples through `s_LinearClamp`. Auditing all 19 sampling sites in the callbacks header: exactly five remain linear — input color (the engine's Rgba16Float), the reactive-mask and transparency/composition slots (both bound to the crate's 1×1 R8Unorm default mask when the engine passes `None`), dilated reactive masks (Rgba8Unorm), and the Lanczos LUT (R16Snorm). PrepareReactivity's binding 4 is InputTransparencyAndComposition and binding 14 is s_LinearClamp — the first such pair reached. The other fourteen sites are point-sampled through `s_PointClamp` (Ninth patch) and never trip this check, because that sampler entry is NonFiltering.

Changes:

- `wgpu-ffx/src/resources.rs`: the SRV arm of `to_bgl_entry` now declares exactly those five names (`InputColor`, `InputReactiveMask`, `InputTransparencyAndComposition`, `DilatedReactiveMasks`, `Lanczos2Lut`) as `Float { filterable: true }` and everything else as `Float { filterable: false }`. Both checks are satisfied at once: the pipeline check passes because every filtering sampler now pairs with a filterable-declared texture, and the bind group check passes because wgpu-types computes only R32F/RG32F/RGBA32F views as non-filterable (gated on `FLOAT32_FILTERABLE`, which NVIDIA does not expose) — all five views above are unconditionally filterable formats.
- No shader changes; no `.spv` rebuild (the linear sites were already correct after the Ninth patch's audit).

Caveat: applications supplying reactive or transparency/composition masks should keep using a natively filterable format such as R8Unorm — those entries now *require* it, which is what FSR's mask design assumes anyway.

Verified headless: `cargo check -p bri-client --all-targets` and clippy clean; the 40 SPIR-V modules from the Ninth patch are unchanged. Awaiting a laptop run — this should clear compute pipeline creation for every pass at once, since no filtering sampler pairs with a non-filterable texture entry anymore.

## Ninth change (engine-side) — non-filterable source entries for FSR's convert/blit copies

Not a patch to this crate; it is in `crates/client/src/app/fsr.rs`. After the Tenth patch, every AMD pass validated and dispatched on NVIDIA/Blackwell and the failure moved into the engine's own final step: `Device::create_bind_group` for "fsr blit" rejected binding 0 with "Texture binding 0 expects sample type Float { filterable: true }, but was given a view with format Rgba32Float (sample type Float { filterable: false })". Root cause: the shared source entry (`src_entry`) of the convert and blit full-screen copies declared `Float { filterable: true }`, while both fragment shaders use only `textureLoad` — exact texel fetches, no sampler in sight — so filtering is not required at all. The blit's source is this crate's Rgba32Float output (the format AMD writes because no GPU exposes 16-bit float storage), whose views compute as non-filterable on NVIDIA, so the over-declaration was rejected.

Change: `src_entry` now declares `Float { filterable: false }`. A non-filterable entry accepts any float view, so convert keeps working against whatever world color format is bound (Rgba8Unorm in tests) and blit can read Rgba32Float on conformant drivers; with no sampler used anywhere in the module, wgpu-core's pipeline-level filtering cross-check has nothing to flag. `cargo check -p bri-client --all-targets` and clippy clean. Awaiting a laptop run.

## Tenth change (engine-side) — `CLEAR_TEXTURE` device feature for FSR's per-frame clears

Not a patch to this crate; it is in `crates/client/src/platform.rs`, `crates/ui/src/gpu.rs`, and the FSR gate in `crates/client/src/app/mod.rs`. After the ninth change, every AMD pass validated on NVIDIA/Blackwell and the failure moved into command *encoding*: "In a CommandEncoder — Features { CLEAR_TEXTURE } are required but not enabled on the device". Root cause: this crate clears its SPD mips with `CommandEncoder::clear_texture` every frame (unconditionally in lib.rs; plus AccumulationPrevious and SpdMips again inside the history-reset block), and wgpu gates that command behind `wgpu::Features::CLEAR_TEXTURE`, which neither device path requested. wgpu-hal grants the feature unconditionally on Vulkan — it is in the backend's base feature set, gated by no extension (adapter.rs) — so every conformant Vulkan driver exposes it; this was purely a missing request, and nothing in the crate needs to change.

Change: both device paths now also request `CLEAR_TEXTURE`, intersected with adapter features like the fifth and eighth changes, and `render_scene`'s FSR capability gate checks for it too, falling back to bilinear where absent (Vulkan always has it; other backends may not). An audit of every encoder/queue call in the crate found `clear_texture` its only feature-gated API — everything else is plain compute dispatches and buffer/texture copies. `cargo check -p bri-client --all-targets` and clippy clean. Awaiting a laptop run.
