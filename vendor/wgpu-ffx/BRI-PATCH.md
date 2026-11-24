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
