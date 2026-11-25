# Vendored: NVIDIA Image Scaling SDK v1.0.3 (NVScaler)

Source: https://github.com/NVIDIAGameWorks/NVIDIAImageScaling, commit
`35e13ba316c98eeecf16f37eae70ce88019911f6`, vendored 2026-11-25. License: MIT
(`LICENSE`).

Files kept from the SDK's `NIS/` folder:

- `NIS_Main.glsl` — the NVScaler compute shader entry point (GLSL).
  **Modified:** the output image carries an explicit `rgba8` format. The
  SDK's original declaration is a format-less writeonly image2D, which wgpu
  30's naga rejects (`StorageImageWriteWithoutFormat` unsupported), and no
  other qualifier works either: GLSL has no BGRA format token and naga maps
  neither that nor any sRGB storage format. `rgba8` is the display-encoded
  8-bit type the SDK itself recommends for its output; the engine writes a
  window-sized Rgba8Unorm intermediate and blits it over the window (see
  `crates/client/src/app/nis.rs`).
- `NIS_Scaler.h` — the algorithm it includes (6-tap phase scale, four
  directional edge filters, adaptive unsharp masking) plus the standalone
  NVSharpen pass we do not use.
- `NIS_Config.h` — the SDK's C++ configuration helper and the filter-bank
  coefficient tables (`coef_scale`, `coef_usm`).

The engine side lives in `crates/client/src/app/nis.rs`:

- The SDR half of `NVScalerUpdateConfig` is ported to Rust there as
  `update_config`; keep it in step with `NIS_Config.h`.
- The coefficient tables are embedded verbatim as the `COEF_SCALE`/`COEF_USM`
  consts (regenerate them from `NIS_Config.h` if NVIDIA ever changes them).

Build artifact:

- `NIS_Main.spv` — compiled once for the engine, not rebuilt by cargo. To
  recompile after touching the GLSL (from this folder):

      glslc --target-env=vulkan1.2 -fshader-stage=comp -O NIS_Main.glsl -o NIS_Main.spv
      spirv-val NIS_Main.spv

  The result must carry only the `Shader` capability (`spirv-dis NIS_Main.spv
  | grep Capability`) — if `StorageImageWriteWithoutFormat` reappears, the
  explicit `rgba8` on the output image was lost.

The engine embeds it with `include_bytes!`; a stale one fails shader
validation at device setup, so the mismatch is loud.
