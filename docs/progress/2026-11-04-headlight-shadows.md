# 2026-11-04 Headlight shadow-casting experiment (Dynamic mode)

Local-only visual experiment: make vehicle headlights cast realtime shadows in
**Dynamic** lighting mode (`$pref::Video::Lighting=3`). Taillights are excluded
by design (only cone lights become candidates). Not intended to merge as-is.

Code lives on branch `bionic/headlight-shadows` (commit `000867472`, 14 files
across `crates/render` and `crates/client`).

## Current state (2026-11-04)

- **Works:** dynamic objects (bricks, players, items, vehicles) occlude the
  headlight beam — a crate or another vehicle between you and the light casts a
  shadow on the ground behind it.
- **Does not work:** static map geometry (interiors, walls) does not cast
  shadows for runtime headlights. A wall between eye and headlight does not
  block the light.

### Why the gap exists

Lamp shadow cubes get their casters from the dynamic caster pass in
`crates/render/src/scene.rs` (~line 5458), which draws moving/dynamic geometry
into each lamp slot's cube faces. Static map geometry only participates in
lamp visibility through the *baked* path (`lamp_reach`, `cube_seen`, map faces
in `scene.wgsl`), and that machinery is wired to the map's own shaded lamps —
not to runtime vehicle lamps. To close the gap, static/interior geometry would
need to be drawn as casters into the lamp slots' cubes (or the baked-visibility
machinery extended to cover runtime lamps).

## What was implemented

- `PointLight.lamp_slot: [f32; 4]` — shadow slot the light casts from, -1 when
  it does not. Full vec4 so the struct stays 64 bytes (WGSL uniform array
  stride); shader reads `.x`. See `crates/render/src/scene.rs`.
- `crates/client/src/app/render.rs` — vehicle lamp lights are hoisted before
  `update_camera`; cone lights become shadow candidates (`set_runtime_lamps`),
  and after slot selection each light is stamped with its slot (or -1).
- `crates/render/src/shadow.rs` — `ShadowMaps::lamp_slot_of(light)` maps a
  candidate back to the slot it won.
- Both shaders (`scene.wgsl`, `scene_original.wgsl`) — every live point-light
  contribution site multiplies by `lamp_lit(slot, ...)` when `lamp_slot.x >= 0`
  (point sums, vertex illumination, actor light, brick glint, both GGX
  highlight loops). Gated on Unified+Dynamic (`camera.ambient[3] >= 0.5`);
  Classic is untouched.
- Pipeline layout: bindings 6/7/8 of the scene camera group gained VERTEX
  visibility (vertex-lit surfaces read the shadow data too).

## How to test

1. Lighting = Dynamic, Shadow Quality = Best (4 lamp slots; shared with map
   lamps — brightest-nearest wins via `pick_lamps`).
2. Get in the jeep, headlights on (driver present + switch), look at ground or
   objects behind obstacles relative to a headlight.

## Verification status

- Format checker, `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  and `cargo check` all green; bri-render lib tests 85/85.
- GPU test `point_light_cone` passes (validates shader compile, pipeline layout,
  buffer sizes, pixel values).
- Full bri-render GPU integration suite not run to completion on the dev box
  (llvmpipe OOM under load — failures were non-deterministic/environmental).
