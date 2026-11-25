//! FSR 3.1 temporal upscaling: AMD's FidelityFX Super Resolution through the
//! vendored `wgpu-ffx` port (`vendor/wgpu-ffx`, MIT; local changes in its
//! BRI-PATCH.md). When the Upscaler graphics option is on and Render Scale
//! draws fewer pixels than the window, the world pass renders with a
//! sub-pixel jittered camera into the render scale's texture, and this module
//! reconstructs it at the window's size:
//!
//! 1. `convert` copies the resolved world colour (the render scale's texture)
//!    as-is into an Rgba16Float input — both sides are display-encoded, so no
//!    conversion happens.
//! 2. `depth copy` moves the world-pass depth (Depth32Float, multisampled
//!    while MSAA is on) into a single-sample R32Float texture: wgpu will not
//!    let FSR's non-filterable float binding read a depth-format view.
//! 3. `motion vectors` unprojects each pixel through this frame's jittered
//!    camera, reprojects it through the previous frame's, and stores the
//!    screen-space delta — exact for camera motion over static geometry;
//!    moving objects ghost until per-object MVs arrive (Phase 2 of the plan).
//! 4. `FsrContext::dispatch` runs AMD's temporal passes into an Rgba32Float
//!    output at the window's size (FSR writes it as a storage image, and no
//!    GPU exposes 16-bit float formats for that), and `blit` copies it back
//!    to the window view as-is.
//!
//! History resets whenever `gpu.fsr` is dropped (device loss, map change) or
//! the sizes move (`begin_frame` reallocates); a discontinuous camera jump
//! resets it for one frame instead of smearing across the teleport.

use super::*;
use glam::{Mat4, Vec3};
use std::time::Instant;
use wgpu_ffx::{
    get_jitter_offset, get_jitter_phase_count, FsrContext, FsrContextFlags, FsrContextInfo,
    FsrDispatchFlags, FsrDispatchInfo,
};

/// The camera the world pass drew with this frame, decomposed for the
/// upscaler: its view and jittered projection matrices plus the projection's
/// planes. Built in the render pass where the engine builds its own `Camera`
/// (`render.rs`), so a zero offset gives exactly the engine matrix.
#[derive(Clone, Copy)]
pub(super) struct FrameCamera {
    /// World to view (right-handed, looking down -Z).
    pub(super) view: Mat4,
    /// Jittered projection: AMD's NDC translation over
    /// `bri_render::scene::perspective`'s reversed-Z one.
    pub(super) projection: Mat4,
    /// The true near plane (depth 1 in the world's reversed depth).
    pub(super) near: f32,
    /// The true far plane (depth 0).
    pub(super) far: f32,
    pub(super) tan_half_fov_y: f32,
    pub(super) aspect: f32,
}

impl FrameCamera {
    /// `bri_render::scene::Camera::oriented` with FSR's sub-pixel jitter
    /// folded into the projection. The basis and its degenerate fallback are
    /// mirrored exactly, so a zero offset reproduces the engine camera.
    #[allow(clippy::too_many_arguments)] // one per camera input; they change together
    pub(super) fn oriented(
        eye: Vec3,
        forward: Vec3,
        up: Vec3,
        aspect: f32,
        fov_y: f32,
        near: f32,
        far: f32,
        jitter: [f32; 2],
        render_size: (u32, u32),
    ) -> Self {
        let base = bri_render::scene::perspective(fov_y, aspect, near, far);
        // AMD's documented composition for the unit-pixel offsets
        // `get_jitter_offset` returns: a translation in NDC space, y negated
        // because pixel rows run down. It shifts every rendered point by
        // exactly (jx, jy) pixels and leaves depth untouched.
        let projection = if jitter[0] == 0.0 && jitter[1] == 0.0 {
            base
        } else {
            Mat4::from_translation(Vec3::new(
                2.0 * jitter[0] / render_size.0 as f32,
                -2.0 * jitter[1] / render_size.1 as f32,
                0.0,
            )) * base
        };
        Self {
            view: Self::view_matrix(eye, forward, up),
            projection,
            near,
            far,
            tan_half_fov_y: (fov_y * 0.5).tan(),
            aspect,
        }
    }

    /// The engine's `Camera::oriented` view decomposition (`bri_render/src/
    /// scene.rs`) with its fallback for a degenerate basis.
    fn view_matrix(eye: Vec3, forward: Vec3, up: Vec3) -> Mat4 {
        let forward = forward.normalize_or_zero();
        let up = up.normalize_or_zero();
        if forward.length_squared() >= 0.5
            && up.length_squared() >= 0.5
            && forward.cross(up).length_squared() >= 1e-6
        {
            return glam::camera::rh::view::look_to_mat4(eye, forward, up);
        }
        let target = eye + forward;
        let direction = (target - eye).normalize_or_zero();
        let target = if direction.length_squared() < 0.5 {
            eye - Vec3::Z
        } else {
            target
        };
        let up = if direction.dot(Vec3::Y).abs() > 0.9999 { Vec3::Z } else { Vec3::Y };
        glam::camera::rh::view::look_at_mat4(eye, target, up)
    }

    /// The vertical field of view in radians (what FSR's `camera_fov_y` wants).
    pub(super) fn fov_y(self) -> f32 {
        2.0 * self.tan_half_fov_y.atan()
    }

    /// A discontinuous camera move (teleport, a map mover's jump): the
    /// previous frame's history would reproject to garbage, so this frame
    /// resets it instead of smearing across the gap.
    pub(super) fn jumped_from(self, previous: &Self) -> bool {
        // The view matrix maps each eye to the origin; its inverse puts the
        // eyes back in world space for a true distance between them (the
        // translation column alone mixes in each frame's rotation).
        let eye = self.view.inverse().col(3);
        let prev_eye = previous.view.inverse().col(3);
        if (eye - prev_eye).length_squared() > 50.0 * 50.0 {
            return true;
        }
        // The cameras' backward axes (view space's +Z, the view matrix's
        // third row); more than ~45 degrees apart is a jump, not a turn.
        let a = Vec3::new(self.view.col(0).z, self.view.col(1).z, self.view.col(2).z)
            .normalize_or_zero();
        let b = Vec3::new(previous.view.col(0).z, previous.view.col(1).z, previous.view.col(2).z)
            .normalize_or_zero();
        // cos of the ~45-degree limit.
        a.dot(b) < std::f32::consts::FRAC_1_SQRT_2
    }
}

/// The view-space distance of a stored reversed-Z depth `d` (1 at the near
/// plane, 0 at the far one) for planes `near` and `far`. FSR's own
/// `GetViewSpaceDepth` computes the same value: `n*f / (d*(f-n) + n)`.
#[cfg(test)]
pub(super) fn reversed_depth_distance(d: f32, near: f32, far: f32) -> f32 {
    (near * far) / (d * (far - near) + near)
}

/// The view-space position of the centre of render-resolution pixel `[x, y]`
/// holding reversed depth `d`, for a camera with this `tan_half_fov_y` and
/// `aspect`. NDC x grows right and NDC y grows up (pixel rows run down); the
/// eye looks down -Z. The WGSL motion-vector pass does the same math; keep
/// the two in step.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // one per projection input, mirroring the WGSL uniforms
pub(super) fn view_position_from_depth(
    d: f32,
    x: u32,
    y: u32,
    size: (u32, u32),
    tan_half_fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> Vec3 {
    let ndc_x = 2.0 * (x as f32 + 0.5) / size.0 as f32 - 1.0;
    let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / size.1 as f32;
    Vec3::new(
        ndc_x * tan_half_fov_y * aspect,
        ndc_y * tan_half_fov_y,
        -1.0,
    ) * reversed_depth_distance(d, near, far)
}

/// Storage textures per compute stage FSR 3.1 needs: its luma pyramid pass
/// binds FrameInfo, six SPD mips and FarthestDepthMip1 in one bind group.
/// A GPU with fewer cannot build that layout (wgpu would panic), so
/// `render_scene` falls back to bilinear on such devices.
pub(super) const MIN_STORAGE_TEXTURES_PER_STAGE: u32 = 8;

/// The FSR 3.1 upscaler for the current device: AMD's compiled passes plus
/// this frame's scratch textures and temporal state. Created lazily on the
/// first scaled frame with Upscaler = FSR; dropped by `gpu_ready`/
/// `gpu_stopped`, which also clears its history.
pub(super) struct Fsr {
    /// The compiled AMD passes, one per flag set: they outlive resizes.
    context: FsrContext,
    /// The upscaler's internal textures and temporal state at the current
    /// sizes; `resize` reallocates them and resets history.
    view: Option<wgpu_ffx::FsrView>,
    jitter_index: i32,
    phase_count: i32,
    /// This cycle's Halton phases in nearest-neighbour order (rebuilt when
    /// `phase_count` changes); see `nearest_neighbor_jitter_tour`.
    jitter_tour: Vec<[f32; 2]>,
    /// This frame's offset (render-resolution pixels): applied to the world
    /// pass by `begin_frame`'s caller and passed to `dispatch`.
    current_jitter: [f32; 2],
    /// Milliseconds since the previous `begin_frame`, for FSR's
    /// `frame_time_delta`; None before the second frame.
    frame_dt_ms: Option<f32>,
    last_begin: Option<Instant>,
    /// The camera the last drawn frame used, for motion vectors and jump
    /// detection; None until a frame draws (history resets with it).
    previous_camera: Option<FrameCamera>,
    // Scratch at the current render/upscale sizes, recreated on resize.
    input_color: wgpu::Texture,
    depth_single: wgpu::Texture,
    motion_vectors: wgpu::Texture,
    dilated_depth: wgpu::Texture,
    dilated_motion_vectors: wgpu::Texture,
    reconstructed_previous_depth: wgpu::Buffer,
    output: wgpu::Texture,
    // Pipelines, layouts and modules (device-lifetime).
    shader_passes: wgpu::ShaderModule,
    convert_layout: wgpu::BindGroupLayout,
    convert_pipeline: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    /// One per window format the blit has drawn into; the window's format
    /// changes only on a surface reconfigure.
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    depth_copy_single_layout: wgpu::BindGroupLayout,
    depth_copy_single_pipeline: wgpu::ComputePipeline,
    depth_copy_msaa_layout: wgpu::BindGroupLayout,
    depth_copy_msaa_pipeline: wgpu::ComputePipeline,
    mv_layout: wgpu::BindGroupLayout,
    mv_pipeline: wgpu::ComputePipeline,
    /// The zero-motion variant for first frames and camera jumps.
    mv_clear_pipeline: wgpu::ComputePipeline,
    mv_uniform: wgpu::Buffer,
}

// (the motion-vector shader module is local to `new`: both pipelines keep
// their own copy, so the struct holds no reference to it)

/// The motion-vector pass's uniforms. Layout must match `MvUniforms` in the
/// WGSL below (same field order; both are 16-byte aligned with no gaps).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MvUniforms {
    /// The previous frame's view applied to this frame's view space:
    /// `View_prev * inv(View_curr)`, a pure rotation and translation.
    view_prev_from_curr: [f32; 16],
    /// The previous frame's jittered projection.
    proj_prev_jittered: [f32; 16],
    /// This frame's camera: near, far, tan(half fov y), aspect.
    current: [f32; 4],
    /// This frame's jitter offset in render-resolution pixels.
    jitter_current: [f32; 2],
    /// The render size in pixels.
    size: [f32; 2],
}

/// The full-screen triangle and the two identity copies (convert, blit). An
/// exact texel fetch on each side — both are display-encoded, so nothing
/// changes value.
const PASSES_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let p = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    return VsOut(vec4(p[i], 0.0, 1.0));
}

@group(0) @binding(0) var src: texture_2d<f32>;

@fragment
fn fs_convert(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<u32>(pos.xy), 0);
}

@group(0) @binding(0) var src_out: texture_2d<f32>;

@fragment
fn fs_blit(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src_out, vec2<u32>(pos.xy), 0);
}
"#;

const DEPTH_COPY_SINGLE_WGSL: &str = r#"
@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var out: texture_storage_2d<r32float, write>;

// The world-pass depth (single-sample) into FSR's R32Float input. A depth
// load is a scalar f32; the store takes a vec4 of which only the .r lands.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(out);
    if id.x >= dims.x || id.y >= dims.y { return; }
    let d = textureLoad(src_depth, vec2(id.x, id.y), 0);
    textureStore(out, vec2(id.x, id.y), vec4(d));
}
"#;

const DEPTH_COPY_MSAA_WGSL: &str = r#"
@group(0) @binding(0) var src_depth: texture_depth_multisampled_2d;
@group(0) @binding(1) var out: texture_storage_2d<r32float, write>;

// The world-pass depth (MSAA): sample 0 is what the colour resolve used. A
// depth load is a scalar f32; the store takes a vec4 of which only the .r
// lands.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(out);
    if id.x >= dims.x || id.y >= dims.y { return; }
    let d = textureLoad(src_depth, vec2(id.x, id.y), 0);
    textureStore(out, vec2(id.x, id.y), vec4(d));
}
"#;

/// The motion-vector pass: per pixel, unproject this frame's depth through
/// the jittered camera (undoing its NDC translation first — it shifts x and y
/// only, never depth), rotate into the previous frame's view space, project
/// through that frame's jittered projection, and store where the pixel's
/// content sat then minus where it sits now. FSR adds the stored vector to a
/// current position to find its history sample (and cancels the frames'
/// differing jitters itself), so the vector points from this frame toward the
/// previous one, in each frame's own jittered pixels.
const MV_WGSL: &str = r#"
struct MvUniforms {
    view_prev_from_curr: mat4x4<f32>,
    proj_prev_jittered: mat4x4<f32>,
    current: vec4<f32>,
    jitter_current: vec2<f32>,
    size: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: MvUniforms;
@group(0) @binding(1) var depth: texture_2d<f32>;
// rg32float (not rg16f): NVIDIA exposes no 16-bit float formats for
// storage images. WGSL's textureStore always takes a vec4; only the .xy lands
// in this RG image, which is what FSR reads.
@group(0) @binding(2) var mv_out: texture_storage_2d<rg32float, write>;

fn unproject(ndc: vec2<f32>, d: f32) -> vec3<f32> {
    // Reversed-Z (1 near, 0 far): dist = n*f / (d*(f-n)+n), the same as FSR's
    // GetViewSpaceDepth.
    let dist = (u.current.x * u.current.y) / (d * (u.current.y - u.current.x) + u.current.x);
    return vec3(ndc.x * u.current.z * u.current.w, ndc.y * u.current.z, -1.0) * dist;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(mv_out);
    if id.x >= dims.x || id.y >= dims.y { return; }
    let px = vec2<f32>(f32(id.x), f32(id.y));
    // Pixel centre to NDC (rows run down, so y flips).
    var ndc = 2.0 * (px + 0.5) / u.size - 1.0;
    ndc.y = -ndc.y + 1.0;
    // Undo this frame's jitter: its projection carried an NDC translation of
    // (2jx/w, -2jy/h).
    let t = vec2(2.0 * u.jitter_current.x / u.size.x, -2.0 * u.jitter_current.y / u.size.y);
    ndc -= t;
    var v = unproject(ndc, textureLoad(depth, vec2(id.x, id.y), 0).r);
    // This frame's view space to the previous one (both orthonormal: exact).
    v = (u.view_prev_from_curr * vec4(v, 1.0)).xyz;
    // Project through the previous frame's jittered projection.
    let clip = u.proj_prev_jittered * vec4(v, 1.0);
    if clip.w <= 0.0 {
        // Behind the previous eye: no valid reprojection. Zero reads as "no
        // motion"; FSR's depth checks treat it as a disocclusion.
        textureStore(mv_out, vec2(id.x, id.y), vec4<f32>(0.0));
        return;
    }
    let ndc_prev = clip.xy / clip.w;
    let px_prev = vec2(
        (ndc_prev.x + 1.0) * 0.5 * u.size.x,
        (1.0 - ndc_prev.y) * 0.5 * u.size.y,
    );
    textureStore(mv_out, vec2(id.x, id.y), vec4(px_prev - (px + 0.5), 0.0, 0.0));
}

// The first frame after a reset has no previous camera: store zero motion and
// let FSR start its history fresh.
@compute @workgroup_size(8, 8)
fn main_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(mv_out);
    if id.x >= dims.x || id.y >= dims.y { return; }
    textureStore(mv_out, vec2(id.x, id.y), vec4<f32>(0.0));
}
"#;

/// This cycle's Halton phases visited in an order that keeps each frame-to-
/// frame sub-pixel jump small (greedy nearest-neighbour over the phase set).
///
/// Why not AMD's plain index order: on Vulkan every one of FSR's 32-bit float
/// history textures is point-sampled (core Vulkan exposes no filterable f32
/// formats), so a jitter delta between consecutive frames shifts those reads
/// to different integer texels even for perfectly still content. The shading-
/// change, luma-instability and reactivity passes then flicker sub-pixel
/// every frame — the "heat shimmer" at a still camera. Visiting all phases
/// once per cycle keeps TAA's sample coverage identical; only their order
/// changes, shrinking the typical delta from ~0.25-0.5 px to well under 0.1.
fn nearest_neighbor_jitter_tour(phase_count: i32) -> Vec<[f32; 2]> {
    let phases: Vec<[f32; 2]> = (0..phase_count).map(|i| get_jitter_offset(i, phase_count)).collect();
    let n = phases.len();
    if n <= 1 {
        return phases;
    }
    let mut used = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    let mut current = 0usize;
    for step in 0..n {
        used[current] = true;
        tour.push(phases[current]);
        if step + 1 == n {
            break; // the cycle is complete
        }
        let from = phases[current];
        let mut best: Option<usize> = None;
        let mut best_d2 = f32::INFINITY;
        for (i, phase) in phases.iter().enumerate() {
            if used[i] {
                continue;
            }
            let dx = phase[0] - from[0];
            let dy = phase[1] - from[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = Some(i);
            }
        }
        current = best.expect("every step before the last has an unused phase");
    }
    tour
}

impl Fsr {
    /// Compiles AMD's passes (a one-time hitch on first enable) and the
    /// engine-side pipelines. Scratch resources follow in `begin_frame`.
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let context = FsrContext::new(FsrContextInfo {
            device: device.clone(),
            // The world's depth runs reversed (1 near, 0 far); the motion
            // vectors are at render size and carry the jitter pattern, which
            // the crate cancels. No HDR input (display-encoded colour) and no
            // auto exposure; sharpening stays off in v1.
            flags: FsrContextFlags::DEPTH_INVERTED | FsrContextFlags::MOTION_VECTORS_JITTER_CANCELLATION,
        });

        let shader_passes = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fsr passes"),
            source: wgpu::ShaderSource::Wgsl(PASSES_WGSL.into()),
        });
        let src_layout = |label: &str| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[src_entry(0)],
            })
        };
        let convert_layout = src_layout("fsr convert");
        let convert_pipeline = fullscreen_pipeline(
            device,
            &shader_passes,
            "fs_convert",
            wgpu::TextureFormat::Rgba16Float,
            &convert_layout,
        );
        // The window's format is unknown until the first frame; blit pipelines
        // are cached per format in `draw`.
        let blit_layout = src_layout("fsr blit");

        let depth_entries = |multisampled: bool| {
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
                storage_entry(1, wgpu::TextureFormat::R32Float),
            ]
        };
        let depth_copy_single_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("fsr depth copy"),
                entries: &depth_entries(false),
            },
        );
        let depth_copy_msaa_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("fsr depth copy msaa"),
                entries: &depth_entries(true),
            },
        );
        let pipeline_layout = |label: &str, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            })
        };
        let compute = |label: &str, module: &wgpu::ShaderModule, entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let depth_copy_single_layout_pl = pipeline_layout("fsr depth copy", &depth_copy_single_layout);
        let depth_copy_single_pipeline = compute(
            "fsr depth copy",
            &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fsr depth copy"),
                source: wgpu::ShaderSource::Wgsl(DEPTH_COPY_SINGLE_WGSL.into()),
            }),
            "main",
            &depth_copy_single_layout_pl,
        );
        let depth_copy_msaa_pipeline = compute(
            "fsr depth copy msaa",
            &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fsr depth copy msaa"),
                source: wgpu::ShaderSource::Wgsl(DEPTH_COPY_MSAA_WGSL.into()),
            }),
            "main",
            &pipeline_layout("fsr depth copy msaa", &depth_copy_msaa_layout),
        );

        let shader_mv = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fsr motion vectors"),
            source: wgpu::ShaderSource::Wgsl(MV_WGSL.into()),
        });
        let mv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fsr motion vectors"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Rg32Float to match MV_WGSL's storage image (NVIDIA has no
                // 16-bit float storage).
                storage_entry(2, wgpu::TextureFormat::Rg32Float),
            ],
        });
        // The clear variant binds the same three entries, so both pipelines
        // share one layout.
        let mv_pipeline_layout = pipeline_layout("fsr motion vectors", &mv_layout);
        let mv_pipeline = compute("fsr motion vectors", &shader_mv, "main", &mv_pipeline_layout);
        let mv_clear_pipeline = compute(
            "fsr motion vectors clear",
            &shader_mv,
            "main_clear",
            &mv_pipeline_layout,
        );

        Self {
            context,
            view: None,
            jitter_index: 0,
            phase_count: 0,
            jitter_tour: Vec::new(),
            current_jitter: [0.0; 2],
            frame_dt_ms: None,
            last_begin: None,
            previous_camera: None,
            // One texel of each, replaced by `begin_frame` before any draw;
            // FSR never reads them at these sizes.
            input_color: placeholder_texture(device, "fsr input color"),
            depth_single: placeholder_texture(device, "fsr depth copy"),
            motion_vectors: placeholder_texture(device, "fsr motion vectors"),
            dilated_depth: placeholder_texture(device, "fsr dilated depth"),
            dilated_motion_vectors: placeholder_texture(
                device,
                "fsr dilated motion vectors",
            ),
            reconstructed_previous_depth: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fsr reconstructed previous depth"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            output: placeholder_texture(device, "fsr output"),
            shader_passes,
            convert_layout,
            convert_pipeline,
            blit_layout,
            blit_pipelines: HashMap::new(),
            depth_copy_single_layout,
            depth_copy_single_pipeline,
            depth_copy_msaa_layout,
            depth_copy_msaa_pipeline,
            mv_layout,
            mv_pipeline,
            mv_clear_pipeline,
            mv_uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fsr motion vector uniforms"),
                size: std::mem::size_of::<MvUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Prepares this frame before the world pass draws: (re)allocates for a
    /// new size pair and returns the sub-pixel jitter the caller folds into
    /// the camera. None when the sizes are too small for FSR's internal
    /// half-resolution textures; the caller then falls back to bilinear.
    pub(super) fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_size: (u32, u32),
        upscale_size: (u32, u32),
    ) -> Option<[f32; 2]> {
        if render_size.0 < 2 || render_size.1 < 2 || upscale_size.0 == 0 || upscale_size.1 == 0 {
            return None;
        }
        let resized = self.view.as_ref().is_none_or(|v| {
            v.max_render_size() != [render_size.0, render_size.1]
                || v.max_upscale_size() != [upscale_size.0, upscale_size.1]
        });
        if resized {
            match &mut self.view {
                Some(view) => view.resize(
                    queue,
                    [render_size.0, render_size.1],
                    [upscale_size.0, upscale_size.1],
                ),
                None => {
                    self.view = Some(self.context.create_view(
                        queue,
                        [render_size.0, render_size.1],
                        [upscale_size.0, upscale_size.1],
                    ));
                }
            }
            self.reallocate(device, render_size, upscale_size);
            // Fresh history: the first frame after a resize has no previous
            // camera to reproject from.
            self.previous_camera = None;
        }
        let phase_count = get_jitter_phase_count(render_size.0 as i32, upscale_size.0 as i32).max(1);
        if phase_count != self.phase_count {
            // A new sequence length restarts the cycle at its head and
            // reorders the phases for small frame-to-frame jumps.
            self.phase_count = phase_count;
            self.jitter_index = 0;
            self.jitter_tour = nearest_neighbor_jitter_tour(phase_count);
        }
        let now = Instant::now();
        self.frame_dt_ms = Some(
            self.last_begin
                .map(|at| now.duration_since(at).as_secs_f32() * 1000.0)
                .unwrap_or(16.7),
        );
        self.last_begin = Some(now);
        self.current_jitter = self.jitter_tour[self.jitter_index as usize];
        self.jitter_index = (self.jitter_index + 1).rem_euclid(phase_count);
        Some(self.current_jitter)
    }

    /// Runs the upscaler for a drawn frame: convert, depth copy, motion
    /// vectors, AMD's passes, and the blit back to the window. False when the
    /// dispatch failed validation (the caller falls back to bilinear).
    #[allow(clippy::too_many_arguments)] // one per input FSR needs; they change together
    pub(super) fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::TextureView,
        engine_depth: &wgpu::Texture,
        camera: FrameCamera,
        window_view: &wgpu::TextureView,
        window_format: wgpu::TextureFormat,
        render_size: (u32, u32),
        upscale_size: (u32, u32),
    ) -> bool {
        let Some(view) = self.view.as_mut() else {
            return false; // begin_frame never ran this frame
        };

        // 1. World colour into the f16 input (identity copy).
        let input_view = self.input_color.create_view(&Default::default());
        {
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fsr convert"),
                layout: &self.convert_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_color),
                }],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fsr convert"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &input_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.convert_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2. World depth into the single-sample f32 input (FSR's binding is a
        // non-filterable float; a depth-format view cannot fill it).
        {
            let src_view = engine_depth.create_view(&Default::default());
            let out_view = self.depth_single.create_view(&Default::default());
            let msaa = engine_depth.sample_count() > 1;
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fsr depth copy"),
                layout: if msaa { &self.depth_copy_msaa_layout } else { &self.depth_copy_single_layout },
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&out_view),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(if msaa { &self.depth_copy_msaa_pipeline } else { &self.depth_copy_single_pipeline });
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(render_size.0.div_ceil(8), render_size.1.div_ceil(8), 1);
        }

        // 3. Motion vectors (or zeros on the first frame after a reset).
        let jumped = self.previous_camera.is_some_and(|prev| camera.jumped_from(&prev));
        {
            let depth_view = self.depth_single.create_view(&Default::default());
            let mv_view = self.motion_vectors.create_view(&Default::default());
            if let Some(previous) = &self.previous_camera {
                queue.write_buffer(
                    &self.mv_uniform,
                    0,
                    bytemuck::bytes_of(&MvUniforms {
                        view_prev_from_curr: (previous.view * camera.view.inverse())
                            .to_cols_array(),
                        proj_prev_jittered: previous.projection.to_cols_array(),
                        current: [
                            camera.near,
                            camera.far,
                            camera.tan_half_fov_y,
                            camera.aspect,
                        ],
                        jitter_current: self.current_jitter,
                        size: [render_size.0 as f32, render_size.1 as f32],
                    }),
                );
            }
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fsr motion vectors"),
                layout: &self.mv_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            self.mv_uniform.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&mv_view),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            // No previous camera (or a teleport): zero motion, fresh history.
            pass.set_pipeline(if self.previous_camera.is_some() && !jumped {
                &self.mv_pipeline
            } else {
                &self.mv_clear_pipeline
            });
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(render_size.0.div_ceil(8), render_size.1.div_ceil(8), 1);
        }

        // 4. AMD's temporal passes into the f32 output at window size.
        let dt = self.frame_dt_ms.unwrap_or(16.7).clamped(1.0, 250.0);
        let info = FsrDispatchInfo {
            color: self.input_color.clone(),
            depth: self.depth_single.clone(),
            motion_vectors: self.motion_vectors.clone(),
            // None: the crate binds its internal 1x1 exposure; with a constant
            // pre-exposure of 1.0 every multiply and divide cancels out.
            exposure: None,
            reactive_mask: None,
            transparency_and_composition: None,
            dilated_depth: self.dilated_depth.clone(),
            dilated_motion_vectors: self.dilated_motion_vectors.clone(),
            reconstructed_previous_depth: self.reconstructed_previous_depth.clone(),
            output: self.output.clone(),
            jitter_offset: self.current_jitter,
            // The stored vectors are in render-resolution pixels.
            motion_vector_scale: [1.0, 1.0],
            render_size: [render_size.0, render_size.1],
            upscale_size: [upscale_size.0, upscale_size.1],
            enable_sharpening: false, // RCAS stays off in v1; a tuning knob for playtests
            sharpness: 0.5,
            frame_time_delta: dt,
            pre_exposure: 1.0,
            reset_history: self.previous_camera.is_none() || jumped,
            // The inverted convention passes the planes swapped: what FSR
            // calls "near" is our far plane and vice versa (validation wants
            // near >= far under DEPTH_INVERTED).
            camera_near: camera.far,
            camera_far: camera.near,
            camera_fov_y: camera.fov_y(),
            view_space_to_meters_factor: 1.0,
            flags: FsrDispatchFlags::empty(),
        };
        if self.context.dispatch(view, encoder, &info).is_err() {
            eprintln!("fsr: dispatch failed validation; falling back to bilinear this frame");
            return false;
        }

        // 5. The f32 output back to the window view (identity copy).
        let blit_pipeline = self.blit_pipelines.entry(window_format).or_insert_with(|| {
            fullscreen_pipeline(
                device,
                &self.shader_passes,
                "fs_blit",
                window_format,
                &self.blit_layout,
            )
        });
        let output_view = self.output.create_view(&Default::default());
        {
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fsr blit"),
                layout: &self.blit_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                }],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fsr blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: window_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(blit_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }

        self.previous_camera = Some(camera);
        true
    }

    /// The scratch textures and buffer at the current sizes.
    fn reallocate(&mut self, device: &wgpu::Device, render_size: (u32, u32), upscale_size: (u32, u32)) {
        let tex = |label: &str, size: (u32, u32), format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let read = wgpu::TextureUsages::TEXTURE_BINDING;
        self.input_color = tex(
            "fsr input color",
            render_size,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | read,
        );
        self.depth_single = tex(
            "fsr depth copy",
            render_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::STORAGE_BINDING | read,
        );
        // Rg32Float: written by the MV compute pass as a storage image (NVIDIA
        // exposes no 16-bit float formats for that) and sampled by FSR.
        self.motion_vectors = tex(
            "fsr motion vectors",
            render_size,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureUsages::STORAGE_BINDING | read,
        );
        // FSR's dilated outputs and reconstructed previous depth: the crate
        // writes them each frame and reads them back in later passes.
        self.dilated_depth = tex(
            "fsr dilated depth",
            render_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::STORAGE_BINDING | read,
        );
        // Rg32Float: FSR's prepare inputs writes it as a storage image.
        self.dilated_motion_vectors = tex(
            "fsr dilated motion vectors",
            render_size,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureUsages::STORAGE_BINDING | read,
        );
        let bytes = u64::from(render_size.0 * render_size.1) * 4;
        self.reconstructed_previous_depth = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fsr reconstructed previous depth"),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Rgba32Float: FSR's accumulate writes it as a storage image; the blit
        // samples it back. (The input stays Rgba16Float — it is only attached
        // and sampled, never stored.)
        self.output = tex(
            "fsr output",
            upscale_size,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::STORAGE_BINDING | read,
        );
    }
}

/// A one-texel stand-in for a scratch texture until `begin_frame` sizes it
/// (wgpu's textures have no Default; FSR never reads these at this size).
fn placeholder_texture(device: &wgpu::Device, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

// The source entry for `fsr convert` and `fsr blit`. Both fragment shaders
// use only textureLoad (exact texel fetches, no sampler), so filtering is not
// required — declaring the entry non-filterable also lets the blit read FSR's
// Rgba32Float output, whose views are non-filterable on drivers without
// FLOAT32_FILTERABLE (NVIDIA).
fn src_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            format,
            access: wgpu::StorageTextureAccess::WriteOnly,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    fragment_entry: &str,
    target_format: wgpu::TextureFormat,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment_entry),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4Swizzles;

    /// A zero offset reproduces the engine's own camera exactly.
    #[test]
    fn a_zero_jitter_gives_the_engine_camera() {
        let eye = Vec3::new(1.5, 2.0, -3.25);
        let forward = Vec3::new(-0.4, -0.1, 0.9).normalize();
        let up = Vec3::Y;
        let (aspect, fov, near, far) = (16.0 / 9.0, 70.0f32.to_radians(), 0.05, 4000.0);
        let ours = FrameCamera::oriented(eye, forward, up, aspect, fov, near, far, [0.0; 2], (1600, 900));
        let engine = bri_render::scene::Camera::oriented(
            eye.to_array(),
            forward.to_array(),
            up.to_array(),
            aspect,
            fov,
            near,
            far,
        );
        assert_eq!(
            (ours.projection * ours.view).to_cols_array(),
            engine.view_projection,
        );
    }

    /// The degenerate basis falls back the way `Camera::oriented` does.
    #[test]
    fn a_degenerate_basis_falls_back_like_the_engine_camera() {
        let eye = Vec3::new(0.0, 10.0, 0.0);
        for (forward, up) in [
            (Vec3::ZERO, Vec3::Y),
            (Vec3::Y, Vec3::X),
            (-Vec3::Y, Vec3::X),
        ] {
            let (aspect, fov, near, far) = (1.0, 90.0f32.to_radians(), 0.05, 4000.0);
            let ours = FrameCamera::oriented(eye, forward, up, aspect, fov, near, far, [0.0; 2], (800, 600));
            let engine = bri_render::scene::Camera::oriented(
                eye.to_array(),
                forward.to_array(),
                up.to_array(),
                aspect,
                fov,
                near,
                far,
            );
            assert_eq!(
                (ours.projection * ours.view).to_cols_array(),
                engine.view_projection,
                "forward {forward:?} up {up:?}",
            );
        }
    }

    /// The jittered projection shifts NDC by exactly (2jx/w, -2jy/h), leaves
    /// depth untouched, and moves every point by exactly (jx, jy) pixels.
    #[test]
    fn the_jitter_shifts_ndc_by_the_documented_amount() {
        let eye = Vec3::ZERO;
        let forward = -Vec3::Z;
        let up = Vec3::Y;
        let (aspect, fov, near, far) = (16.0 / 9.0, 75.0f32.to_radians(), 0.05, 4000.0);
        let size = (1280u32, 720u32);
        let jitter = [0.31, -0.42];
        let cam = FrameCamera::oriented(eye, forward, up, aspect, fov, near, far, jitter, size);

        // A point on the view axis at mid distance.
        let p_view = Vec3::new(0.0, 0.0, -100.0);
        let p_clip = glam::vec4(p_view.x, p_view.y, p_view.z, 1.0);
        let clip_base = bri_render::scene::perspective(fov, aspect, near, far) * p_clip;
        let clip_jittered = cam.projection * p_clip;
        assert!((clip_base.w - clip_jittered.w).abs() < 1e-6, "jitter must not touch w");
        assert!((clip_base.z - clip_jittered.z).abs() < 1e-4, "jitter must not touch depth");

        let ndc = |c: glam::Vec4| c.xy() / c.w;
        let shift = ndc(clip_jittered) - ndc(clip_base);
        assert!((shift.x - 2.0 * jitter[0] / size.0 as f32).abs() < 1e-6, "{shift:?}");
        assert!((shift.y + 2.0 * jitter[1] / size.1 as f32).abs() < 1e-6, "{shift:?}");

        // In pixels the shift is exactly (jx, jy), rows down.
        let px = |c: glam::Vec4| {
            let n = ndc(c);
            ((n.x + 1.0) * 0.5 * size.0 as f32, (1.0 - n.y) * 0.5 * size.1 as f32)
        };
        let dpx = px(clip_jittered).0 - px(clip_base).0;
        let dpy = px(clip_jittered).1 - px(clip_base).1;
        assert!((dpx - jitter[0]).abs() < 1e-4, "{dpx}");
        assert!((dpy - jitter[1]).abs() < 1e-4, "{dpy}");
    }

    /// The reversed-Z unprojection inverts the engine's projection: project a
    /// view-space point, read off its pixel and device depth, unproject it
    /// back — within half a texel of where it started.
    #[test]
    fn the_depth_unprojection_round_trips_the_engine_projection() {
        let (aspect, fov, near, far) = (16.0 / 9.0, 70.0f32.to_radians(), 0.05, 4000.0);
        let size = (1600u32, 900u32);
        let proj = bri_render::scene::perspective(fov, aspect, near, far);
        for p_view in [
            Vec3::new(0.0, 0.0, -0.5),
            Vec3::new(12.5, -3.75, -42.0),
            Vec3::new(-88.0, 61.0, -999.0),
        ] {
            let clip = proj * glam::vec4(p_view.x, p_view.y, p_view.z, 1.0);
            let ndc = clip.xy() / clip.w;
            // Device depth of the point: near -> 1, far -> 0 (reversed-Z).
            let device_depth = clip.z / clip.w;
            let x = ((ndc.x + 1.0) * 0.5 * size.0 as f32 - 0.5).round() as u32;
            let y = ((1.0 - ndc.y) * 0.5 * size.1 as f32 - 0.5).round() as u32;
            let back = view_position_from_depth(
                device_depth,
                x,
                y,
                size,
                (fov * 0.5).tan(),
                aspect,
                near,
                far,
            );
            // One full texel of slack per axis covers the pixel rounding
            // (the point can sit on a pixel edge, half a texel from its
            // centre; the factor of two is that full texel).
            let t = p_view.z.abs();
            let tan_y = (fov * 0.5).tan();
            assert!(
                (back.x - p_view.x).abs() < 2.0 * t * tan_y * aspect / size.0 as f32,
                "x: {p_view:?} -> ({x}, {y}) -> {back:?}"
            );
            assert!(
                (back.y - p_view.y).abs() < 2.0 * t * tan_y / size.1 as f32,
                "y: {p_view:?} -> ({x}, {y}) -> {back:?}"
            );
            assert!((back.z - p_view.z).abs() < 1e-3 * t.max(1.0), "z: {p_view:?} -> {back:?}");
        }
    }

    /// `reversed_depth_distance` is FSR's `GetViewSpaceDepth`: the device depth
    /// of a plane distance inverts to that same distance, and FSR's own
    /// inverted-depth factors give it too.
    #[test]
    fn the_reversed_depth_matches_fsrs_view_space_depth() {
        let (near, far) = (0.05f32, 4000.0);
        // wgpu-ffx's setup_device_depth_to_view_space_depth_params with
        // DEPTH_INVERTED and a finite far: after its swap f_min is our far and
        // f_max our near; Z = e / (d - c).
        let f_min = far;
        let f_max = near;
        let f_q = f_max / (f_min - f_max);
        let c = -f_q;
        let e = f_q * f_min;
        for t in [0.05, 0.75, 2.0, 33.0, 480.0, 3999.0] {
            // Device depth of distance t under the engine's reversed-Z.
            let proj = bri_render::scene::perspective(1.0, 1.0, near, far);
            let clip = proj * glam::Vec4::new(0.0, 0.0, -t, 1.0);
            let d = clip.z / clip.w;
            assert!(
                (reversed_depth_distance(d, near, far) - t).abs() < t * 1e-3 + 1e-4,
                "ours: t {t} -> d {d}"
            );
            let fsr = e / (d - c);
            assert!((fsr - t).abs() < t * 1e-2 + 1e-3, "fsr: t {t} -> {fsr}");
        }
    }

    /// The tour visits every Halton phase exactly once per cycle, so TAA's
    /// sample coverage is identical to AMD's plain index order.
    #[test]
    fn the_jitter_tour_visits_every_phase_once() {
        for phase_count in [1, 2, 8, 32, 64] {
            let tour = nearest_neighbor_jitter_tour(phase_count);
            assert_eq!(tour.len(), phase_count as usize);
            let mut plain: Vec<[f32; 2]> =
                (0..phase_count).map(|i| get_jitter_offset(i, phase_count)).collect();
            let mut ordered = tour;
            let by_phase = |a: &[f32; 2], b: &[f32; 2]| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1]));
            plain.sort_by(by_phase);
            ordered.sort_by(by_phase);
            assert_eq!(plain, ordered, "phase_count {phase_count}");
        }
    }

    /// The tour's total path length (including the cycle wrap) is shorter than
    /// AMD's plain Halton order — that is what shrinks the per-frame jitter
    /// delta FSR's point-sampled history reads quantize.
    #[test]
    fn the_jitter_tour_keeps_consecutive_jumps_small() {
        let phase_count = 32;
        let plain: Vec<[f32; 2]> = (0..phase_count).map(|i| get_jitter_offset(i, phase_count)).collect();
        let tour = nearest_neighbor_jitter_tour(phase_count);
        let path_len = |seq: &[[f32; 2]]| {
            let mut total = 0.0f32;
            for w in seq.windows(2) {
                total += (w[0][0] - w[1][0]).abs() + (w[0][1] - w[1][1]).abs();
            }
            // The cycle wraps back to its first phase.
            let last = seq.last().expect("a tour of 32 phases is non-empty");
            let first = seq.first().expect("a tour of 32 phases is non-empty");
            total += (last[0] - first[0]).abs() + (last[1] - first[1]).abs();
            total
        };
        assert!(
            path_len(&tour) < path_len(&plain),
            "tour {} >= plain {}",
            path_len(&tour),
            path_len(&plain)
        );
    }

    /// A teleport or a sharp turn resets history; walking does not.
    #[test]
    fn jumps_reset_history_but_walking_does_not() {
        let base = FrameCamera::oriented(
            Vec3::ZERO,
            -Vec3::Z,
            Vec3::Y,
            16.0 / 9.0,
            70.0f32.to_radians(),
            0.05,
            4000.0,
            [0.0; 2],
            (1600, 900),
        );
        // A metre of walking and a slow turn: no jump.
        let walked = FrameCamera::oriented(
            Vec3::new(0.8, 0.0, -1.0),
            (-Vec3::Z * 0.9 + Vec3::X * 0.435).normalize(),
            Vec3::Y,
            16.0 / 9.0,
            70.0f32.to_radians(),
            0.05,
            4000.0,
            [0.0; 2],
            (1600, 900),
        );
        assert!(!walked.jumped_from(&base));
        // A teleport: a jump.
        let teleported = FrameCamera::oriented(
            Vec3::new(120.0, 5.0, -40.0),
            -Vec3::Z,
            Vec3::Y,
            16.0 / 9.0,
            70.0f32.to_radians(),
            0.05,
            4000.0,
            [0.0; 2],
            (1600, 900),
        );
        assert!(teleported.jumped_from(&base));
        // A 90-degree turn in place: a jump.
        let spun = FrameCamera::oriented(
            Vec3::ZERO,
            -Vec3::X,
            Vec3::Y,
            16.0 / 9.0,
            70.0f32.to_radians(),
            0.05,
            4000.0,
            [0.0; 2],
            (1600, 900),
        );
        assert!(spun.jumped_from(&base));
    }
}
