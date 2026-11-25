//! NVIDIA Image Scaling (NVScaler): spatial upscaling of the scaled-down
//! world to window size when Render Scale draws fewer pixels than the window
//! and Upscaler = NIS. One compute pass from NVIDIA's MIT-licensed Image
//! Scaling SDK v1.0.3 (`vendor/nvidia-image-scaling`): a 6-tap phase-filtered
//! scale, four directional edge filters selected by an edge map, and adaptive
//! unsharp masking, all in the same pass.
//!
//! The pass writes a window-sized Rgba8Unorm intermediate, which a blit then
//! copies over the window: naga rejects format-less storage images (the SDK's
//! own declaration) and GLSL has no BGRA qualifier, so the window itself can
//! never be the storage image. A colour attachment accepts any 8-bit RGBA/
//! BGRA/sRGB surface.
//!
//! Unlike FSR 3.1 it is purely spatial — no sub-pixel jitter, no motion
//! vectors, no temporal history — so there is nothing for its output to
//! converge toward: a still camera draws identical frames, which is exactly
//! what FSR's shimmer lacks. The trade-off is that it reconstructs no detail
//! beyond the render scale's own pixels (a very good bilinear with edge-aware
//! sharpening). It only scales 50%..=100%; below `MIN_RENDER_SCALE`,
//! `render_scene` keeps the plain stretch.

use bri_console::Clamp;

/// NVScaler's sharpness on the SDK's 0..=1 slider (0: no sharpening, 0.5:
/// NVIDIA's reference middle). Fixed for now; a graphics option is a
/// follow-up if playtests want it.
const SHARPNESS: f32 = 0.5;

/// The lowest Render Scale NVScaler can scale from: its filter banks cover
/// 50%..=100%, so below this `render_scene` keeps the plain stretch.
pub(super) const MIN_RENDER_SCALE: u32 = 50;

/// The pass's per-frame configuration, in the exact field order of
/// `NIS_Main.glsl`'s `const_buffer` (the SDK's `NISConfig`, SDR mode — see
/// `vendor/nvidia-image-scaling/NIS_Config.h`).
// The layout is padding-free (28 four-byte words, verified by the derive),
// so `bytemuck::bytes_of` can hand it to the config buffer as-is.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct NisConfig {
    k_detect_ratio: f32,
    k_detect_thres: f32,
    k_min_contrast_ratio: f32,
    k_ratio_norm: f32,
    k_contrast_boost: f32,
    k_eps: f32,
    k_sharp_start_y: f32,
    k_sharp_scale_y: f32,
    k_sharp_strength_min: f32,
    k_sharp_strength_scale: f32,
    k_sharp_limit_min: f32,
    k_sharp_limit_scale: f32,
    k_scale_x: f32,
    k_scale_y: f32,
    k_dst_norm_x: f32,
    k_dst_norm_y: f32,
    k_src_norm_x: f32,
    k_src_norm_y: f32,
    k_input_viewport_origin_x: u32,
    k_input_viewport_origin_y: u32,
    k_input_viewport_width: u32,
    k_input_viewport_height: u32,
    k_output_viewport_origin_x: u32,
    k_output_viewport_origin_y: u32,
    k_output_viewport_width: u32,
    k_output_viewport_height: u32,
    reserved0: f32,
    reserved1: f32,
}

/// The SDK's `NVScalerUpdateConfig` (`NIS_Config.h`) for its SDR mode: our
/// world texture is display-encoded Rgba8Unorm. Returns None where the SDK
/// would refuse to run (a scale outside 50%..=100%).
pub(super) fn update_config(
    sharpness: f32,
    input_size: (u32, u32),
    output_size: (u32, u32),
) -> Option<NisConfig> {
    let sharpen = sharpness.clamped(0.0, 1.0) - 0.5; // the SDK's slider mapping
    let max_scale = if sharpen >= 0.0 { 1.25 } else { 1.75 };
    let min_scale = if sharpen >= 0.0 { 1.25 } else { 1.0 };
    let limit_scale = if sharpen >= 0.0 { 1.25 } else { 1.0 };

    // The SDK's SDR constants.
    const DETECT_RATIO: f32 = 2.0 * 1127.0 / 1024.0;
    const DETECT_THRES: f32 = 64.0 / 1024.0;
    const MIN_CONTRAST_RATIO: f32 = 2.0;
    const MAX_CONTRAST_RATIO: f32 = 10.0;
    const SHARP_START_Y: f32 = 0.45;
    const SHARP_END_Y: f32 = 0.9;

    let strength_min = (0.4 + sharpen * min_scale * 1.2).max(0.0);
    let strength_max = 1.6 + sharpen * max_scale * 1.8;
    let limit_min = (0.14 + sharpen * limit_scale * 0.32).max(0.1);
    let limit_max = 0.5 + sharpen * limit_scale * 0.6;

    let scale_x = input_size.0 as f32 / output_size.0 as f32;
    let scale_y = input_size.1 as f32 / output_size.1 as f32;
    if !(0.5..=1.0).contains(&scale_x) || !(0.5..=1.0).contains(&scale_y) {
        return None;
    }

    Some(NisConfig {
        k_detect_ratio: DETECT_RATIO,
        k_detect_thres: DETECT_THRES,
        k_min_contrast_ratio: MIN_CONTRAST_RATIO,
        k_ratio_norm: 1.0 / (MAX_CONTRAST_RATIO - MIN_CONTRAST_RATIO),
        k_contrast_boost: 1.0,
        k_eps: 1.0 / 255.0,
        k_sharp_start_y: SHARP_START_Y,
        k_sharp_scale_y: 1.0 / (SHARP_END_Y - SHARP_START_Y),
        k_sharp_strength_min: strength_min,
        k_sharp_strength_scale: strength_max - strength_min,
        k_sharp_limit_min: limit_min,
        k_sharp_limit_scale: limit_max - limit_min,
        k_scale_x: scale_x,
        k_scale_y: scale_y,
        k_dst_norm_x: 1.0 / output_size.0 as f32,
        k_dst_norm_y: 1.0 / output_size.1 as f32,
        k_src_norm_x: 1.0 / input_size.0 as f32,
        k_src_norm_y: 1.0 / input_size.1 as f32,
        // The whole texture is the viewport (the SDK leaves origins at zero).
        k_input_viewport_origin_x: 0,
        k_input_viewport_origin_y: 0,
        k_input_viewport_width: input_size.0,
        k_input_viewport_height: input_size.1,
        k_output_viewport_origin_x: 0,
        k_output_viewport_origin_y: 0,
        k_output_viewport_width: output_size.0,
        k_output_viewport_height: output_size.1,
        reserved0: 0.0,
        reserved1: 0.0,
    })
}

// The NVScaler filter banks, verbatim from the SDK's NIS_Config.h
// (`coef_scale` / `coef_usm`, 64 phases x 8 taps; the pass uses taps 0..5).
const COEF_SCALE: [[f32; 8]; 64] = [
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0029, -0.0127, 1.0, 0.0132, -0.0034, 0.0, 0.0, 0.0],
    [0.0063, -0.0249, 0.9985, 0.0269, -0.0068, 0.0, 0.0, 0.0],
    [0.0088, -0.0361, 0.9956, 0.0415, -0.0103, 0.0005, 0.0, 0.0],
    [0.0117, -0.0474, 0.9932, 0.0562, -0.0142, 0.0005, 0.0, 0.0],
    [0.0142, -0.0576, 0.9897, 0.0713, -0.0181, 0.0005, 0.0, 0.0],
    [0.0166, -0.0674, 0.9844, 0.0874, -0.022, 0.001, 0.0, 0.0],
    [0.0186, -0.0762, 0.9785, 0.104, -0.0264, 0.0015, 0.0, 0.0],
    [0.0205, -0.085, 0.9727, 0.1206, -0.0308, 0.002, 0.0, 0.0],
    [0.0225, -0.0928, 0.9648, 0.1382, -0.0352, 0.0024, 0.0, 0.0],
    [0.0239, -0.1006, 0.9575, 0.1558, -0.0396, 0.0029, 0.0, 0.0],
    [0.0254, -0.1074, 0.9487, 0.1738, -0.0439, 0.0034, 0.0, 0.0],
    [0.0264, -0.1138, 0.939, 0.1929, -0.0488, 0.0044, 0.0, 0.0],
    [0.0278, -0.1191, 0.9282, 0.2119, -0.0537, 0.0049, 0.0, 0.0],
    [0.0288, -0.1245, 0.917, 0.231, -0.0581, 0.0059, 0.0, 0.0],
    [0.0293, -0.1294, 0.9058, 0.251, -0.063, 0.0063, 0.0, 0.0],
    [0.0303, -0.1333, 0.8926, 0.271, -0.0679, 0.0073, 0.0, 0.0],
    [0.0308, -0.1367, 0.8789, 0.2915, -0.0728, 0.0083, 0.0, 0.0],
    [0.0308, -0.1401, 0.8657, 0.312, -0.0776, 0.0093, 0.0, 0.0],
    [0.0313, -0.1426, 0.8506, 0.333, -0.0825, 0.0103, 0.0, 0.0],
    [0.0313, -0.1445, 0.8354, 0.354, -0.0874, 0.0112, 0.0, 0.0],
    [0.0313, -0.146, 0.8193, 0.3755, -0.0923, 0.0122, 0.0, 0.0],
    [0.0313, -0.147, 0.8022, 0.3965, -0.0967, 0.0137, 0.0, 0.0],
    [0.0308, -0.1479, 0.7856, 0.4185, -0.1016, 0.0146, 0.0, 0.0],
    [0.0303, -0.1479, 0.7681, 0.4399, -0.106, 0.0156, 0.0, 0.0],
    [0.0298, -0.1479, 0.7505, 0.4614, -0.1104, 0.0166, 0.0, 0.0],
    [0.0293, -0.147, 0.7314, 0.4829, -0.1147, 0.0181, 0.0, 0.0],
    [0.0288, -0.146, 0.7119, 0.5049, -0.1187, 0.019, 0.0, 0.0],
    [0.0278, -0.1445, 0.6929, 0.5264, -0.1226, 0.02, 0.0, 0.0],
    [0.0273, -0.1431, 0.6724, 0.5479, -0.126, 0.0215, 0.0, 0.0],
    [0.0264, -0.1411, 0.6528, 0.5693, -0.1299, 0.0225, 0.0, 0.0],
    [0.0254, -0.1387, 0.6323, 0.5903, -0.1328, 0.0234, 0.0, 0.0],
    [0.0244, -0.1357, 0.6113, 0.6113, -0.1357, 0.0244, 0.0, 0.0],
    [0.0234, -0.1328, 0.5903, 0.6323, -0.1387, 0.0254, 0.0, 0.0],
    [0.0225, -0.1299, 0.5693, 0.6528, -0.1411, 0.0264, 0.0, 0.0],
    [0.0215, -0.126, 0.5479, 0.6724, -0.1431, 0.0273, 0.0, 0.0],
    [0.02, -0.1226, 0.5264, 0.6929, -0.1445, 0.0278, 0.0, 0.0],
    [0.019, -0.1187, 0.5049, 0.7119, -0.146, 0.0288, 0.0, 0.0],
    [0.0181, -0.1147, 0.4829, 0.7314, -0.147, 0.0293, 0.0, 0.0],
    [0.0166, -0.1104, 0.4614, 0.7505, -0.1479, 0.0298, 0.0, 0.0],
    [0.0156, -0.106, 0.4399, 0.7681, -0.1479, 0.0303, 0.0, 0.0],
    [0.0146, -0.1016, 0.4185, 0.7856, -0.1479, 0.0308, 0.0, 0.0],
    [0.0137, -0.0967, 0.3965, 0.8022, -0.147, 0.0313, 0.0, 0.0],
    [0.0122, -0.0923, 0.3755, 0.8193, -0.146, 0.0313, 0.0, 0.0],
    [0.0112, -0.0874, 0.354, 0.8354, -0.1445, 0.0313, 0.0, 0.0],
    [0.0103, -0.0825, 0.333, 0.8506, -0.1426, 0.0313, 0.0, 0.0],
    [0.0093, -0.0776, 0.312, 0.8657, -0.1401, 0.0308, 0.0, 0.0],
    [0.0083, -0.0728, 0.2915, 0.8789, -0.1367, 0.0308, 0.0, 0.0],
    [0.0073, -0.0679, 0.271, 0.8926, -0.1333, 0.0303, 0.0, 0.0],
    [0.0063, -0.063, 0.251, 0.9058, -0.1294, 0.0293, 0.0, 0.0],
    [0.0059, -0.0581, 0.231, 0.917, -0.1245, 0.0288, 0.0, 0.0],
    [0.0049, -0.0537, 0.2119, 0.9282, -0.1191, 0.0278, 0.0, 0.0],
    [0.0044, -0.0488, 0.1929, 0.939, -0.1138, 0.0264, 0.0, 0.0],
    [0.0034, -0.0439, 0.1738, 0.9487, -0.1074, 0.0254, 0.0, 0.0],
    [0.0029, -0.0396, 0.1558, 0.9575, -0.1006, 0.0239, 0.0, 0.0],
    [0.0024, -0.0352, 0.1382, 0.9648, -0.0928, 0.0225, 0.0, 0.0],
    [0.002, -0.0308, 0.1206, 0.9727, -0.085, 0.0205, 0.0, 0.0],
    [0.0015, -0.0264, 0.104, 0.9785, -0.0762, 0.0186, 0.0, 0.0],
    [0.001, -0.022, 0.0874, 0.9844, -0.0674, 0.0166, 0.0, 0.0],
    [0.0005, -0.0181, 0.0713, 0.9897, -0.0576, 0.0142, 0.0, 0.0],
    [0.0005, -0.0142, 0.0562, 0.9932, -0.0474, 0.0117, 0.0, 0.0],
    [0.0005, -0.0103, 0.0415, 0.9956, -0.0361, 0.0088, 0.0, 0.0],
    [0.0, -0.0068, 0.0269, 0.9985, -0.0249, 0.0063, 0.0, 0.0],
    [0.0, -0.0034, 0.0132, 1.0, -0.0127, 0.0029, 0.0, 0.0],
];

const COEF_USM: [[f32; 8]; 64] = [
    [0.0, -0.6001, 1.2002, -0.6001, 0.0, 0.0, 0.0, 0.0],
    [0.0029, -0.6084, 1.1987, -0.5903, -0.0029, 0.0, 0.0, 0.0],
    [0.0049, -0.6147, 1.1958, -0.5791, -0.0068, 0.0005, 0.0, 0.0],
    [0.0073, -0.6196, 1.189, -0.5659, -0.0103, 0.0, 0.0, 0.0],
    [0.0093, -0.6235, 1.1802, -0.5513, -0.0151, 0.0, 0.0, 0.0],
    [0.0112, -0.6265, 1.1699, -0.5352, -0.0195, 0.0005, 0.0, 0.0],
    [0.0122, -0.627, 1.1582, -0.5181, -0.0259, 0.0005, 0.0, 0.0],
    [0.0142, -0.6284, 1.1455, -0.5005, -0.0317, 0.0005, 0.0, 0.0],
    [0.0156, -0.6265, 1.1274, -0.479, -0.0386, 0.0005, 0.0, 0.0],
    [0.0166, -0.6235, 1.1089, -0.457, -0.0454, 0.001, 0.0, 0.0],
    [0.0176, -0.6187, 1.0879, -0.4346, -0.0532, 0.001, 0.0, 0.0],
    [0.0181, -0.6138, 1.0659, -0.4102, -0.0615, 0.0015, 0.0, 0.0],
    [0.019, -0.6069, 1.0405, -0.3843, -0.0698, 0.0015, 0.0, 0.0],
    [0.0195, -0.6006, 1.0161, -0.3574, -0.0796, 0.002, 0.0, 0.0],
    [0.02, -0.5928, 0.9893, -0.3286, -0.0898, 0.0024, 0.0, 0.0],
    [0.02, -0.582, 0.958, -0.2988, -0.1001, 0.0029, 0.0, 0.0],
    [0.02, -0.5728, 0.9292, -0.269, -0.1104, 0.0034, 0.0, 0.0],
    [0.02, -0.562, 0.8975, -0.2368, -0.1226, 0.0039, 0.0, 0.0],
    [0.0205, -0.5498, 0.8643, -0.2046, -0.1343, 0.0044, 0.0, 0.0],
    [0.02, -0.5371, 0.8301, -0.1709, -0.1465, 0.0049, 0.0, 0.0],
    [0.0195, -0.5239, 0.7944, -0.1367, -0.1587, 0.0054, 0.0, 0.0],
    [0.0195, -0.5107, 0.7598, -0.1021, -0.1724, 0.0059, 0.0, 0.0],
    [0.019, -0.4966, 0.7231, -0.0649, -0.1865, 0.0063, 0.0, 0.0],
    [0.0186, -0.4819, 0.6846, -0.0288, -0.1997, 0.0068, 0.0, 0.0],
    [0.0186, -0.4668, 0.646, 0.0093, -0.2144, 0.0073, 0.0, 0.0],
    [0.0176, -0.4507, 0.6055, 0.0479, -0.229, 0.0083, 0.0, 0.0],
    [0.0171, -0.437, 0.5693, 0.0859, -0.2446, 0.0088, 0.0, 0.0],
    [0.0161, -0.4199, 0.5283, 0.1255, -0.2598, 0.0098, 0.0, 0.0],
    [0.0161, -0.4048, 0.4883, 0.1655, -0.2754, 0.0103, 0.0, 0.0],
    [0.0151, -0.3887, 0.4497, 0.2041, -0.291, 0.0107, 0.0, 0.0],
    [0.0142, -0.3711, 0.4072, 0.2446, -0.3066, 0.0117, 0.0, 0.0],
    [0.0137, -0.3555, 0.3672, 0.2852, -0.3228, 0.0122, 0.0, 0.0],
    [0.0132, -0.3394, 0.3262, 0.3262, -0.3394, 0.0132, 0.0, 0.0],
    [0.0122, -0.3228, 0.2852, 0.3672, -0.3555, 0.0137, 0.0, 0.0],
    [0.0117, -0.3066, 0.2446, 0.4072, -0.3711, 0.0142, 0.0, 0.0],
    [0.0107, -0.291, 0.2041, 0.4497, -0.3887, 0.0151, 0.0, 0.0],
    [0.0103, -0.2754, 0.1655, 0.4883, -0.4048, 0.0161, 0.0, 0.0],
    [0.0098, -0.2598, 0.1255, 0.5283, -0.4199, 0.0161, 0.0, 0.0],
    [0.0088, -0.2446, 0.0859, 0.5693, -0.437, 0.0171, 0.0, 0.0],
    [0.0083, -0.229, 0.0479, 0.6055, -0.4507, 0.0176, 0.0, 0.0],
    [0.0073, -0.2144, 0.0093, 0.646, -0.4668, 0.0186, 0.0, 0.0],
    [0.0068, -0.1997, -0.0288, 0.6846, -0.4819, 0.0186, 0.0, 0.0],
    [0.0063, -0.1865, -0.0649, 0.7231, -0.4966, 0.019, 0.0, 0.0],
    [0.0059, -0.1724, -0.1021, 0.7598, -0.5107, 0.0195, 0.0, 0.0],
    [0.0054, -0.1587, -0.1367, 0.7944, -0.5239, 0.0195, 0.0, 0.0],
    [0.0049, -0.1465, -0.1709, 0.8301, -0.5371, 0.02, 0.0, 0.0],
    [0.0044, -0.1343, -0.2046, 0.8643, -0.5498, 0.0205, 0.0, 0.0],
    [0.0039, -0.1226, -0.2368, 0.8975, -0.562, 0.02, 0.0, 0.0],
    [0.0034, -0.1104, -0.269, 0.9292, -0.5728, 0.02, 0.0, 0.0],
    [0.0029, -0.1001, -0.2988, 0.958, -0.582, 0.02, 0.0, 0.0],
    [0.0024, -0.0898, -0.3286, 0.9893, -0.5928, 0.02, 0.0, 0.0],
    [0.002, -0.0796, -0.3574, 1.0161, -0.6006, 0.0195, 0.0, 0.0],
    [0.0015, -0.0698, -0.3843, 1.0405, -0.6069, 0.019, 0.0, 0.0],
    [0.0015, -0.0615, -0.4102, 1.0659, -0.6138, 0.0181, 0.0, 0.0],
    [0.001, -0.0532, -0.4346, 1.0879, -0.6187, 0.0176, 0.0, 0.0],
    [0.001, -0.0454, -0.457, 1.1089, -0.6235, 0.0166, 0.0, 0.0],
    [0.0005, -0.0386, -0.479, 1.1274, -0.6265, 0.0156, 0.0, 0.0],
    [0.0005, -0.0317, -0.5005, 1.1455, -0.6284, 0.0142, 0.0, 0.0],
    [0.0005, -0.0259, -0.5181, 1.1582, -0.627, 0.0122, 0.0, 0.0],
    [0.0005, -0.0195, -0.5352, 1.1699, -0.6265, 0.0112, 0.0, 0.0],
    [0.0, -0.0151, -0.5513, 1.1802, -0.6235, 0.0093, 0.0, 0.0],
    [0.0, -0.0103, -0.5659, 1.189, -0.6196, 0.0073, 0.0, 0.0],
    [0.0005, -0.0068, -0.5791, 1.1958, -0.6147, 0.0049, 0.0, 0.0],
    [0.0, -0.0029, -0.5903, 1.1987, -0.6084, 0.0029, 0.0, 0.0],
];

/// The bytes of a filter bank's 2x64 Rgba32Float texture: texel `(vIdx,
/// phase)` holds taps `4*vIdx..` of the phase's row, as the pass's
/// `LoadFilterBanksSh` fetches them.
pub(super) fn coef_texture_data(table: &[[f32; 8]; 64]) -> Vec<u8> {
    let mut data = Vec::with_capacity(64 * 2 * 16);
    for phase in table {
        for v_idx in 0..2 {
            for tap in 0..4 {
                data.extend_from_slice(&phase[v_idx * 4 + tap].to_le_bytes());
            }
        }
    }
    data
}

/// NVScaler for the current device and window format. Created lazily on the
/// first scaled frame with Upscaler = NIS; dropped by `gpu_ready`/
/// `gpu_stopped`. It holds no per-frame state, so toggling the upscaler or
/// resizing never resets anything.
pub(super) struct Nis {
    /// The window's format the blit draws into: a different one needs a new
    /// pipeline, so `render_scene` replaces us like it does `Upscale`.
    format: wgpu::TextureFormat,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    coef_scaler_view: wgpu::TextureView,
    coef_usm_view: wgpu::TextureView,
    /// This frame's `NisConfig`, written before each dispatch.
    config_buffer: wgpu::Buffer,
    /// The pass's window-sized Rgba8Unorm output, remade when the size
    /// changes; the blit copies it over the window each frame.
    output: Option<(wgpu::Texture, wgpu::TextureView, (u32, u32))>,
    blit_layout: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
}

impl Nis {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let coef_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("NIS NVScaler"),
            entries: &[
                // The per-frame configuration (see `NisConfig`).
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZero::new(std::mem::size_of::<NisConfig>() as u64)
                                .expect("the config buffer is never empty"),
                        ),
                    },
                    count: None,
                },
                // The linear clamp sampler the pass samples with.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The world's texture (filterable: the pass samples it).
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // The pass's output (written as a storage image). Always
                // Rgba8Unorm — the shader declares `rgba8`, and naga rejects
                // format-less images, so the window itself is never written
                // here; the blit below copies this over it.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // The filter banks (fetched unfiltered).
                coef_entry(4),
                coef_entry(5),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NIS NVScaler"),
            source: wgpu::ShaderSource::SpirV(std::borrow::Cow::Borrowed(
                bytemuck::cast_slice(include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../vendor/nvidia-image-scaling/NIS_Main.spv",
                ))),
            )),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NIS NVScaler"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NIS NVScaler"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        // The window's texture is never written by the pass itself: naga
        // rejects format-less storage images and GLSL has no BGRA qualifier,
        // so the pass writes an Rgba8Unorm intermediate and this blit copies
        // it over the window. A colour attachment accepts any 8-bit RGBA/
        // BGRA/sRGB surface, which a storage image could not.
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NIS NVScaler blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("NIS NVScaler blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NIS NVScaler blit"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("NIS NVScaler blit"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("NIS NVScaler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // The filter banks' textures (2x64 Rgba32Float each), built once.
        let coef = |label: &str, data: Vec<u8>| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 2,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &data[..],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(2 * 16),
                    rows_per_image: None,
                },
                texture.size(),
            );
            texture.create_view(&Default::default())
        };
        let coef_scaler_view = coef("NIS scaler coefficients", coef_texture_data(&COEF_SCALE));
        let coef_usm_view = coef("NIS USM coefficients", coef_texture_data(&COEF_USM));
        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NIS NVScaler config"),
            size: std::mem::size_of::<NisConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            format,
            pipeline,
            layout,
            sampler,
            coef_scaler_view,
            coef_usm_view,
            config_buffer,
            output: None,
            blit_layout,
            blit_pipeline,
        }
    }

    /// The pass's window-sized Rgba8Unorm output, remade when the size
    /// changes.
    fn output(&mut self, device: &wgpu::Device, size: (u32, u32)) -> &wgpu::TextureView {
        if self.output.as_ref().is_none_or(|o| o.2 != size) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("NIS NVScaler output"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.output = Some((texture, view, size));
        }
        &self.output.as_ref().unwrap().1
    }

    pub(super) fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Scales `source` (the render scale's texture) over the window-sized
    /// `dst`. Returns false where NVScaler can't run (a scale outside its
    /// 50%..=100% range), so the caller falls back to the plain stretch.
    #[allow(clippy::too_many_arguments)] // one per input; they change together
    pub(super) fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        src_size: (u32, u32),
        dst_size: (u32, u32),
    ) -> bool {
        let Some(config) = update_config(SHARPNESS, src_size, dst_size) else {
            return false;
        };
        queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&config));
        // Cloned so the mutable borrow of our output ends before the passes
        // below use the rest of us.
        let out_view = self.output(device, dst_size).clone();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NIS NVScaler"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.coef_scaler_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.coef_usm_view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("NIS NVScaler"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One 32x24 block of output pixels per workgroup.
        pass.dispatch_workgroups(dst_size.0.div_ceil(32), dst_size.1.div_ceil(24), 1);
        drop(pass); // the blit below takes the encoder's mutable borrow
        // Copy the intermediate over the window (one texel in, one out at
        // this size, so bilinear sampling is exact).
        let blit_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NIS NVScaler blit"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("NIS NVScaler blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
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
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &blit_bind, &[]);
        pass.draw(0..3, 0..1);
        true
    }
}

/// The blit's shader: a fullscreen triangle that copies the intermediate
/// over the window. At one-to-one scale `textureSample` lands on texel
/// centres, so it is an exact copy.
const BLIT_SHADER: &str = r"
@group(0) @binding(0) var out_texture:texture_2d<f32>;
@group(0) @binding(1) var out_sampler:sampler;
struct Out { @builtin(position) position:vec4<f32>, @location(0) uv:vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) i:u32)->Out {
    let uv=vec2<f32>(f32((i<<1u)&2u),f32(i&2u));
    return Out(vec4<f32>(uv.x*2.0-1.0,1.0-uv.y*2.0,0.0,1.0),uv);
}
@fragment fn fs_main(in:Out)->@location(0) vec4<f32> {
    return textureSample(out_texture,out_sampler,in.uv);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout the GLSL `const_buffer` expects: 28 words, no padding.
    #[test]
    fn the_config_is_the_glsl_const_buffer() {
        assert_eq!(std::mem::size_of::<NisConfig>(), 112);
        assert_eq!(std::mem::align_of::<NisConfig>(), 4);
    }

    /// The SDR values at the slider's middle, where every SDK constant is
    /// visible (half scale: the smallest NVScaler accepts).
    #[test]
    fn the_middle_config_matches_the_sdk() {
        let c = update_config(0.5, (160, 120), (320, 240)).expect("half scale runs");
        assert_eq!(c.k_detect_ratio, 2.0 * 1127.0 / 1024.0);
        assert_eq!(c.k_detect_thres, 64.0 / 1024.0);
        assert_eq!(c.k_min_contrast_ratio, 2.0);
        assert_eq!(c.k_ratio_norm, 1.0 / (10.0 - 2.0));
        assert_eq!(c.k_contrast_boost, 1.0);
        assert_eq!(c.k_eps, 1.0 / 255.0);
        assert_eq!(c.k_sharp_start_y, 0.45);
        assert!((c.k_sharp_scale_y - 1.0 / 0.45).abs() < 1e-6);
        // Slider at zero: the SDK's reference middle.
        assert_eq!(c.k_sharp_strength_min, 0.4);
        assert!((c.k_sharp_strength_scale - 1.2).abs() < 1e-6);
        assert!((c.k_sharp_limit_min - 0.14).abs() < 1e-6);
        assert!((c.k_sharp_limit_scale - 0.36).abs() < 1e-6);
        assert_eq!(c.k_scale_x, 0.5);
        assert_eq!(c.k_scale_y, 0.5);
        assert_eq!(c.k_src_norm_x, 1.0 / 160.0);
        assert_eq!(c.k_dst_norm_y, 1.0 / 240.0);
        assert_eq!((c.k_input_viewport_width, c.k_output_viewport_height), (160, 240));
    }

    /// Sharpness 0 maps to no sharpening and 1 to the SDK's strongest SDR
    /// settings; out-of-range values clamp like a slider.
    #[test]
    fn sharpness_maps_like_the_sdk_slider() {
        let none = update_config(0.0, (160, 120), (320, 240)).expect("half scale runs");
        assert_eq!(none.k_sharp_strength_min, 0.0);
        assert!((none.k_sharp_limit_min - 0.1).abs() < 1e-6);

        let full = update_config(1.0, (160, 120), (320, 240)).expect("half scale runs");
        assert!((full.k_sharp_strength_min - 1.15).abs() < 1e-6);
        assert!((full.k_sharp_limit_min - 0.34).abs() < 1e-6);

        // Beyond the slider's ends it holds, like the SDK clamps it.
        let over = update_config(7.0, (160, 120), (320, 240)).expect("half scale runs");
        assert_eq!(over.k_sharp_strength_min, full.k_sharp_strength_min);
    }

    /// NVScaler only scales 50%..=100%; outside that the SDK refuses and so
    /// do we (the caller then stretches bilinearly).
    #[test]
    fn out_of_range_scales_are_refused() {
        assert!(update_config(0.5, (80, 60), (320, 240)).is_none()); // quarter scale
        assert!(update_config(0.5, (160, 120), (160, 120)).is_some()); // 100%
        assert!(update_config(0.5, (320, 240), (160, 120)).is_none()); // downscale
    }

    /// The filter banks' texture packing: texel `(vIdx, phase)` holds taps
    /// `4*vIdx..` of the row, little-endian f32.
    #[test]
    fn the_coefficient_texture_packs_the_rows() {
        let data = coef_texture_data(&COEF_SCALE);
        assert_eq!(data.len(), 64 * 2 * 16);
        let texel = |x: usize, y: usize| -> [f32; 4] {
            let at = (y * 2 + x) * 16;
            bytemuck::cast_slice::<u8, f32>(&data[at..at + 16]).try_into().unwrap()
        };
        for (phase, taps) in COEF_SCALE.iter().enumerate() {
            assert_eq!(texel(0, phase).as_slice(), &taps[0..4]);
            assert_eq!(texel(1, phase).as_slice(), &taps[4..8]);
        }
    }

    /// The banks carry the SDK's numbers: the middle phase's centre tap and
    /// a spot value from each table.
    #[test]
    fn the_filter_banks_carry_the_sdk_values() {
        // The middle phase (half scale) is symmetric about its two centre taps.
        assert!((COEF_SCALE[32][2] - 0.6113).abs() < 1e-9);
        assert!((COEF_SCALE[0][2] - 1.0).abs() < 1e-9); // phase 0: a delta at the centre
        assert!((COEF_USM[0][1] + 0.6001).abs() < 1e-9);
    }
}
