#!/usr/bin/env python3
"""Rebuild FSR3 upscaler SPIR-V for wgpu-ffx-shaders-spv from the GLSL sources.

Recipe mirrors vendor/wgpu-ffx/BRI-PATCH.md ("Recompile recipe") and the
upstream xtask permutation table in perm.toml:

    glslc --target-env=vulkan1.2 -Ishaders/include <base+perm defines> \
        -fshader-stage=comp -O <pass.glsl> -o out.spv

Modes:
  --dry-run   build every file into a temp dir and byte-compare with the
              committed .spv files (validates the recipe; writes nothing)
  default     rebuild in place, optionally restricted to families via
              --only accumulate,luma_instability,... ; spirv-val each output

Run from anywhere: paths resolve relative to --root (default: repo root).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

BASE_DEFINES = [
    "FFX_GPU=1",
    "FFX_GLSL=1",
    "FFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
    "FFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
    "FFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
    "FFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
    "FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2",
    # perm.toml [base]: no wave ops in these modules
    "FFX_SPD_NO_WAVE_OPERATIONS=1",
    # perm.toml [permutations] fixed values
    "FFX_HALF=0",
    "FFX_WAVE64=0",
]

# axis name -> (mod.rs enum field position, define prefix)
AXES = [
    ("hdr_color_input", "FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT"),
    ("inverted_depth", "FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH"),
    ("jittered_motion_vectors", "FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS"),
    ("low_resolution_motion_vectors", "FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS"),
    ("reproject_use_lanczos_type", "FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE"),
]

ARM_RE = re.compile(
    r"\(\s*Fsr3upscalerOptionHdrColorInput::(Off|On),\s*"
    r"Fsr3upscalerOptionInvertedDepth::(Off|On),\s*"
    r"Fsr3upscalerOptionJitteredMotionVectors::(Off|On),\s*"
    r"Fsr3upscalerOptionLowResolutionMotionVectors::(Off|On),\s*"
    r"Fsr3upscalerOptionReprojectUseLanczosType::(Off|On),\s*"
    r"Half::(Off|On),\s*Wave64::(Off|On),?\s*\)\s*=>\s*Shaders\s*\{([^}]+)\}",
    re.S,
)


def parse_mod_rs(path):
    src = open(path).read()
    consts = {k.upper(): v for k, v in re.findall(r'SHADER_([0-9A-Fa-f]{8})\s*=\s*"([^"]+)"', src)}
    # filename -> list of (axis values tuple, sharpen: bool)
    refs = {}
    for m in ARM_RE.finditer(src):
        axes = [1 if v == "On" else 0 for v in m.groups()[:5]]
        body = m.group(8)  # the Shaders { ... } block (groups 1-7 are axis values)
        for field, sym in re.findall(r"(\w+)\s*:\s*(SHADER_[0-9A-Fa-f]{8})", body):
            fname = consts[sym.split("_")[1].upper()]
            sharpen = field.endswith("_sharpen")
            refs.setdefault(fname, []).append((axes, sharpen))
    return refs


def defines_for(axes, sharpen):
    d = list(BASE_DEFINES)
    for (name, prefix), v in zip(AXES, axes):
        d.append(f"{prefix}={v}")
    d.append(f"FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING={'1' if sharpen else '0'}")
    return ["-D" + x for x in d]


def family_of(fname):
    m = re.match(r"ffx_fsr3upscaler_(.+)_pass_[0-9a-f]{8}\.spv$", fname)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated pass families to rebuild (default: all)")
    args = ap.parse_args()

    spv_dir = os.path.join(args.root, "vendor/wgpu-ffx/wgpu-ffx-shaders-spv/src/fsr3upscaler")
    include_dir = os.path.join(args.root, "vendor/wgpu-ffx/shaders/include")
    entry_dir = os.path.join(args.root, "vendor/wgpu-ffx/shaders/src/fsr3upscaler")
    refs = parse_mod_rs(os.path.join(spv_dir, "mod.rs"))

    only = set(args.only.split(",")) if args.only else None
    files = sorted(f for f in refs if (only is None or family_of(f) in only))
    print(f"{len(files)} file(s) to build")

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for fname in files:
            entries = refs[fname]
            # Resolve axis values (5 permutation axes + the sharpen field
            # permutation); an axis referenced with both values is free for
            # this file (codegen-identical), pick 0 but verify by double build.
            vals = {i: sorted({e[0][i] for e in entries}) for i in range(5)}
            vals[6] = sorted({1 if e[1] else 0 for e in entries})
            axes = [v[0] if len(v) == 1 else 0 for v in list(vals.values())[:5]]
            sharpen = vals[6][0]

            m = re.match(r"ffx_fsr3upscaler_(.+)_pass_", fname)
            entry = os.path.join(entry_dir, "ffx_fsr3upscaler_" + m.group(1) + "_pass.glsl")
            out = os.path.join(tmp, fname)
            cmd = ["glslc", "--target-env=vulkan1.2", f"-I{include_dir}"] + defines_for(axes, sharpen) + [
                "-fshader-stage=comp", "-O", entry, "-o", out]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL {fname}: glslc:\n{r.stderr}")
                failures += 1
                continue

            # verify free axes produce identical bytes
            for i in list(range(5)) + [6]:
                if len(vals[i]) > 1:
                    alt_axes, alt_sharpen = list(axes), sharpen
                    if i < 5:
                        alt_axes[i] = 1 - alt_axes[i]
                    else:
                        alt_sharpen = not alt_sharpen
                    out2 = os.path.join(tmp, fname + ".alt")
                    r2 = subprocess.run(["glslc", "--target-env=vulkan1.2", f"-I{include_dir}"] + defines_for(alt_axes, alt_sharpen) + ["-fshader-stage=comp", "-O", entry, "-o", out2], capture_output=True, text=True)
                    name = AXES[i][0] if i < 5 else "apply_sharpening"
                    if r2.returncode != 0 or open(out2, "rb").read() != open(out, "rb").read():
                        print(f"FAIL {fname}: axis {name} not codegen-free (bytes differ)")
                        failures += 1

            committed = os.path.join(spv_dir, fname)
            if args.dry_run:
                same = open(committed, "rb").read() == open(out, "rb").read()
                print(f"{'OK  ' if same else 'DIFF'} {fname}")
                if not same:
                    failures += 1
            else:
                with open(committed, "wb") as f:
                    f.write(open(out, "rb").read())
                v = subprocess.run(["spirv-val", out], capture_output=True, text=True)
                status = "OK" if v.returncode == 0 else "VAL-FAIL"
                print(f"{status} {fname}" + ("" if v.returncode == 0 else f": {v.stderr.strip()}"))
                if v.returncode != 0:
                    failures += 1

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
