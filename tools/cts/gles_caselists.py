#!/usr/bin/env python3
"""dEQP-GLES2 / GLES3 / GLES31 case lists for vpgles (VPKEngine's OpenGL ES on VPVulkan) on macOS CI, sharded per module.

Port of visionquestpro/conformance/gles/caselists.py: groups in run order (foundations first), a deterministic CRC32
sample per group for "rapida" (stable across runs, so pass rates compare), the groups a UE4/Unity GLES game exercises.

Left out: geometry/tessellation shaders and everything vpgles does not offer by design (debug output, robustness, advanced
blend equations), the stress/performance/accuracy groups. The rest of ES 3.1 (compute, SSBO, images, atomics) is sampled thin
so its state shows up in the report.

Usage: gles_caselists.py <mustpass/gles/aosp_mustpass/main dir> <out dir> [--profile rapida|completa] [--shard-size N]
Writes <out>/<profile>/<module>-NNN.txt (one case per line), <out>/<profile>/index.tsv and <out>/matrix.json
({"include": [{"shard", "module", "file", "cases"}]}; a shard never mixes modules: each runs its own deqp-glesN).
"""
import argparse, json, os, zlib

# (module, group prefix after "dEQP-GLESx.", stride for "rapida", name prefixes (after the same head) left out)
GROUPS = [
    ("gles2", "info.", 1, []),
    ("gles2", "functional.prerequisite.", 1, []),
    ("gles2", "functional.implementation_limits.", 1, []),
    ("gles2", "functional.color_clear.", 2, []),
    ("gles2", "functional.buffer.", 4, []),
    ("gles2", "functional.texture.", 8, []),
    ("gles2", "functional.fbo.", 4, []),
    ("gles2", "functional.shaders.", 10, []),
    ("gles2", "functional.uniform_api.", 6, []),
    ("gles2", "functional.attribute_location.", 4, []),
    ("gles2", "functional.vertex_arrays.", 12, []),
    ("gles2", "functional.draw.", 4, []),
    ("gles2", "functional.fragment_ops.", 8, []),
    ("gles2", "functional.depth_range.", 2, []),
    ("gles2", "functional.polygon_offset.", 2, []),
    ("gles2", "functional.read_pixels.", 2, []),
    ("gles2", "functional.state_query.", 4, []),
    ("gles2", "functional.negative_api.", 2, []),
    ("gles2", "functional.multisampled_render_to_texture.", 1, []),
    ("gles2", "functional.", 20,
     ["functional.prerequisite.", "functional.implementation_limits.", "functional.color_clear.",
      "functional.buffer.", "functional.texture.", "functional.fbo.", "functional.shaders.",
      "functional.uniform_api.", "functional.attribute_location.", "functional.vertex_arrays.",
      "functional.draw.", "functional.fragment_ops.", "functional.depth_range.",
      "functional.polygon_offset.", "functional.read_pixels.", "functional.state_query.",
      "functional.negative_api.", "functional.multisampled_render_to_texture.", "functional.debug_marker."]),

    ("gles3", "info.", 1, []),
    ("gles3", "functional.prerequisite.", 1, []),
    ("gles3", "functional.implementation_limits.", 1, []),
    ("gles3", "functional.color_clear.", 2, []),
    ("gles3", "functional.buffer.", 6, []),
    ("gles3", "functional.texture.", 12, []),
    ("gles3", "functional.fbo.", 8, []),
    ("gles3", "functional.shaders.", 12, []),
    ("gles3", "functional.ubo.", 8, []),
    ("gles3", "functional.uniform_api.", 8, []),
    ("gles3", "functional.draw.", 6, []),
    ("gles3", "functional.instanced.", 4, []),
    ("gles3", "functional.fragment_out.", 4, []),
    ("gles3", "functional.fragment_ops.", 24, []),
    ("gles3", "functional.samplers.", 4, []),
    ("gles3", "functional.sync.", 1, []),
    ("gles3", "functional.state_query.", 6, []),
    ("gles3", "functional.negative_api.", 2, []),
    ("gles3", "functional.transform_feedback.", 24, []),
    ("gles3", "functional.", 24,
     ["functional.prerequisite.", "functional.implementation_limits.", "functional.color_clear.",
      "functional.buffer.", "functional.texture.", "functional.fbo.", "functional.shaders.",
      "functional.ubo.", "functional.uniform_api.", "functional.draw.", "functional.instanced.",
      "functional.fragment_out.", "functional.fragment_ops.", "functional.samplers.", "functional.sync.",
      "functional.state_query.", "functional.negative_api.", "functional.transform_feedback.",
      "functional.shader_api.program_binary.", "functional.lifetime."]),

    ("gles31", "info.", 1, []),
    ("gles31", "functional.compute.", 2, []),
    ("gles31", "functional.shaders.", 16,
     ["functional.shaders.builtin_functions.precision.", "functional.shaders.opaque_type_indexing.ssbo",
      "functional.shaders.opaque_type_indexing.atomic_counter", "functional.shaders.linkage.es31.geometry",
      "functional.shaders.linkage.es31.tessellation"]),
    ("gles31", "functional.texture.", 12, ["functional.texture.texture_buffer.", "functional.texture.border_clamp."]),
    ("gles31", "functional.state_query.", 8, []),
    ("gles31", "functional.fbo.", 4, []),
    ("gles31", "functional.ubo.", 12, []),
    ("gles31", "functional.draw_indirect.", 4, []),
    ("gles31", "functional.program_uniform.", 16, []),
    ("gles31", "functional.uniform_location.", 8, []),
    ("gles31", "functional.layout_binding.", 4, []),
    ("gles31", "functional.ssbo.", 24, []),
    ("gles31", "functional.image_load_store.", 12, []),
    ("gles31", "functional.atomic_counter.", 8, []),
    ("gles31", "functional.synchronization.", 8, []),
    ("gles31", "functional.vertex_attribute_binding.", 2, []),
    ("gles31", "functional.stencil_texturing.", 1, []),
    ("gles31", "functional.program_interface_query.", 60, []),
    ("gles31", "functional.separate_shader.", 12, []),
    ("gles31", "functional.copy_image.", 150, []),
]

# Never run: geometry / tessellation stages, debug output, robustness, advanced blend, sample shading, stress, performance.
GLOBAL_EXCLUDE = [
    "functional.geometry_shading.", "functional.tessellation.", "functional.tessellation_geometry_interaction.",
    "functional.debug.", "functional.robustness.", "functional.blend_equation_advanced.",
    "functional.sample_shading.", "functional.primitive_bounding_box.", "stress.", "performance.", "accuracy.",
    "functional.shaders.implicit_conversions.",  # need GL_EXT_shader_implicit_conversions (not offered): always NotSupported
    "functional.android_extension_pack.",  # GL_ANDROID_extension_pack_es31a (ES 3.2 stages): not offered, ES 3.1 mustpass has none
]
# Cases of the allowed groups that need a geometry or tessellation stage (not offered; glCreateShader of it is INVALID_ENUM).
NAME_EXCLUDE = ("_geometry", "_tess_control", "_tess_eval", "tessellation", "geometry_shader", "geometry_shading")


def read_list(path):
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip() and not l.startswith("#")]


def keep_sample(name, stride):
    return stride <= 1 or zlib.crc32(name.encode()) % stride == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mustpass")
    ap.add_argument("out")
    ap.add_argument("--profile", default="rapida", choices=("rapida", "completa"))
    ap.add_argument("--shard-size", type=int, default=1500)
    ap.add_argument("--modules", default="gles2,gles3,gles31")
    a = ap.parse_args()
    mods = a.modules.split(",")
    per_module = {m: [] for m in mods}
    cache = {}
    for module, prefix, stride, skip in GROUPS:
        if module not in per_module:
            continue
        path = os.path.join(a.mustpass, f"{module}-main.txt")
        if not os.path.exists(path):
            print(f"warning: {path} missing")
            continue
        if path not in cache:
            cache[path] = read_list(path)
        head = f"dEQP-{module.upper()}."
        label = prefix.rstrip(".").replace(".", "-")
        for n in cache[path]:
            if (n.startswith(head + prefix) and not any(n.startswith(head + s) for s in skip)
                    and not any(n.startswith(head + g) for g in GLOBAL_EXCLUDE)
                    and not any(x in n for x in NAME_EXCLUDE)
                    and (a.profile == "completa" or keep_sample(n, stride))):
                per_module[module].append((label, n))
    d = os.path.join(a.out, a.profile)
    os.makedirs(d, exist_ok=True)
    include, rows, total = [], [], 0
    for module in mods:
        seen, uniq = set(), []
        for label, n in per_module[module]:
            if n not in seen:
                seen.add(n)
                uniq.append((label, n))
        for i in range(0, len(uniq), a.shard_size):
            chunk = uniq[i:i + a.shard_size]
            sid = f"{module}-{i // a.shard_size + 1:03d}"
            fname = f"{sid}.txt"
            with open(os.path.join(d, fname), "w") as fh:
                fh.write("\n".join(n for _, n in chunk) + "\n")
            rows.append((sid, len(chunk), chunk[0][0], chunk[-1][0]))
            include.append({"shard": sid, "module": module, "file": f"{a.profile}/{fname}", "cases": len(chunk)})
        total += len(uniq)
        print(f"{module}: {len(uniq)} cases")
    with open(os.path.join(d, "index.tsv"), "w") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    with open(os.path.join(a.out, "matrix.json"), "w") as fh:
        json.dump({"include": include}, fh)
    print(f"{a.profile}: {total} cases in {len(include)} shards of up to {a.shard_size}")


if __name__ == "__main__":
    main()
