#!/usr/bin/env python3
"""Vulkan CTS case lists for the VPVulkan run on macOS (CI), sharded for a job matrix.

Same selection as VisionQuestPro's in-headset run (visionquestpro/conformance/vulkan/caselists.py): the groups a Quest
game exercises, in run order (cheap, foundational groups first), and a deterministic sample (CRC32 of the case name)
so the set is stable across runs and the pass rate is comparable from one run to the next.

Profiles
  rapida    sample of every relevant group (~25k cases).
  completa  every case of the relevant groups (~1.5M; for later).

Usage: caselists.py <mustpass/main dir> <out dir> [--profile rapida|completa] [--shard-size N]
Writes <out>/<profile>/shard-NNN.txt (one case name per line), <out>/<profile>/index.tsv
("shard<TAB>count<TAB>first group<TAB>last group") and <out>/matrix.json (GitHub Actions matrix: {"include": [...]}).
"""
import argparse, json, os, zlib

# group (mustpass file stem, or directory for nested lists), stride for "rapida", and name
# prefixes (after "dEQP-VK.") that are left out. Order = run order.
GROUPS = [
    ("info",                  1,  []),
    ("api",                   3,  ["api.copy_and_blit.", "api.image_clearing.", "api.external.",
                                   "api.image_compression_control.", "api.device_group.",
                                   "api.object_management.max_concurrent.", "api.object_management.multithreaded"]),
    ("api",                 120,  None),   # second pass over api: only the heavy sub-groups, sampled
    ("memory",                4,  ["memory.external_memory_host.", "memory.device_group_allocation."]),
    ("multiview",             1,  []),
    ("binding-model",        60,  ["binding_model.mutable_descriptor.", "binding_model.descriptor_buffer.",
                                   "binding_model.descriptor_heap."]),
    ("pipeline/monolithic", 150,  []),
    ("renderpasses",         24,  ["renderpasses.dynamic_rendering."]),
    ("draw",                  6,  ["draw.dynamic_rendering."]),
    ("depth",                 6,  []),
    ("clipping",              2,  []),
    ("rasterization",         12,  []),
    ("fragment-operations",   4,  []),
    ("dynamic-state",         4,  []),
    ("texture",               12,  []),
    ("image",                40,  ["image.host_image_copy.", "image.swapchain_mutable."]),
    ("compute",              20,  ["compute.shader_object_spirv.", "compute.shader_object_binary."]),
    ("glsl",                  12,  []),
    ("ubo",                  20,  []),
    ("ssbo",                 20,  []),
    ("spirv-assembly",       60,  []),
    ("query-pool",           20,  []),
    ("synchronization",      50,  []),
    ("synchronization2",     80,  []),
    ("subgroups",            40,  []),
    ("geometry",              6,  []),
    ("tessellation",          6,  []),
    ("robustness",           120,  []),
]
# The heavy api sub-groups the second "api" entry samples (and the first one skips).
API_HEAVY = ["api.copy_and_blit.", "api.image_clearing."]

def read_list(path):
    out = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith(".txt"):
                    out += read_list(os.path.join(root, f))
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def keep_sample(name, stride):
    return stride <= 1 or zlib.crc32(name.encode()) % stride == 0


def keep_sample(name, stride):
    return stride <= 1 or zlib.crc32(name.encode()) % stride == 0


def selected(base, profile):
    """[(group label, case name)] in run order."""
    out = []
    for group, stride, skip in GROUPS:
        path = os.path.join(base, group)
        path = path if os.path.isdir(path) else path + ".txt"
        if not os.path.exists(path):
            print(f"warning: {group} not in this mustpass")
            continue
        names = read_list(path)
        if skip is None:          # the sampled pass over the heavy api groups
            picked = [n for n in names if any(n.startswith("dEQP-VK." + p) for p in API_HEAVY)]
        else:
            picked = [n for n in names if not any(n.startswith("dEQP-VK." + p) for p in skip)]
        label = group.replace("/", "-") + ("-heavy" if skip is None else "")
        s = stride if profile == "rapida" else 1
        if profile == "completa" and skip is None:
            continue              # the heavy api groups are already in the full "api" pass
        out += [(label, n) for n in picked if keep_sample(n, s)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mustpass")
    ap.add_argument("out")
    ap.add_argument("--profile", default="rapida", choices=("rapida", "completa"))
    ap.add_argument("--shard-size", type=int, default=1000)
    a = ap.parse_args()
    cases = selected(os.path.join(a.mustpass, "vk-default"), a.profile)
    seen, uniq = set(), []
    for label, n in cases:        # a case may appear in two groups' files: it runs once
        if n not in seen:
            seen.add(n)
            uniq.append((label, n))
    d = os.path.join(a.out, a.profile)
    os.makedirs(d, exist_ok=True)
    include, rows = [], []
    for i in range(0, len(uniq), a.shard_size):
        chunk = uniq[i:i + a.shard_size]
        sid = f"{i // a.shard_size + 1:03d}"
        fname = f"shard-{sid}.txt"
        with open(os.path.join(d, fname), "w") as fh:
            fh.write("\n".join(n for _, n in chunk) + "\n")
        rows.append((sid, len(chunk), chunk[0][0], chunk[-1][0]))
        include.append({"shard": sid, "file": f"{a.profile}/{fname}", "cases": len(chunk)})
    with open(os.path.join(d, "index.tsv"), "w") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    with open(os.path.join(a.out, "matrix.json"), "w") as fh:
        json.dump({"include": include}, fh)
    print(f"{a.profile}: {len(uniq)} cases in {len(include)} shards of up to {a.shard_size}")


if __name__ == "__main__":
    main()
