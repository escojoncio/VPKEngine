#!/usr/bin/env python3
"""Writes the geometry-pipeline test programs (es_*, gs_*, copy_*, rect_*, vs_misc*.s) in this folder.
Run it, then assemble.py, when one of them has to change. The layouts are the ones geom.h documents
(ES stores swizzled, GS loads linear, GS stores swizzled in 16-lane groups, copy loads linear)."""
import os

here = os.path.dirname(os.path.abspath(__file__))


def write(name, header, lines):
    with open(os.path.join(here, name + ".s"), "w") as f:
        for h in header:
            f.write("## " + h + "\n")
        for l in lines:
            f.write(l + "\n")


# ---- ES: two vec4 per vertex from a vertex buffer (V# in s[0:3], 32 byte stride) -> ring ----
write("es_basic", ["stage export", "user_sgprs 4", "geom esgs_item 8",
                   "vgpr_comp_cnt 0"],
      ["// ES: position and colour from the vertex buffer to the ES->GS ring (es2gs offset = s4).",
       "buffer_load_dwordx4 v[4:7], v0, s[0:3], 0 idxen",
       "buffer_load_dwordx4 v[8:11], v0, s[0:3], 0 idxen offset:16",
       "s_waitcnt vmcnt(0)",
       "buffer_store_dwordx4 v[4:7], off, s[8:11], s4 glc slc",
       "buffer_store_dwordx4 v[8:11], off, s[8:11], s4 offset:16 glc slc",
       "s_endpgm"])

VREG = [0, 1, 3]  # GS vertex offset VGPRs of vertices 0, 1, 2


def load_vertex(j, comps=8, first=16):
    out = [f"v_lshlrev_b32 v12, 2, v{VREG[j]}"]
    for c in range(comps):
        out += [f"s_movk_i32 s2, {c * 256}",
                f"buffer_load_dword v{first + c}, v12, s[4:7], s2 offen glc slc"]
    out.append("s_waitcnt vmcnt(0)")
    return out


def store_vertex(maxv, e_imm, comps=8, first=16):
    return [f"buffer_store_dword v{first + c}, off, s[8:11], s0 offset:{(c * maxv + e_imm) * 4} glc slc"
            for c in range(comps)]


GEOM = lambda item, maxv, inp, outp, inst=1: f"geom esgs_item 8 gsvs_vert {item} max_vert {maxv} in_prim {inp} out_prim {outp} gs_instances {inst}"

# Triangle passthrough.
lines = ["// GS: each input triangle out again (3 vertices, then a cut)."]
for j in range(3):
    lines += load_vertex(j) + store_vertex(3, j) + ["s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)"]
lines += ["s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)", "s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)", "s_endpgm"]
write("gs_tri", ["stage geometry", "user_sgprs 0", GEOM(8, 3, 4, 2)], lines)

# Stereo: both eyes, x shifted by -/+0.25, layer = eye; the emit index in a VGPR (dynamic offset).
lines = ["// GS: every triangle to layer 0 and layer 1, x shifted per eye (the PSVR single-pass stereo pattern).",
         "v_mov_b32 v30, 0"]
for eye in range(2):
    for j in range(3):
        lines += load_vertex(j)
        lines += [f"v_add_f32 v16, {'-0.25' if eye == 0 else '0.25'}, v16",
                  f"v_mov_b32 v24, {eye}"]
        for c in range(9):
            lines.append(f"buffer_store_dword v{16 + c}, v30, s[8:11], s0 offen offset:{c * 6 * 4} glc slc")
        lines += ["v_add_i32 v30, vcc, 4, v30", "s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)"]
    lines.append("s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)")
lines += ["s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)", "s_endpgm"]
write("gs_stereo", ["stage geometry", "user_sgprs 0", GEOM(9, 6, 4, 2)], lines)

# Strip of four: v0 v1 v2 and v0 + v2 - v1 (position), colour of v2.
lines = ["// GS: a triangle strip of four vertices per input triangle."]
for j in range(3):
    lines += load_vertex(j) + store_vertex(4, j) + ["s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)"]
lines += load_vertex(0, 2, 32)  # v32,v33 = v0.xy
lines += load_vertex(1, 2, 34)  # v34,v35 = v1.xy
lines += load_vertex(2, 8, 16)
lines += ["v_add_f32 v16, v32, v16", "v_sub_f32 v16, v16, v34", "v_add_f32 v17, v33, v17", "v_sub_f32 v17, v17, v35"]
lines += store_vertex(4, 3) + ["s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)"]
lines += ["s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)", "s_endpgm"]
write("gs_strip", ["stage geometry", "user_sgprs 0", GEOM(8, 4, 4, 2)], lines)

# Line strip of three.
lines = ["// GS: the triangle's edge loop as a line strip (3 vertices)."]
for j in range(3):
    lines += load_vertex(j) + store_vertex(3, j) + ["s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)"]
lines += ["s_endpgm"]
write("gs_lines", ["stage geometry", "user_sgprs 0", GEOM(8, 3, 4, 1)], lines)

# Conditional emit: only even primitive ids (EXEC masking), points out.
lines = ["// GS: one point (vertex 0) for even primitive ids only.",
         "v_and_b32 v13, 1, v2", "v_cmp_eq_u32 vcc, 0, v13",
         "s_and_saveexec_b64 s[12:13], vcc"]
lines += load_vertex(0) + store_vertex(1, 0) + ["s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)",
          "s_or_b64 exec, exec, s[12:13]", "s_endpgm"]
write("gs_cond_points", ["stage geometry", "user_sgprs 0", GEOM(8, 1, 4, 0)], lines)

# GS instancing: instance k shifts the triangle by 0.5 * k.
lines = ["// GS: two instances; instance (v7) shifts x by 0.5."]
lines += ["v_cvt_f32_u32 v14, v7", "v_mul_f32 v14, 0.5, v14"]
for j in range(3):
    lines += load_vertex(j) + ["v_add_f32 v16, v14, v16"] + store_vertex(3, j) + ["s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)"]
lines += ["s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)", "s_endpgm"]
write("gs_instanced", ["stage geometry", "user_sgprs 0", GEOM(8, 3, 4, 2, 2)], lines)


def copy_shader(maxv, comps, layer):
    lines = ["// Copy shader: position, colour (and layer) back from the GS->VS ring.",
             "s_mov_b32 vcc_hi, 0x03ff", "v_lshlrev_b32 v1, 2, v0"]
    for c in range(comps):
        lines += [f"s_movk_i32 s0, {c * maxv * 64}", f"buffer_load_dword v{4 + c}, v1, s[4:7], s0 offen glc slc"]
    lines.append("s_waitcnt vmcnt(0)")
    lines.append("exp pos0 v4, v5, v6, v7")
    if layer:
        lines.append("exp pos1 off, off, v12, off")
    lines.append("exp param0 v8, v9, v10, v11 done")
    lines.append("s_endpgm")
    return lines


write("copy_tri", ["stage copyvertex", "user_sgprs 0", "vgpr_comp_cnt 0", GEOM(8, 3, 4, 2)], copy_shader(3, 8, False))
write("copy_stereo", ["stage copyvertex", "user_sgprs 0", "vgpr_comp_cnt 0", "vs_out_cntl 0x40000", GEOM(9, 6, 4, 2)],
      copy_shader(6, 9, True))
write("copy_strip", ["stage copyvertex", "user_sgprs 0", "vgpr_comp_cnt 0", GEOM(8, 4, 4, 2)], copy_shader(4, 8, False))
write("copy_one", ["stage copyvertex", "user_sgprs 0", "vgpr_comp_cnt 0", GEOM(8, 1, 4, 0)], copy_shader(1, 8, False))

# RectList vertex shader: three vertices from the vertex id, uv in param0.
write("vs_rect", ["stage vertex", "user_sgprs 0", "vgpr_comp_cnt 0", "rect_list 1"],
      ["// Rectangle corners from the vertex id: (-1,-1) (1,-1) (-1,1); uv in param0.",
       "v_and_b32 v1, 1, v0", "v_cvt_f32_u32 v1, v1", "v_lshrrev_b32 v2, 1, v0", "v_cvt_f32_u32 v2, v2",
       "v_mad_f32 v3, v1, 2.0, -1.0", "v_mad_f32 v4, v2, 2.0, -1.0",
       "v_mov_b32 v5, 0", "v_mov_b32 v6, 1.0",
       "exp pos0 v3, v4, v5, v6", "exp param0 v1, v2, v5, v6 done", "s_endpgm"])

# pos1 / pos2 exports: point size, layer, clip distances.
write("vs_misc", ["stage vertex", "user_sgprs 0", "vgpr_comp_cnt 0", "vs_out_cntl 0x450003"],
      ["// Point size 2 and layer 1 in pos1, clip distances 0.5 and -0.5 in pos2.",
       "v_mov_b32 v1, 0", "v_mov_b32 v2, 2.0", "v_mov_b32 v3, 1", "v_mov_b32 v4, 0.5", "v_mov_b32 v5, -0.5",
       "exp pos0 v1, v1, v1, v1", "exp pos1 v2, off, v3, off", "exp pos2 v4, v5, off, off done", "s_endpgm"])
print("ok")
