#!/usr/bin/env python3
# Writes subgroup_cube.comp.spv: a compute shader hand-assembled (no glslang needed) covering
#  * imageLoad from a cube-array storage image: the texel coordinate is (x, y, layer * 6 + face),
#    which Metal's texturecube_array takes as face and array index apart;
#  * subgroupBroadcastFirst / subgroupAllEqual on a bool (MSL's simd_* take no bool, and Metal
#    has no all-equal vote):
#
#   layout(local_size_x = 32) in;
#   layout(set = 0, binding = 0, r32ui) uniform uimageCubeArray img;
#   layout(set = 0, binding = 1, std430) buffer Out { uint data[]; } o;
#   void main() {
#     uint x = gl_GlobalInvocationID.x;
#     bool b = x == 0u;
#     o.data[0] = imageLoad(img, ivec3(1, 2, 7)).x;  // layer 1, face 1
#     o.data[1] = subgroupBroadcastFirst(b) ? 1u : 0u;
#     o.data[2] = subgroupAllEqual(b) ? 1u : 0u;
#   }
#
# Usage: gen_subgroup_cube.py <out.spv>
import struct
import sys

OP = dict(Name=5, MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeBool=20,
          TypeInt=21, TypeVector=23, TypeImage=25, TypeRuntimeArray=29, TypeStruct=30, TypePointer=32,
          TypeFunction=33, Constant=43, ConstantComposite=44, Function=54, FunctionEnd=56, Variable=59, Load=61,
          Store=62, AccessChain=65, Decorate=71, MemberDecorate=72, CompositeExtract=81, ImageRead=98,
          Select=169, IEqual=170, Label=248, Return=253, GroupNonUniformAllEqual=336,
          GroupNonUniformBroadcastFirst=338)

words = []


def ins(op, *operands):
    ops = []
    for o in operands:
        if isinstance(o, str):
            b = o.encode() + b"\0"
            b += b"\0" * (-len(b) % 4)
            ops += list(struct.unpack("<%dI" % (len(b) // 4), b))
        else:
            ops.append(o)
    words.append(((len(ops) + 1) << 16) | OP[op])
    words.extend(ops)


ids = {}


def i(name):
    if name not in ids:
        ids[name] = len(ids) + 1
    return ids[name]


for cap in (1, 34, 61, 62, 64):  # Shader, ImageCubeArray, GroupNonUniform, Vote, Ballot
    ins("Capability", cap)
ins("MemoryModel", 0, 1)
ins("EntryPoint", 5, i("main"), "main", i("gid"))
ins("ExecutionMode", i("main"), 17, 32, 1, 1)  # LocalSize 32 1 1
ins("Decorate", i("gid"), 11, 28)    # BuiltIn GlobalInvocationId
ins("Decorate", i("img"), 34, 0)     # DescriptorSet 0
ins("Decorate", i("img"), 33, 0)     # Binding 0
ins("Decorate", i("rta"), 6, 4)      # ArrayStride 4
ins("MemberDecorate", i("Out"), 0, 35, 0)
ins("Decorate", i("Out"), 2)         # Block
ins("Decorate", i("o"), 34, 0)
ins("Decorate", i("o"), 33, 1)
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeBool", i("bool"))
ins("TypeInt", i("uint"), 32, 0)
ins("TypeInt", i("int"), 32, 1)
ins("TypeVector", i("v3uint"), i("uint"), 3)
ins("TypeVector", i("v4uint"), i("uint"), 4)
ins("TypeVector", i("v3int"), i("int"), 3)
ins("TypeImage", i("imgty"), i("uint"), 3, 0, 1, 0, 2, 33)  # Cube, arrayed, storage, R32ui
ins("TypePointer", i("p_uc_img"), 0, i("imgty"))
ins("TypeRuntimeArray", i("rta"), i("uint"))
ins("TypeStruct", i("Out"), i("rta"))
ins("TypePointer", i("p_sb_Out"), 12, i("Out"))
ins("TypePointer", i("p_sb_uint"), 12, i("uint"))
ins("TypePointer", i("p_in_v3"), 1, i("v3uint"))
ins("Constant", i("uint"), i("u0"), 0)
ins("Constant", i("uint"), i("u1"), 1)
ins("Constant", i("uint"), i("u2"), 2)
ins("Constant", i("uint"), i("scope"), 3)  # Subgroup
ins("Constant", i("int"), i("i0"), 0)
ins("Constant", i("int"), i("i1"), 1)
ins("Constant", i("int"), i("i2"), 2)
ins("Constant", i("int"), i("i7"), 7)
ins("ConstantComposite", i("v3int"), i("coord"), i("i1"), i("i2"), i("i7"))
ins("Variable", i("p_in_v3"), i("gid"), 1)
ins("Variable", i("p_uc_img"), i("img"), 0)
ins("Variable", i("p_sb_Out"), i("o"), 12)
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("v3uint"), i("g"), i("gid"))
ins("CompositeExtract", i("uint"), i("gx"), i("g"), 0)
ins("IEqual", i("bool"), i("b"), i("gx"), i("u0"))
ins("Load", i("imgty"), i("im"), i("img"))
ins("ImageRead", i("v4uint"), i("texel"), i("im"), i("coord"))
ins("CompositeExtract", i("uint"), i("tx"), i("texel"), 0)
ins("AccessChain", i("p_sb_uint"), i("d0"), i("o"), i("i0"), i("i0"))
ins("Store", i("d0"), i("tx"))
ins("GroupNonUniformBroadcastFirst", i("bool"), i("bf"), i("scope"), i("b"))
ins("Select", i("uint"), i("bfu"), i("bf"), i("u1"), i("u0"))
ins("AccessChain", i("p_sb_uint"), i("d1"), i("o"), i("i0"), i("i1"))
ins("Store", i("d1"), i("bfu"))
ins("GroupNonUniformAllEqual", i("bool"), i("ae"), i("scope"), i("b"))
ins("Select", i("uint"), i("aeu"), i("ae"), i("u1"), i("u0"))
ins("AccessChain", i("p_sb_uint"), i("d2"), i("o"), i("i0"), i("i2"))
ins("Store", i("d2"), i("aeu"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
