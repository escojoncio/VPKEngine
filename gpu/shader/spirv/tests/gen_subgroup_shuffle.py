#!/usr/bin/env python3
# Writes subgroup_shuffle.comp.spv: a compute shader hand-assembled (no glslang needed) covering the relative
# shuffles (Xor / Up / Down, a shuffle of the computed lane) and the quad broadcast:
#
#   layout(local_size_x = 32) in;
#   layout(set = 0, binding = 0, std430) buffer Out { uint data[]; } o;
#   void main() {
#     uint x = gl_GlobalInvocationID.x;
#     o.data[0] = subgroupShuffleXor(x, 1u);
#     o.data[1] = subgroupShuffleUp(x, 1u);
#     o.data[2] = subgroupShuffleDown(x, 1u);
#     o.data[3] = subgroupQuadBroadcast(x, 2u);
#   }
#
# Usage: gen_subgroup_shuffle.py <out.spv>
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeInt=21, TypeVector=23,
          TypeRuntimeArray=29, TypeStruct=30, TypePointer=32, TypeFunction=33, Constant=43, Function=54,
          FunctionEnd=56, Variable=59, Load=61, Store=62, AccessChain=65, Decorate=71, MemberDecorate=72,
          CompositeExtract=81, Label=248, Return=253, ShuffleXor=346, ShuffleUp=347, ShuffleDown=348,
          QuadBroadcast=365)

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


for cap in (1, 61, 65, 66, 68):  # Shader, GroupNonUniform, Shuffle, ShuffleRelative, Quad
    ins("Capability", cap)
ins("MemoryModel", 0, 1)
ins("EntryPoint", 5, i("main"), "main", i("gid"))
ins("ExecutionMode", i("main"), 17, 32, 1, 1)  # LocalSize 32 1 1
ins("Decorate", i("gid"), 11, 28)    # BuiltIn GlobalInvocationId
ins("Decorate", i("rta"), 6, 4)      # ArrayStride 4
ins("MemberDecorate", i("Out"), 0, 35, 0)
ins("Decorate", i("Out"), 2)         # Block
ins("Decorate", i("o"), 34, 0)
ins("Decorate", i("o"), 33, 0)
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeInt", i("uint"), 32, 0)
ins("TypeInt", i("int"), 32, 1)
ins("TypeVector", i("v3uint"), i("uint"), 3)
ins("TypeRuntimeArray", i("rta"), i("uint"))
ins("TypeStruct", i("Out"), i("rta"))
ins("TypePointer", i("p_sb_Out"), 12, i("Out"))
ins("TypePointer", i("p_sb_uint"), 12, i("uint"))
ins("TypePointer", i("p_in_v3"), 1, i("v3uint"))
for n in range(4):
    ins("Constant", i("int"), i("i%d" % n), n)
ins("Constant", i("uint"), i("u1"), 1)
ins("Constant", i("uint"), i("u2"), 2)
ins("Constant", i("uint"), i("scope"), 3)  # Subgroup
ins("Variable", i("p_in_v3"), i("gid"), 1)
ins("Variable", i("p_sb_Out"), i("o"), 12)
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("v3uint"), i("g"), i("gid"))
ins("CompositeExtract", i("uint"), i("x"), i("g"), 0)
for n, (op, delta) in enumerate((("ShuffleXor", "u1"), ("ShuffleUp", "u1"), ("ShuffleDown", "u1"), ("QuadBroadcast", "u2"))):
    ins(op, i("uint"), i("r%d" % n), i("scope"), i("x"), i(delta))
    ins("AccessChain", i("p_sb_uint"), i("d%d" % n), i("o"), i("i0"), i("i%d" % n))
    ins("Store", i("d%d" % n), i("r%d" % n))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
