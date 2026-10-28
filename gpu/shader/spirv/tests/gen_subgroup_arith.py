#!/usr/bin/env python3
# Writes subgroup_arith.comp.spv: a compute shader hand-assembled (no glslang needed) covering the arithmetic and
# clustered subgroup operations:
#
#   layout(local_size_x = 32) in;
#   layout(set = 0, binding = 0, std430) buffer Out { uint data[]; } o;
#   void main() {
#     uint x = gl_GlobalInvocationID.x;
#     float f = float(x) + 0.5;
#     o.data[0] = subgroupAdd(x);                 // OpGroupNonUniformIAdd Reduce
#     o.data[1] = floatBitsToUint(subgroupInclusiveAdd(f));  // FAdd InclusiveScan (Metal builtin)
#     o.data[2] = uint(subgroupExclusiveMin(int(x)));       // SMin ExclusiveScan (lane loop)
#     o.data[3] = subgroupClusteredMax(x, 4u);              // UMax ClusteredReduce
#     o.data[4] = subgroupXor(x);                           // BitwiseXor Reduce
#     o.data[5] = subgroupAnd(x > 3u) ? 1u : 0u;            // LogicalAnd Reduce (bool)
#   }
#
# Usage: gen_subgroup_arith.py <out.spv>
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeBool=20, TypeInt=21,
          TypeFloat=22, TypeVector=23, TypeRuntimeArray=29, TypeStruct=30, TypePointer=32, TypeFunction=33,
          Constant=43, Function=54, FunctionEnd=56, Variable=59, Load=61, Store=62, AccessChain=65, Decorate=71,
          MemberDecorate=72, Bitcast=124, ConvertUToF=112, FAdd=129, UGreaterThan=172, Select=169,
          CompositeExtract=81, Label=248, Return=253,
          GNUIAdd=349, GNUFAdd=350, GNUSMin=353, GNUUMax=357, GNUBitwiseXor=361, GNULogicalAnd=362)

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


for cap in (1, 61, 63, 67):  # Shader, GroupNonUniform, GroupNonUniformArithmetic, GroupNonUniformClustered
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
ins("TypeBool", i("bool"))
ins("TypeInt", i("uint"), 32, 0)
ins("TypeInt", i("int"), 32, 1)
ins("TypeFloat", i("float"), 32)
ins("TypeVector", i("v3uint"), i("uint"), 3)
ins("TypeRuntimeArray", i("rta"), i("uint"))
ins("TypeStruct", i("Out"), i("rta"))
ins("TypePointer", i("p_sb_Out"), 12, i("Out"))
ins("TypePointer", i("p_sb_uint"), 12, i("uint"))
ins("TypePointer", i("p_in_v3"), 1, i("v3uint"))
for n in range(6):
    ins("Constant", i("int"), i("i%d" % n), n)
ins("Constant", i("uint"), i("u0"), 0)
ins("Constant", i("uint"), i("u1"), 1)
ins("Constant", i("uint"), i("u3"), 3)
ins("Constant", i("uint"), i("u4"), 4)
ins("Constant", i("float"), i("half"), 0x3F000000)
ins("Constant", i("uint"), i("scope"), 3)  # Subgroup
ins("Variable", i("p_in_v3"), i("gid"), 1)
ins("Variable", i("p_sb_Out"), i("o"), 12)
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("v3uint"), i("g"), i("gid"))
ins("CompositeExtract", i("uint"), i("x"), i("g"), 0)
ins("ConvertUToF", i("float"), i("xf"), i("x"))
ins("FAdd", i("float"), i("f"), i("xf"), i("half"))
ins("Bitcast", i("int"), i("xi"), i("x"))
ins("UGreaterThan", i("bool"), i("gt"), i("x"), i("u3"))
results = []
ins("GNUIAdd", i("uint"), i("r0"), i("scope"), 0, i("x"))                    # Reduce
ins("GNUFAdd", i("float"), i("r1f"), i("scope"), 1, i("f"))                  # InclusiveScan
ins("Bitcast", i("uint"), i("r1"), i("r1f"))
ins("GNUSMin", i("int"), i("r2i"), i("scope"), 2, i("xi"))                   # ExclusiveScan
ins("Bitcast", i("uint"), i("r2"), i("r2i"))
ins("GNUUMax", i("uint"), i("r3"), i("scope"), 3, i("x"), i("u4"))           # ClusteredReduce, 4
ins("GNUBitwiseXor", i("uint"), i("r4"), i("scope"), 0, i("x"))              # Reduce
ins("GNULogicalAnd", i("bool"), i("r5b"), i("scope"), 0, i("gt"))            # Reduce (bool)
ins("Select", i("uint"), i("r5"), i("r5b"), i("u1"), i("u0"))
for n in range(6):
    ins("AccessChain", i("p_sb_uint"), i("d%d" % n), i("o"), i("i0"), i("i%d" % n))
    ins("Store", i("d%d" % n), i("r%d" % n))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
