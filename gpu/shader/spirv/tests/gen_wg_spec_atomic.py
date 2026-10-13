#!/usr/bin/env python3
# Writes wg_spec_atomic.comp.spv: a compute shader hand-assembled (no glslang needed) the way
# glslang/DXC emit this pattern, roughly
#
#   layout(local_size_x_id = 0) in;                         // WorkgroupSize spec composite
#   layout(set = 0, binding = 0, r32ui) uniform uimageBuffer counters;
#   layout(set = 0, binding = 1, std430) buffer Out { uint data[]; } outb;
#   shared uint tile[gl_WorkGroupSize.x];                   // length = OpSpecConstantOp
#   void main() {
#       uint i = gl_LocalInvocationIndex;
#       tile[i] = imageAtomicAdd(counters, int(gl_WorkGroupID.x), 1u);   // texel buffer atomic
#       barrier();
#       outb.data[gl_GlobalInvocationID.x] = tile[(i + 1u) % gl_WorkGroupSize.x];
#   }
#
# Usage: gen_wg_spec_atomic.py <out.spv>
import struct
import sys

OP = dict(Name=5, MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19,
          TypeInt=21, TypeVector=23, TypeImage=25, TypeArray=28, TypeRuntimeArray=29, TypeStruct=30,
          TypePointer=32, TypeFunction=33, Constant=43, SpecConstant=50, SpecConstantComposite=51,
          SpecConstantOp=52, Function=54, FunctionEnd=56, Variable=59, ImageTexelPointer=60, Load=61,
          Store=62, AccessChain=65, Decorate=71, MemberDecorate=72, CompositeExtract=81, Bitcast=124,
          IAdd=128, UMod=137, ControlBarrier=224, AtomicIAdd=234, Label=248, Return=253)

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


body = []
ins("Capability", 1)    # Shader
ins("Capability", 47)   # ImageBuffer
ins("MemoryModel", 0, 1)
ins("EntryPoint", 5, i("main"), "main", i("lii"), i("wgid"), i("gid"))
ins("ExecutionMode", i("main"), 17, 1, 1, 1)  # LocalSize 1 1 1 (overridden by the spec composite)
ins("Name", i("tile"), "tile")
ins("Decorate", i("lii"), 11, 29)    # BuiltIn LocalInvocationIndex
ins("Decorate", i("wgid"), 11, 26)   # BuiltIn WorkgroupId
ins("Decorate", i("gid"), 11, 28)    # BuiltIn GlobalInvocationId
ins("Decorate", i("wsx"), 1, 0)      # SpecId 0
ins("Decorate", i("wgsize"), 11, 25) # BuiltIn WorkgroupSize
ins("Decorate", i("counters"), 34, 0)
ins("Decorate", i("counters"), 33, 0)
ins("Decorate", i("rta"), 6, 4)      # ArrayStride 4
ins("MemberDecorate", i("Out"), 0, 35, 0)  # Offset 0
ins("Decorate", i("Out"), 2)         # Block
ins("Decorate", i("outb"), 34, 0)
ins("Decorate", i("outb"), 33, 1)
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeInt", i("uint"), 32, 0)
ins("TypeInt", i("int"), 32, 1)
ins("TypeVector", i("v3uint"), i("uint"), 3)
ins("Constant", i("uint"), i("c0"), 0)
ins("Constant", i("uint"), i("c1"), 1)
ins("Constant", i("uint"), i("c2"), 2)
ins("Constant", i("uint"), i("c264"), 264)
ins("Constant", i("int"), i("i0"), 0)
ins("SpecConstant", i("uint"), i("wsx"), 64)
ins("SpecConstantComposite", i("v3uint"), i("wgsize"), i("wsx"), i("c1"), i("c1"))
ins("SpecConstantOp", i("uint"), i("tilelen"), OP["CompositeExtract"], i("wgsize"), 0)
ins("TypePointer", i("p_in_uint"), 1, i("uint"))
ins("TypePointer", i("p_in_v3"), 1, i("v3uint"))
ins("TypeImage", i("img"), i("uint"), 5, 0, 0, 0, 2, 33)  # Buffer, sampled 2, R32ui
ins("TypePointer", i("p_uc_img"), 0, i("img"))
ins("TypePointer", i("p_image_uint"), 11, i("uint"))
ins("TypeArray", i("tilety"), i("uint"), i("tilelen"))
ins("TypePointer", i("p_wg_tile"), 4, i("tilety"))
ins("TypePointer", i("p_wg_uint"), 4, i("uint"))
ins("TypeRuntimeArray", i("rta"), i("uint"))
ins("TypeStruct", i("Out"), i("rta"))
ins("TypePointer", i("p_sb_Out"), 12, i("Out"))
ins("TypePointer", i("p_sb_uint"), 12, i("uint"))
ins("Variable", i("p_in_uint"), i("lii"), 1)
ins("Variable", i("p_in_v3"), i("wgid"), 1)
ins("Variable", i("p_in_v3"), i("gid"), 1)
ins("Variable", i("p_uc_img"), i("counters"), 0)
ins("Variable", i("p_wg_tile"), i("tile"), 4)
ins("Variable", i("p_sb_Out"), i("outb"), 12)
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("uint"), i("idx"), i("lii"))
ins("Load", i("v3uint"), i("wg"), i("wgid"))
ins("CompositeExtract", i("uint"), i("wgx"), i("wg"), 0)
ins("Bitcast", i("int"), i("wgi"), i("wgx"))
ins("ImageTexelPointer", i("p_image_uint"), i("tp"), i("counters"), i("wgi"), i("c0"))
ins("AtomicIAdd", i("uint"), i("old"), i("tp"), i("c1"), i("c0"), i("c1"))
ins("AccessChain", i("p_wg_uint"), i("tptr"), i("tile"), i("idx"))
ins("Store", i("tptr"), i("old"))
ins("ControlBarrier", i("c2"), i("c2"), i("c264"))
ins("Load", i("v3uint"), i("g"), i("gid"))
ins("CompositeExtract", i("uint"), i("gx"), i("g"), 0)
ins("IAdd", i("uint"), i("i1"), i("idx"), i("c1"))
ins("CompositeExtract", i("uint"), i("wsxv"), i("wgsize"), 0)
ins("UMod", i("uint"), i("m"), i("i1"), i("wsxv"))
ins("AccessChain", i("p_wg_uint"), i("sp"), i("tile"), i("m"))
ins("Load", i("uint"), i("v"), i("sp"))
ins("AccessChain", i("p_sb_uint"), i("op"), i("outb"), i("i0"), i("gx"))
ins("Store", i("op"), i("v"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
