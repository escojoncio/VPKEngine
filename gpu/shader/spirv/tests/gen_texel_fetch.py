#!/usr/bin/env python3
# Writes texel_fetch.comp.spv: a compute shader hand-assembled (no glslang needed) that reads a
# uniform texel buffer of Unknown format (its element format comes from the bound VkBufferView,
# the case of the Vulkan CTS api.buffer_view.access tests):
#
#   layout(local_size_x = 1) in;
#   layout(set = 0, binding = 0, std430) buffer Out { uvec4 v[]; } o;
#   layout(set = 0, binding = 1) uniform utextureBuffer t;
#   void main() { o.v[gl_GlobalInvocationID.x] = texelFetch(t, int(gl_GlobalInvocationID.x)); }
#
# Usage: gen_texel_fetch.py <out.spv>
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeInt=21,
          TypeVector=23, TypeImage=25, TypeRuntimeArray=29, TypeStruct=30, TypePointer=32, TypeFunction=33,
          Constant=43, Function=54, FunctionEnd=56, Variable=59, Load=61, Store=62, AccessChain=65,
          Decorate=71, MemberDecorate=72, CompositeExtract=81, ImageFetch=95, Bitcast=124, Label=248,
          Return=253)

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


ins("Capability", 1)    # Shader
ins("Capability", 46)   # SampledBuffer
ins("MemoryModel", 0, 1)
ins("EntryPoint", 5, i("main"), "main", i("gid"))
ins("ExecutionMode", i("main"), 17, 1, 1, 1)  # LocalSize 1 1 1
ins("Decorate", i("gid"), 11, 28)    # BuiltIn GlobalInvocationId
ins("Decorate", i("rta"), 6, 16)     # ArrayStride 16
ins("MemberDecorate", i("Out"), 0, 35, 0)  # Offset 0
ins("Decorate", i("Out"), 2)         # Block
ins("Decorate", i("o"), 34, 0)       # DescriptorSet 0
ins("Decorate", i("o"), 33, 0)       # Binding 0
ins("Decorate", i("t"), 34, 0)
ins("Decorate", i("t"), 33, 1)
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeInt", i("uint"), 32, 0)
ins("TypeInt", i("int"), 32, 1)
ins("TypeVector", i("v3uint"), i("uint"), 3)
ins("TypeVector", i("v4uint"), i("uint"), 4)
ins("Constant", i("int"), i("i0"), 0)
# OpTypeImage: sampled type uint, Dim Buffer (5), depth 0, arrayed 0, ms 0, sampled 1, format Unknown
ins("TypeImage", i("img"), i("uint"), 5, 0, 0, 0, 1, 0)
ins("TypeRuntimeArray", i("rta"), i("v4uint"))
ins("TypeStruct", i("Out"), i("rta"))
ins("TypePointer", i("p_sb_Out"), 12, i("Out"))
ins("TypePointer", i("p_sb_v4"), 12, i("v4uint"))
ins("TypePointer", i("p_uc_img"), 0, i("img"))
ins("TypePointer", i("p_in_v3"), 1, i("v3uint"))
ins("Variable", i("p_in_v3"), i("gid"), 1)
ins("Variable", i("p_sb_Out"), i("o"), 12)
ins("Variable", i("p_uc_img"), i("t"), 0)
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("v3uint"), i("g"), i("gid"))
ins("CompositeExtract", i("uint"), i("gx"), i("g"), 0)
ins("Bitcast", i("int"), i("ix"), i("gx"))
ins("Load", i("img"), i("tv"), i("t"))
ins("ImageFetch", i("v4uint"), i("texel"), i("tv"), i("ix"))
ins("AccessChain", i("p_sb_v4"), i("dp"), i("o"), i("i0"), i("gx"))
ins("Store", i("dp"), i("texel"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
