#!/usr/bin/env python3
# Writes array_length.comp.spv: a compute shader hand-assembled (no glslang needed) using
# OpArrayLength the way glslang emits `.length()` on the runtime array that ends a storage block
# (the Vulkan CTS indirect-draw "data from compute" shaders do this):
#
#   layout(local_size_x = 1) in;
#   layout(set = 0, binding = 0, std430) buffer Data { uint header[4]; uint data[]; } b;
#   void main() { b.data[gl_GlobalInvocationID.x] = uint(b.data.length()); }
#
# Usage: gen_array_length.py <out.spv>
import struct
import sys

OP = dict(Name=5, MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19,
          TypeInt=21, TypeVector=23, TypeArray=28, TypeRuntimeArray=29, TypeStruct=30, TypePointer=32,
          TypeFunction=33, Constant=43, Function=54, FunctionEnd=56, Variable=59, Load=61, Store=62,
          AccessChain=65, ArrayLength=68, Decorate=71, MemberDecorate=72, CompositeExtract=81, Label=248,
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
ins("MemoryModel", 0, 1)
ins("EntryPoint", 5, i("main"), "main", i("gid"))
ins("ExecutionMode", i("main"), 17, 1, 1, 1)  # LocalSize 1 1 1
ins("Decorate", i("gid"), 11, 28)    # BuiltIn GlobalInvocationId
ins("Decorate", i("arr4"), 6, 4)     # ArrayStride 4
ins("Decorate", i("rta"), 6, 4)      # ArrayStride 4
ins("MemberDecorate", i("Data"), 0, 35, 0)   # header: Offset 0
ins("MemberDecorate", i("Data"), 1, 35, 16)  # data: Offset 16
ins("Decorate", i("Data"), 2)        # Block
ins("Decorate", i("b"), 34, 0)       # DescriptorSet 0
ins("Decorate", i("b"), 33, 0)       # Binding 0
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeInt", i("uint"), 32, 0)
ins("TypeInt", i("int"), 32, 1)
ins("TypeVector", i("v3uint"), i("uint"), 3)
ins("Constant", i("uint"), i("c4"), 4)
ins("Constant", i("int"), i("i1"), 1)
ins("TypeArray", i("arr4"), i("uint"), i("c4"))
ins("TypeRuntimeArray", i("rta"), i("uint"))
ins("TypeStruct", i("Data"), i("arr4"), i("rta"))
ins("TypePointer", i("p_sb_Data"), 12, i("Data"))
ins("TypePointer", i("p_sb_uint"), 12, i("uint"))
ins("TypePointer", i("p_in_v3"), 1, i("v3uint"))
ins("Variable", i("p_in_v3"), i("gid"), 1)
ins("Variable", i("p_sb_Data"), i("b"), 12)
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("v3uint"), i("g"), i("gid"))
ins("CompositeExtract", i("uint"), i("gx"), i("g"), 0)
ins("ArrayLength", i("uint"), i("n"), i("b"), 1)
ins("AccessChain", i("p_sb_uint"), i("dp"), i("b"), i("i1"), i("gx"))
ins("Store", i("dp"), i("n"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
