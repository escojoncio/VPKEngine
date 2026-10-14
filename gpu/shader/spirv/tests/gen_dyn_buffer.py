#!/usr/bin/env python3
# Writes dyn_buffer.comp.spv: a storage-buffer descriptor array indexed at run time.
# Hand-assembled (no glslang needed):
#
#   layout(local_size_x = 1) in;
#   layout(set = 0, binding = 0) buffer B { uint v; } b[2];
#   void main() { uint i = gl_GlobalInvocationID.x; b[i].v = i; }
#
# Usage: gen_dyn_buffer.py <out.spv>
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeInt=21,
          TypeVector=23, TypeArray=28, TypeStruct=30, TypePointer=32, TypeFunction=33, Constant=43,
          Function=54, FunctionEnd=56, Variable=59, Load=61, Store=62, AccessChain=65, Decorate=71,
          MemberDecorate=72, CompositeExtract=81, Label=248, Return=253)

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


ins("Capability", 1)     # Shader
ins("Capability", 31)    # StorageBufferArrayDynamicIndexing
ins("MemoryModel", 0, 1)
ins("EntryPoint", 5, i("main"), "main", i("gid"), i("b"))
ins("ExecutionMode", i("main"), 17, 1, 1, 1)  # LocalSize 1 1 1
ins("Decorate", i("gid"), 11, 28)             # BuiltIn GlobalInvocationId
ins("MemberDecorate", i("B"), 0, 35, 0)       # Offset 0
ins("Decorate", i("B"), 2)                    # Block
ins("Decorate", i("b"), 34, 0)
ins("Decorate", i("b"), 33, 0)
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeInt", i("uint"), 32, 0)
ins("TypeInt", i("int"), 32, 1)
ins("TypeVector", i("v3u"), i("uint"), 3)
ins("TypeStruct", i("B"), i("uint"))
ins("Constant", i("uint"), i("u2"), 2)
ins("TypeArray", i("arr"), i("B"), i("u2"))
ins("TypePointer", i("p_arr"), 12, i("arr"))
ins("TypePointer", i("p_u"), 12, i("uint"))
ins("TypePointer", i("p_in_v3"), 1, i("v3u"))
ins("Variable", i("p_arr"), i("b"), 12)
ins("Variable", i("p_in_v3"), i("gid"), 1)
ins("Constant", i("int"), i("i0"), 0)
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("v3u"), i("g"), i("gid"))
ins("CompositeExtract", i("uint"), i("x"), i("g"), 0)
ins("AccessChain", i("p_u"), i("pv"), i("b"), i("x"), i("i0"))
ins("Store", i("pv"), i("x"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
