#!/usr/bin/env python3
# Writes clip.vert.spv: a vertex shader writing gl_ClipDistance[1] through gl_PerVertex (the CTS
# clipping.user_defined cases). Hand-assembled:
#
#   out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[2]; };
#   void main() { gl_Position = vec4(0, 0, 0, 1); gl_ClipDistance[1] = 0.5; }
#
# Usage: gen_clip.py <out.spv>
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, Capability=17, TypeVoid=19, TypeInt=21, TypeFloat=22,
          TypeVector=23, TypeArray=28, TypeStruct=30, TypePointer=32, TypeFunction=33, Constant=43,
          ConstantComposite=44, Function=54, FunctionEnd=56, Variable=59, Store=62, AccessChain=65,
          Decorate=71, MemberDecorate=72, Label=248, Return=253)

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
ins("Capability", 32)    # ClipDistance
ins("MemoryModel", 0, 1)
ins("EntryPoint", 0, i("main"), "main", i("pv"))
ins("MemberDecorate", i("PerVertex"), 0, 11, 0)   # BuiltIn Position
ins("MemberDecorate", i("PerVertex"), 1, 11, 3)   # BuiltIn ClipDistance
ins("Decorate", i("PerVertex"), 2)                # Block
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeFloat", i("float"), 32)
ins("TypeInt", i("int"), 32, 1)
ins("TypeInt", i("uint"), 32, 0)
ins("TypeVector", i("v4f"), i("float"), 4)
ins("Constant", i("uint"), i("u2"), 2)
ins("TypeArray", i("arr2"), i("float"), i("u2"))
ins("TypeStruct", i("PerVertex"), i("v4f"), i("arr2"))
ins("TypePointer", i("p_out_pv"), 3, i("PerVertex"))
ins("TypePointer", i("p_out_v4"), 3, i("v4f"))
ins("TypePointer", i("p_out_f"), 3, i("float"))
ins("Variable", i("p_out_pv"), i("pv"), 3)
ins("Constant", i("int"), i("i0"), 0)
ins("Constant", i("int"), i("i1"), 1)
ins("Constant", i("float"), i("f0"), 0)
ins("Constant", i("float"), i("f1"), 0x3F800000)
ins("Constant", i("float"), i("fh"), 0x3F000000)
ins("ConstantComposite", i("v4f"), i("pos"), i("f0"), i("f0"), i("f0"), i("f1"))
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("AccessChain", i("p_out_v4"), i("pp"), i("pv"), i("i0"))
ins("Store", i("pp"), i("pos"))
ins("AccessChain", i("p_out_f"), i("cp"), i("pv"), i("i1"), i("i1"))
ins("Store", i("cp"), i("fh"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
