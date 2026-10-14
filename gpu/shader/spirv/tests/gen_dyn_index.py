#!/usr/bin/env python3
# Writes dyn_index.frag.spv: a combined image sampler array indexed at run time (the CTS
# binding-model / glsl dynamic-indexing cases, and games selecting one of N textures).
# Hand-assembled (no glslang needed):
#
#   layout(set = 0, binding = 0) uniform sampler2D t[4];
#   layout(location = 0) flat in int idx;
#   layout(location = 0) out vec4 o;
#   void main() { o = texture(t[idx], vec2(0.5)); }
#
# Usage: gen_dyn_index.py <out.spv>
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeInt=21,
          TypeFloat=22, TypeVector=23, TypeImage=25, TypeSampledImage=27, TypeArray=28, TypePointer=32,
          TypeFunction=33, Constant=43, ConstantComposite=44, Function=54, FunctionEnd=56, Variable=59,
          Load=61, Store=62, AccessChain=65, Decorate=71, ImageSampleImplicitLod=87, Label=248, Return=253)

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
ins("Capability", 29)    # SampledImageArrayDynamicIndexing
ins("MemoryModel", 0, 1)
ins("EntryPoint", 4, i("main"), "main", i("idx"), i("o"))
ins("ExecutionMode", i("main"), 7)  # OriginUpperLeft
ins("Decorate", i("t"), 34, 0)      # DescriptorSet 0
ins("Decorate", i("t"), 33, 0)      # Binding 0
ins("Decorate", i("idx"), 30, 0)    # Location 0
ins("Decorate", i("idx"), 14)       # Flat
ins("Decorate", i("o"), 30, 0)
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeFloat", i("float"), 32)
ins("TypeInt", i("int"), 32, 1)
ins("TypeInt", i("uint"), 32, 0)
ins("TypeVector", i("v2f"), i("float"), 2)
ins("TypeVector", i("v4f"), i("float"), 4)
ins("TypeImage", i("img2d"), i("float"), 1, 0, 0, 0, 1, 0)  # 2D, sampled, Unknown
ins("TypeSampledImage", i("si2d"), i("img2d"))
ins("Constant", i("uint"), i("u4"), 4)
ins("TypeArray", i("arr"), i("si2d"), i("u4"))
ins("TypePointer", i("p_arr"), 0, i("arr"))
ins("TypePointer", i("p_si"), 0, i("si2d"))
ins("TypePointer", i("p_in_i"), 1, i("int"))
ins("TypePointer", i("p_out_v4"), 3, i("v4f"))
ins("Variable", i("p_arr"), i("t"), 0)
ins("Variable", i("p_in_i"), i("idx"), 1)
ins("Variable", i("p_out_v4"), i("o"), 3)
ins("Constant", i("float"), i("fh"), 0x3F000000)   # 0.5
ins("ConstantComposite", i("v2f"), i("c"), i("fh"), i("fh"))
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("int"), i("li"), i("idx"))
ins("AccessChain", i("p_si"), i("pt"), i("t"), i("li"))
ins("Load", i("si2d"), i("lt"), i("pt"))
ins("ImageSampleImplicitLod", i("v4f"), i("r"), i("lt"), i("c"))
ins("Store", i("o"), i("r"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
