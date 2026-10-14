#!/usr/bin/env python3
# Writes tex1d.frag.spv: 1D images sampled, fetched, queried and projected the way glslang emits
# them (the Vulkan CTS glsl.texture_functions 1D cases). Hand-assembled (no glslang needed):
#
#   layout(set = 0, binding = 0) uniform sampler1DArray a;
#   layout(set = 0, binding = 1) uniform sampler1D s;
#   layout(location = 0) out vec4 o;
#   void main() {
#     ivec2 sz = textureSize(a, 0);
#     o = textureOffset(a, vec2(0.25, 1.0), 1) + texelFetch(a, ivec2(3, 1), 0) +
#         textureProj(s, vec2(0.5, 2.0)) + vec4(float(sz.x), float(sz.y), 0.0, 0.0);
#   }
#
# Usage: gen_tex1d.py <out.spv>
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeInt=21,
          TypeFloat=22, TypeVector=23, TypeImage=25, TypeSampledImage=27, TypePointer=32, TypeFunction=33,
          Constant=43, ConstantComposite=44, Function=54, FunctionEnd=56, Variable=59, Load=61, Store=62,
          Decorate=71, CompositeExtract=81, CompositeConstruct=80, ImageSampleImplicitLod=87,
          ImageSampleProjImplicitLod=91, ImageFetch=95, Image=100, ImageQuerySizeLod=103, ConvertSToF=111,
          FAdd=129, Label=248, Return=253)

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
ins("Capability", 43)    # Sampled1D
ins("Capability", 50)    # ImageQuery
ins("MemoryModel", 0, 1)
ins("EntryPoint", 4, i("main"), "main", i("o"))
ins("ExecutionMode", i("main"), 7)  # OriginUpperLeft
ins("Decorate", i("a"), 34, 0)
ins("Decorate", i("a"), 33, 0)
ins("Decorate", i("s"), 34, 0)
ins("Decorate", i("s"), 33, 1)
ins("Decorate", i("o"), 30, 0)      # Location 0
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeFloat", i("float"), 32)
ins("TypeInt", i("int"), 32, 1)
ins("TypeVector", i("v2f"), i("float"), 2)
ins("TypeVector", i("v4f"), i("float"), 4)
ins("TypeVector", i("v2i"), i("int"), 2)
ins("TypeImage", i("img1da"), i("float"), 0, 0, 1, 0, 1, 0)  # 1D, arrayed, sampled, Unknown
ins("TypeImage", i("img1d"), i("float"), 0, 0, 0, 0, 1, 0)
ins("TypeSampledImage", i("si1da"), i("img1da"))
ins("TypeSampledImage", i("si1d"), i("img1d"))
ins("TypePointer", i("p_si1da"), 0, i("si1da"))
ins("TypePointer", i("p_si1d"), 0, i("si1d"))
ins("TypePointer", i("p_out_v4"), 3, i("v4f"))
ins("Variable", i("p_si1da"), i("a"), 0)
ins("Variable", i("p_si1d"), i("s"), 0)
ins("Variable", i("p_out_v4"), i("o"), 3)
ins("Constant", i("int"), i("i0"), 0)
ins("Constant", i("int"), i("i1"), 1)
ins("Constant", i("int"), i("i3"), 3)
ins("Constant", i("float"), i("f0"), 0)
ins("Constant", i("float"), i("fq"), 0x3E800000)   # 0.25
ins("Constant", i("float"), i("fh"), 0x3F000000)   # 0.5
ins("Constant", i("float"), i("f1"), 0x3F800000)   # 1.0
ins("Constant", i("float"), i("f2"), 0x40000000)   # 2.0
ins("ConstantComposite", i("v2f"), i("ca"), i("fq"), i("f1"))
ins("ConstantComposite", i("v2f"), i("cp"), i("fh"), i("f2"))
ins("ConstantComposite", i("v2i"), i("cf"), i("i3"), i("i1"))
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("si1da"), i("la"), i("a"))
ins("Load", i("si1d"), i("ls"), i("s"))
ins("Image", i("img1da"), i("ia"), i("la"))
ins("ImageQuerySizeLod", i("v2i"), i("sz"), i("ia"), i("i0"))
ins("ImageSampleImplicitLod", i("v4f"), i("t0"), i("la"), i("ca"), 8, i("i1"))  # ConstOffset 1
ins("ImageFetch", i("v4f"), i("t1"), i("ia"), i("cf"), 2, i("i0"))              # Lod 0
ins("ImageSampleProjImplicitLod", i("v4f"), i("t2"), i("ls"), i("cp"))
ins("CompositeExtract", i("int"), i("sx"), i("sz"), 0)
ins("CompositeExtract", i("int"), i("sy"), i("sz"), 1)
ins("ConvertSToF", i("float"), i("fx"), i("sx"))
ins("ConvertSToF", i("float"), i("fy"), i("sy"))
ins("CompositeConstruct", i("v4f"), i("t3"), i("fx"), i("fy"), i("f0"), i("f0"))
ins("FAdd", i("v4f"), i("s0"), i("t0"), i("t1"))
ins("FAdd", i("v4f"), i("s1"), i("s0"), i("t2"))
ins("FAdd", i("v4f"), i("s2"), i("s1"), i("t3"))
ins("Store", i("o"), i("s2"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
