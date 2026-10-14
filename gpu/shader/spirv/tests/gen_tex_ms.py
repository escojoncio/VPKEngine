#!/usr/bin/env python3
# Writes tex_ms.frag.spv: a multisampled image fetched per sample and its sample count queried
# (CTS glsl.texture_functions.query.texturesamples, custom MSAA resolves). Hand-assembled:
#
#   layout(set = 0, binding = 0) uniform isampler2DMS m;
#   layout(location = 0) out ivec4 o;
#   void main() { o = texelFetch(m, ivec2(2, 3), 1) + ivec4(textureSamples(m)); }
#
# Usage: gen_tex_ms.py <out.spv>
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeInt=21,
          TypeFloat=22, TypeVector=23, TypeImage=25, TypeSampledImage=27, TypePointer=32, TypeFunction=33,
          Constant=43, ConstantComposite=44, Function=54, FunctionEnd=56, Variable=59, Load=61, Store=62,
          Decorate=71, CompositeExtract=81, CompositeConstruct=80, ImageSampleImplicitLod=87,
          ImageSampleProjImplicitLod=91, ImageFetch=95, Image=100, ImageQuerySizeLod=103, ImageQuerySamples=107, ConvertSToF=111,
          FAdd=129, IAdd=128, Label=248, Return=253)

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
ins("Capability", 50)    # ImageQuery
ins("MemoryModel", 0, 1)
ins("EntryPoint", 4, i("main"), "main", i("o"))
ins("ExecutionMode", i("main"), 7)  # OriginUpperLeft
ins("Decorate", i("m"), 34, 0)
ins("Decorate", i("m"), 33, 0)
ins("Decorate", i("o"), 30, 0)
ins("Decorate", i("o"), 14)         # Flat
ins("TypeVoid", i("void"))
ins("TypeFunction", i("fnty"), i("void"))
ins("TypeInt", i("int"), 32, 1)
ins("TypeVector", i("v2i"), i("int"), 2)
ins("TypeVector", i("v4i"), i("int"), 4)
ins("TypeImage", i("imgms"), i("int"), 1, 0, 0, 1, 1, 0)  # 2D, multisampled, sampled
ins("TypeSampledImage", i("sims"), i("imgms"))
ins("TypePointer", i("p_sims"), 0, i("sims"))
ins("TypePointer", i("p_out_v4i"), 3, i("v4i"))
ins("Variable", i("p_sims"), i("m"), 0)
ins("Variable", i("p_out_v4i"), i("o"), 3)
ins("Constant", i("int"), i("i1"), 1)
ins("Constant", i("int"), i("i2"), 2)
ins("Constant", i("int"), i("i3"), 3)
ins("ConstantComposite", i("v2i"), i("cc"), i("i2"), i("i3"))
ins("Function", i("void"), i("main"), 0, i("fnty"))
ins("Label", i("entry"))
ins("Load", i("sims"), i("lm"), i("m"))
ins("Image", i("imgms"), i("im"), i("lm"))
ins("ImageFetch", i("v4i"), i("t0"), i("im"), i("cc"), 0x40, i("i1"))   # Sample 1
ins("ImageQuerySamples", i("int"), i("n"), i("im"))
ins("CompositeConstruct", i("v4i"), i("nv"), i("n"), i("n"), i("n"), i("n"))
ins("IAdd", i("v4i"), i("r"), i("t0"), i("nv"))
ins("Store", i("o"), i("r"))
ins("Return")
ins("FunctionEnd")

header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<%dI" % (len(header) + len(words)), *(header + words)))
