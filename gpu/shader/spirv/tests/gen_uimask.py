#!/usr/bin/env python3
# Writes uimask.vert.spv and uimask.frag.spv: the smallest draw pipeline for the VPKTAA UI-mask
# tests (gpu/vulkan/tests), hand-assembled (no glslang needed):
#
#   uimask.vert: void main() { gl_Position = vec4(0.0, 0.0, 0.5, 1.0); }
#   uimask.frag: layout(location = 0) out vec4 o;  void main() { o = vec4(0.1, 0.2, 0.3, 0.5); }
#
# Usage: gen_uimask.py <out_dir>
import os
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeFloat=22,
          TypeVector=23, TypePointer=32, TypeFunction=33, Constant=43, ConstantComposite=44, Function=54,
          FunctionEnd=56, Variable=59, Store=62, Decorate=71, Label=248, Return=253)


class Module:
    def __init__(self):
        self.words = []
        self.ids = {}

    def i(self, name):
        if name not in self.ids:
            self.ids[name] = len(self.ids) + 1
        return self.ids[name]

    def ins(self, op, *operands):
        ops = []
        for o in operands:
            if isinstance(o, str):
                b = o.encode() + b"\0"
                b += b"\0" * (-len(b) % 4)
                ops += list(struct.unpack("<%dI" % (len(b) // 4), b))
            else:
                ops.append(o)
        self.words.append(((len(ops) + 1) << 16) | OP[op])
        self.words.extend(ops)

    def write(self, path):
        header = [0x07230203, 0x00010300, 0, len(self.ids) + 1, 0]
        with open(path, "wb") as f:
            f.write(struct.pack("<%dI" % (len(header) + len(self.words)), *(header + self.words)))


def build(stage, consts):
    m = Module()
    i, ins = m.i, m.ins
    ins("Capability", 1)  # Shader
    ins("MemoryModel", 0, 1)
    ins("EntryPoint", 0 if stage == "vert" else 4, i("main"), "main", i("out"))
    if stage == "frag":
        ins("ExecutionMode", i("main"), 7)  # OriginUpperLeft
        ins("Decorate", i("out"), 30, 0)    # Location 0
    else:
        ins("Decorate", i("out"), 11, 0)    # BuiltIn Position
    ins("TypeVoid", i("void"))
    ins("TypeFunction", i("fnty"), i("void"))
    ins("TypeFloat", i("float"), 32)
    ins("TypeVector", i("v4f"), i("float"), 4)
    ins("TypePointer", i("p_out"), 3, i("v4f"))
    ins("Variable", i("p_out"), i("out"), 3)
    for n, v in enumerate(consts):
        ins("Constant", i("float"), i("c%d" % n), struct.unpack("<I", struct.pack("<f", v))[0])
    ins("ConstantComposite", i("v4f"), i("val"), *[i("c%d" % n) for n in range(4)])
    ins("Function", i("void"), i("main"), 0, i("fnty"))
    ins("Label", i("entry"))
    ins("Store", i("out"), i("val"))
    ins("Return")
    ins("FunctionEnd")
    return m


out = sys.argv[1]
build("vert", [0.0, 0.0, 0.5, 1.0]).write(os.path.join(out, "uimask.vert.spv"))
build("frag", [0.1, 0.2, 0.3, 0.5]).write(os.path.join(out, "uimask.frag.spv"))
