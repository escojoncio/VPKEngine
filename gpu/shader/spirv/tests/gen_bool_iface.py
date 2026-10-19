#!/usr/bin/env python3
# Writes bool_iface.vert.spv / bool_iface.frag.spv: a bool varying (not legal in Vulkan SPIR-V, but
# some front ends emit it). Hand-assembled:
#
#   // vertex                                  // fragment
#   layout(location = 0) out bool ob;          layout(location = 0) flat in bool ib;
#   void main() { pos = vec4(0); ob = true; }  layout(location = 0) out vec4 o;
#                                              void main() { o = ib ? vec4(1) : vec4(0); }
#
# Usage: gen_bool_iface.py <out dir>
import os
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeBool=20,
          TypeFloat=22, TypeVector=23, TypePointer=32, TypeFunction=33, ConstantTrue=41,
          Constant=43, ConstantComposite=44, Function=54, FunctionEnd=56, Variable=59, Load=61,
          Store=62, Decorate=71, Select=169, Label=248, Return=253)


def build(stage):
    words = []
    ids = {}

    def i(name):
        if name not in ids:
            ids[name] = len(ids) + 1
        return ids[name]

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

    vert = stage == "vert"
    ins("Capability", 1)
    ins("MemoryModel", 0, 1)
    if vert:
        ins("EntryPoint", 0, i("main"), "main", i("ob"), i("pos"))
        ins("Decorate", i("ob"), 30, 0)       # Location 0
        ins("Decorate", i("pos"), 11, 0)      # BuiltIn Position
    else:
        ins("EntryPoint", 4, i("main"), "main", i("ib"), i("o"))
        ins("ExecutionMode", i("main"), 7)    # OriginUpperLeft
        ins("Decorate", i("ib"), 30, 0)
        ins("Decorate", i("ib"), 14)          # Flat
        ins("Decorate", i("o"), 30, 0)
    ins("TypeVoid", i("void"))
    ins("TypeFunction", i("fnty"), i("void"))
    ins("TypeBool", i("bool"))
    ins("TypeFloat", i("float"), 32)
    ins("TypeVector", i("v4f"), i("float"), 4)
    ins("TypePointer", i("p_out_v4"), 3, i("v4f"))
    if vert:
        ins("TypePointer", i("p_out_b"), 3, i("bool"))
        ins("Variable", i("p_out_b"), i("ob"), 3)
        ins("Variable", i("p_out_v4"), i("pos"), 3)
        ins("ConstantTrue", i("bool"), i("t"))
        ins("Constant", i("float"), i("f0"), 0)
        ins("ConstantComposite", i("v4f"), i("c0"), i("f0"), i("f0"), i("f0"), i("f0"))
        ins("Function", i("void"), i("main"), 0, i("fnty"))
        ins("Label", i("entry"))
        ins("Store", i("pos"), i("c0"))
        ins("Store", i("ob"), i("t"))
    else:
        ins("TypePointer", i("p_in_b"), 1, i("bool"))
        ins("Variable", i("p_in_b"), i("ib"), 1)
        ins("Variable", i("p_out_v4"), i("o"), 3)
        ins("Constant", i("float"), i("f0"), 0)
        ins("Constant", i("float"), i("f1"), 0x3F800000)
        ins("ConstantComposite", i("v4f"), i("c0"), i("f0"), i("f0"), i("f0"), i("f0"))
        ins("ConstantComposite", i("v4f"), i("c1"), i("f1"), i("f1"), i("f1"), i("f1"))
        ins("Function", i("void"), i("main"), 0, i("fnty"))
        ins("Label", i("entry"))
        ins("Load", i("bool"), i("lb"), i("ib"))
        ins("Select", i("v4f"), i("r"), i("lb"), i("c1"), i("c0"))
        ins("Store", i("o"), i("r"))
    ins("Return")
    ins("FunctionEnd")
    header = [0x07230203, 0x00010300, 0, len(ids) + 1, 0]
    return struct.pack("<%dI" % (len(header) + len(words)), *(header + words))


for st in ("vert", "frag"):
    with open(os.path.join(sys.argv[1], "bool_iface." + st + ".spv"), "wb") as f:
        f.write(build(st))
