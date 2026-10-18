#!/usr/bin/env python3
# Writes storage16.comp.spv, hand-assembled (no glslang needed).
#
# storage16.comp (storageBuffer16BitAccess: 16-bit integers 2 bytes apart in a buffer):
#   layout(local_size_x = 1) in;
#   layout(set = 0, binding = 0, std430) buffer B {
#     int16_t s[4];      // offset 0, stride 2   (input: 1, -2, 3, -4)
#     u16vec2 v;         // offset 8             (input: 65535, 7)
#     int r[4];          // offset 12
#     int16_t o[2];      // offset 28, stride 2
#     uint tail;         // offset 32 (must survive the half-word stores)
#   } b;
#   void main() {
#     r[0] = int(s[1]); r[1] = int(s[3]);           // -2, -4 (sign-extended from the high half)
#     r[2] = int(uint(v.x)); r[3] = int(uint(v.y));  // 65535, 7
#     o[1] = s[2] + int16_t(10);                      // 13 in the high half of word 7
#   }
# Usage: gen_storage16.py <dir>
import os
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeInt=21,
          TypeVector=23, TypeArray=28, TypeStruct=30, TypePointer=32, TypeFunction=33, Constant=43,
          Function=54, FunctionEnd=56, Variable=59, Load=61, Store=62, AccessChain=65, Decorate=71,
          MemberDecorate=72, CompositeExtract=81, UConvert=113, SConvert=114, Bitcast=124, IAdd=128,
          Label=248, Return=253)


class Asm:
    def __init__(self):
        self.words, self.ids = [], {}

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


def storage16_comp(path):
    a = Asm()
    i, ins = a.i, a.ins
    ins("Capability", 1)     # Shader
    ins("Capability", 22)    # Int16
    ins("Capability", 4433)  # StorageBuffer16BitAccess
    ins("MemoryModel", 0, 1)
    ins("EntryPoint", 5, i("main"), "main")
    ins("ExecutionMode", i("main"), 17, 1, 1, 1)
    ins("Decorate", i("arr_s"), 6, 2)   # ArrayStride 2
    ins("Decorate", i("arr_o"), 6, 2)
    ins("Decorate", i("arr_r"), 6, 4)
    for m, off in enumerate((0, 8, 12, 28, 32)):
        ins("MemberDecorate", i("B"), m, 35, off)
    ins("Decorate", i("B"), 2)          # Block
    ins("Decorate", i("b"), 34, 0)
    ins("Decorate", i("b"), 33, 0)
    ins("TypeVoid", i("void"))
    ins("TypeFunction", i("fnty"), i("void"))
    ins("TypeInt", i("int"), 32, 1)
    ins("TypeInt", i("uint"), 32, 0)
    ins("TypeInt", i("short"), 16, 1)
    ins("TypeInt", i("ushort"), 16, 0)
    ins("TypeVector", i("v2us"), i("ushort"), 2)
    ins("Constant", i("uint"), i("u4"), 4)
    ins("Constant", i("uint"), i("u2"), 2)
    ins("TypeArray", i("arr_s"), i("short"), i("u4"))
    ins("TypeArray", i("arr_r"), i("int"), i("u4"))
    ins("TypeArray", i("arr_o"), i("short"), i("u2"))
    ins("TypeStruct", i("B"), i("arr_s"), i("v2us"), i("arr_r"), i("arr_o"), i("uint"))
    ins("TypePointer", i("p_B"), 12, i("B"))
    ins("TypePointer", i("p_short"), 12, i("short"))
    ins("TypePointer", i("p_v2us"), 12, i("v2us"))
    ins("TypePointer", i("p_int"), 12, i("int"))
    ins("Variable", i("p_B"), i("b"), 12)
    for k in range(5):
        ins("Constant", i("int"), i("k%d" % k), k)
    ins("Constant", i("short"), i("s10"), 10)
    ins("Function", i("void"), i("main"), 0, i("fnty"))
    ins("Label", i("entry"))

    def r(k, v):
        ins("AccessChain", i("p_int"), i("pr%d" % k), i("b"), i("k2"), i("k%d" % k))
        ins("Store", i("pr%d" % k), v)
    for k, idx in ((0, 1), (1, 3)):
        ins("AccessChain", i("p_short"), i("ps%d" % k), i("b"), i("k0"), i("k%d" % idx))
        ins("Load", i("short"), i("ls%d" % k), i("ps%d" % k))
        ins("SConvert", i("int"), i("ws%d" % k), i("ls%d" % k))
        r(k, i("ws%d" % k))
    ins("AccessChain", i("p_v2us"), i("pv"), i("b"), i("k1"))
    ins("Load", i("v2us"), i("lv"), i("pv"))
    for k, c in ((2, 0), (3, 1)):
        ins("CompositeExtract", i("ushort"), i("e%d" % k), i("lv"), c)
        ins("UConvert", i("uint"), i("eu%d" % k), i("e%d" % k))
        ins("Bitcast", i("int"), i("ei%d" % k), i("eu%d" % k))
        r(k, i("ei%d" % k))
    ins("AccessChain", i("p_short"), i("ps2"), i("b"), i("k0"), i("k2"))
    ins("Load", i("short"), i("l2"), i("ps2"))
    ins("IAdd", i("short"), i("sum"), i("l2"), i("s10"))
    ins("AccessChain", i("p_short"), i("po1"), i("b"), i("k3"), i("k1"))
    ins("Store", i("po1"), i("sum"))
    ins("Return")
    ins("FunctionEnd")
    a.write(path)


storage16_comp(os.path.join(sys.argv[1], "storage16.comp.spv"))
