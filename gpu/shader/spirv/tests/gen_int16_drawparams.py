#!/usr/bin/env python3
# Writes int16.comp.spv and draw_params.vert.spv, hand-assembled (no glslang needed).
#
# int16.comp (shaderInt16: 16-bit integers held in 32-bit registers, re-normalised after every op):
#   layout(local_size_x = 1) in;
#   layout(set = 0, binding = 0, std430) buffer Out { int r[8]; } o;
#   void main() {
#     int16_t a = int16_t(30000), b = int16_t(10000);
#     int16_t s = a + b;                 // wraps: -25536
#     uint16_t u = uint16_t(s);          // 40000
#     o.r[0] = int(s);                   // -25536 (sign-extended)
#     o.r[1] = int(uint(u));             // 40000 (zero-extended)
#     o.r[2] = int(s >> 4);              // arithmetic: -1596
#     o.r[3] = int(uint(u >> 4));        // logical on 16 bits: 2500
#     o.r[4] = int(u < uint16_t(50000)); // 1 (unsigned compare)
#     o.r[5] = int(s < int16_t(0));      // 1 (signed compare)
#     o.r[6] = int(int16_t(-3) * int16_t(21846)); // -65538 wrapped to 16 bits = -2
#     o.r[7] = int(uint16_t(65535) / uint16_t(2)); // 32767
#   }
#
# draw_params.vert (VK_KHR_shader_draw_parameters):
#   layout(location = 0) out flat uvec3 p;
#   void main() { p = uvec3(gl_BaseVertex, gl_BaseInstance, gl_DrawID); gl_Position = vec4(0); }
#
# vertex_store.vert (vertexPipelineStoresAndAtomics):
#   layout(set = 0, binding = 0, std430) buffer B { uint count; uint ids[]; } b;
#   void main() { b.ids[gl_VertexIndex] = gl_VertexIndex; atomicAdd(b.count, 1u); gl_Position = vec4(0); }
#
# Usage: gen_int16_drawparams.py <dir>
import os
import struct
import sys

OP = dict(MemoryModel=14, EntryPoint=15, ExecutionMode=16, Capability=17, Extension=10, TypeVoid=19,
          TypeBool=20, TypeInt=21, TypeFloat=22, TypeVector=23, TypeArray=28, TypeStruct=30, TypePointer=32,
          TypeFunction=33, Constant=43, ConstantComposite=44, Function=54, FunctionEnd=56, Variable=59,
          Load=61, Store=62, AccessChain=65, Decorate=71, MemberDecorate=72, CompositeConstruct=80,
          UConvert=113, AtomicIAdd=234, TypeRuntimeArray=29, SConvert=114, Bitcast=124, IAdd=128, IMul=132, UDiv=134, ShiftRightLogical=194,
          ShiftRightArithmetic=195, ULessThan=176, SLessThan=177, Select=169, Label=248, Return=253)


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


def int16_comp(path):
    a = Asm()
    i, ins = a.i, a.ins
    ins("Capability", 1)   # Shader
    ins("Capability", 22)  # Int16
    ins("MemoryModel", 0, 1)
    ins("EntryPoint", 5, i("main"), "main")
    ins("ExecutionMode", i("main"), 17, 1, 1, 1)
    ins("Decorate", i("arr"), 6, 4)               # ArrayStride 4
    ins("MemberDecorate", i("Out"), 0, 35, 0)     # Offset 0
    ins("Decorate", i("Out"), 2)                  # Block
    ins("Decorate", i("o"), 34, 0)
    ins("Decorate", i("o"), 33, 0)
    ins("TypeVoid", i("void"))
    ins("TypeFunction", i("fnty"), i("void"))
    ins("TypeBool", i("bool"))
    ins("TypeInt", i("int"), 32, 1)
    ins("TypeInt", i("uint"), 32, 0)
    ins("TypeInt", i("short"), 16, 1)
    ins("TypeInt", i("ushort"), 16, 0)
    ins("Constant", i("uint"), i("u8"), 8)
    ins("TypeArray", i("arr"), i("int"), i("u8"))
    ins("TypeStruct", i("Out"), i("arr"))
    ins("TypePointer", i("p_sb_Out"), 12, i("Out"))
    ins("TypePointer", i("p_sb_int"), 12, i("int"))
    ins("Variable", i("p_sb_Out"), i("o"), 12)
    ins("Constant", i("int"), i("i0"), 0)
    ins("Constant", i("int"), i("i1"), 1)
    for k in range(8):
        ins("Constant", i("int"), i("k%d" % k), k)
    ins("Constant", i("short"), i("s30000"), 30000)
    ins("Constant", i("short"), i("s10000"), 10000)
    ins("Constant", i("short"), i("s4"), 4)
    ins("Constant", i("short"), i("s0"), 0)
    ins("Constant", i("short"), i("sm3"), 0xFFFFFFFD)  # -3 (sign-extended literal)
    ins("Constant", i("short"), i("s21846"), 21846)
    ins("Constant", i("ushort"), i("u4"), 4)
    ins("Constant", i("ushort"), i("u50000"), 50000)
    ins("Constant", i("ushort"), i("u65535"), 65535)
    ins("Constant", i("ushort"), i("u2"), 2)
    ins("Function", i("void"), i("main"), 0, i("fnty"))
    ins("Label", i("entry"))
    ins("IAdd", i("short"), i("s"), i("s30000"), i("s10000"))
    ins("Bitcast", i("ushort"), i("u"), i("s"))

    def out(k, v):
        ins("AccessChain", i("p_sb_int"), i("p%d" % k), i("o"), i("i0"), i("k%d" % k))
        ins("Store", i("p%d" % k), v)
    ins("SConvert", i("int"), i("r0"), i("s"))
    out(0, i("r0"))
    ins("UConvert", i("uint"), i("r1u"), i("u"))
    ins("Bitcast", i("int"), i("r1"), i("r1u"))
    out(1, i("r1"))
    ins("ShiftRightArithmetic", i("short"), i("sh"), i("s"), i("s4"))
    ins("SConvert", i("int"), i("r2"), i("sh"))
    out(2, i("r2"))
    ins("ShiftRightLogical", i("ushort"), i("uh"), i("u"), i("u4"))
    ins("UConvert", i("uint"), i("r3u"), i("uh"))
    ins("Bitcast", i("int"), i("r3"), i("r3u"))
    out(3, i("r3"))
    ins("ULessThan", i("bool"), i("c4"), i("u"), i("u50000"))
    ins("Select", i("int"), i("r4"), i("c4"), i("i1"), i("i0"))
    out(4, i("r4"))
    ins("SLessThan", i("bool"), i("c5"), i("s"), i("s0"))
    ins("Select", i("int"), i("r5"), i("c5"), i("i1"), i("i0"))
    out(5, i("r5"))
    ins("IMul", i("short"), i("m"), i("sm3"), i("s21846"))
    ins("SConvert", i("int"), i("r6"), i("m"))
    out(6, i("r6"))
    ins("UDiv", i("ushort"), i("d"), i("u65535"), i("u2"))
    ins("UConvert", i("uint"), i("r7u"), i("d"))
    ins("Bitcast", i("int"), i("r7"), i("r7u"))
    out(7, i("r7"))
    ins("Return")
    ins("FunctionEnd")
    a.write(path)


def draw_params_vert(path):
    a = Asm()
    i, ins = a.i, a.ins
    ins("Capability", 1)     # Shader
    ins("Capability", 4427)  # DrawParameters
    ins("Extension", "SPV_KHR_shader_draw_parameters")
    ins("MemoryModel", 0, 1)
    ins("EntryPoint", 0, i("main"), "main", i("bv"), i("bi"), i("di"), i("p"), i("pos"))
    ins("Decorate", i("bv"), 11, 4424)  # BuiltIn BaseVertex
    ins("Decorate", i("bi"), 11, 4425)  # BuiltIn BaseInstance
    ins("Decorate", i("di"), 11, 4426)  # BuiltIn DrawIndex
    ins("Decorate", i("pos"), 11, 0)    # BuiltIn Position
    ins("Decorate", i("p"), 30, 0)      # Location 0
    ins("Decorate", i("p"), 14)         # Flat
    ins("TypeVoid", i("void"))
    ins("TypeFunction", i("fnty"), i("void"))
    ins("TypeInt", i("int"), 32, 1)
    ins("TypeInt", i("uint"), 32, 0)
    ins("TypeFloat", i("float"), 32)
    ins("TypeVector", i("v3uint"), i("uint"), 3)
    ins("TypeVector", i("v4float"), i("float"), 4)
    ins("TypePointer", i("p_in_int"), 1, i("int"))
    ins("TypePointer", i("p_out_v3"), 3, i("v3uint"))
    ins("TypePointer", i("p_out_v4"), 3, i("v4float"))
    ins("Variable", i("p_in_int"), i("bv"), 1)
    ins("Variable", i("p_in_int"), i("bi"), 1)
    ins("Variable", i("p_in_int"), i("di"), 1)
    ins("Variable", i("p_out_v3"), i("p"), 3)
    ins("Variable", i("p_out_v4"), i("pos"), 3)
    ins("Constant", i("float"), i("f0"), 0)
    ins("ConstantComposite", i("v4float"), i("zero4"), i("f0"), i("f0"), i("f0"), i("f0"))
    ins("Function", i("void"), i("main"), 0, i("fnty"))
    ins("Label", i("entry"))
    for n in ("bv", "bi", "di"):
        ins("Load", i("int"), i(n + "_v"), i(n))
        ins("Bitcast", i("uint"), i(n + "_u"), i(n + "_v"))
    ins("CompositeConstruct", i("v3uint"), i("pv"), i("bv_u"), i("bi_u"), i("di_u"))
    ins("Store", i("p"), i("pv"))
    ins("Store", i("pos"), i("zero4"))
    ins("Return")
    ins("FunctionEnd")
    a.write(path)


def vertex_store_vert(path):
    a = Asm()
    i, ins = a.i, a.ins
    ins("Capability", 1)
    ins("MemoryModel", 0, 1)
    ins("EntryPoint", 0, i("main"), "main", i("vid"), i("pos"))
    ins("Decorate", i("vid"), 11, 42)   # BuiltIn VertexIndex
    ins("Decorate", i("pos"), 11, 0)    # BuiltIn Position
    ins("Decorate", i("rta"), 6, 4)     # ArrayStride 4
    ins("MemberDecorate", i("B"), 0, 35, 0)
    ins("MemberDecorate", i("B"), 1, 35, 4)
    ins("Decorate", i("B"), 2)          # Block
    ins("Decorate", i("b"), 34, 0)
    ins("Decorate", i("b"), 33, 0)
    ins("TypeVoid", i("void"))
    ins("TypeFunction", i("fnty"), i("void"))
    ins("TypeInt", i("int"), 32, 1)
    ins("TypeInt", i("uint"), 32, 0)
    ins("TypeFloat", i("float"), 32)
    ins("TypeVector", i("v4float"), i("float"), 4)
    ins("TypeRuntimeArray", i("rta"), i("uint"))
    ins("TypeStruct", i("B"), i("uint"), i("rta"))
    ins("TypePointer", i("p_sb_B"), 12, i("B"))
    ins("TypePointer", i("p_sb_uint"), 12, i("uint"))
    ins("TypePointer", i("p_in_int"), 1, i("int"))
    ins("TypePointer", i("p_out_v4"), 3, i("v4float"))
    ins("Variable", i("p_sb_B"), i("b"), 12)
    ins("Variable", i("p_in_int"), i("vid"), 1)
    ins("Variable", i("p_out_v4"), i("pos"), 3)
    ins("Constant", i("int"), i("i0"), 0)
    ins("Constant", i("int"), i("i1"), 1)
    ins("Constant", i("uint"), i("u1"), 1)
    ins("Constant", i("uint"), i("dev"), 1)    # Scope Device
    ins("Constant", i("uint"), i("sem0"), 0)   # Relaxed
    ins("Constant", i("float"), i("f0"), 0)
    ins("ConstantComposite", i("v4float"), i("zero4"), i("f0"), i("f0"), i("f0"), i("f0"))
    ins("Function", i("void"), i("main"), 0, i("fnty"))
    ins("Label", i("entry"))
    ins("Load", i("int"), i("v"), i("vid"))
    ins("Bitcast", i("uint"), i("vu"), i("v"))
    ins("AccessChain", i("p_sb_uint"), i("pid"), i("b"), i("i1"), i("v"))
    ins("Store", i("pid"), i("vu"))
    ins("AccessChain", i("p_sb_uint"), i("pc"), i("b"), i("i0"))
    ins("AtomicIAdd", i("uint"), i("old"), i("pc"), i("dev"), i("sem0"), i("u1"))
    ins("Store", i("pos"), i("zero4"))
    ins("Return")
    ins("FunctionEnd")
    a.write(path)


int16_comp(os.path.join(sys.argv[1], "int16.comp.spv"))
vertex_store_vert(os.path.join(sys.argv[1], "vertex_store.vert.spv"))
draw_params_vert(os.path.join(sys.argv[1], "draw_params.vert.spv"))
