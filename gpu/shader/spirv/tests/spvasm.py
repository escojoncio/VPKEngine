#!/usr/bin/env python3
# Minimal SPIR-V text assembler (no glslang / spirv-as needed). Usage: spvasm.py in.spvasm out.spv
#
# Syntax, one instruction per line, `;` starts a comment:
#   %r = OpIAdd %type %a %b        result id (and type) first, as spirv-as
#   OpStore %p %v
#   OpName %main "main"            strings in double quotes
#   OpCapability Cap:Shader        enumerants are class-qualified: Cap: Storage: Deco: BuiltIn: Dim: Model:
#   OpDecorate %x Deco:Binding 0   ImgFmt: ExecMode: AddrModel: MemModel: Scope: (see ENUMS below)
#   OpConstant %f32 f32:1.5        literals: 12, 0x10, -3 (32-bit), f32:1.5, f16:1.5, u64:123, s64:-5
#   OpSelectionMerge %m 0          masks as plain numbers; `a|b` ORs enumerant/number operands
# Ids are `%name` (allocated in order of first appearance). The version is SPIR-V 1.3 by default
# (`; version 1.5` as the first comment line overrides it).
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENUM_FILE = HERE.parent / "src" / "spv_enums.h"

# Class-qualified enumerants. Names from spv_enums.h where present, plus capabilities (not in the header).
CAPS = {
    "Matrix": 0, "Shader": 1, "Geometry": 2, "Tessellation": 3, "Addresses": 4, "Linkage": 5, "Kernel": 6,
    "Float16": 9, "Float64": 10, "Int64": 11, "Int16": 22, "ImageGatherExtended": 25, "StorageImageExtendedFormats": 49,
    "SampledBuffer": 46, "ImageBuffer": 47, "ImageQuery": 50, "DerivativeControl": 51, "InterpolationFunction": 52,
    "StorageImageReadWithoutFormat": 55, "StorageImageWriteWithoutFormat": 56, "MultiViewport": 57,
    "GroupNonUniform": 61, "GroupNonUniformVote": 62, "GroupNonUniformArithmetic": 63, "GroupNonUniformBallot": 64,
    "GroupNonUniformShuffle": 65, "GroupNonUniformQuad": 68, "StorageBuffer16BitAccess": 4433,
    "UniformAndStorageBuffer16BitAccess": 4434, "StoragePushConstant16": 4435, "StorageInputOutput16": 4436,
    "DrawParameters": 4427, "MultiView": 4439, "VariablePointersStorageBuffer": 4441, "VariablePointers": 4442,
    "Int8": 39, "StorageBuffer8BitAccess": 4448, "UniformAndStorageBuffer8BitAccess": 4449, "StoragePushConstant8": 4450,
    "ShaderViewportIndexLayerEXT": 5254, "DemoteToHelperInvocation": 5379, "VulkanMemoryModel": 5345,
    "VulkanMemoryModelDeviceScope": 5346, "PhysicalStorageBufferAddresses": 5347, "RuntimeDescriptorArray": 5302,
    "FloatControls2": 6029, "SubgroupBallotKHR": 4423, "InputAttachment": 40, "SampledRect": 37,
    "Sampled1D": 43, "Image1D": 44, "SampledCubeArray": 45, "ImageCubeArray": 34, "StorageImageMultisample": 27,
    "ShaderNonUniform": 5301, "UniformBufferArrayDynamicIndexing": 28, "SampledImageArrayDynamicIndexing": 29,
    "StorageBufferArrayDynamicIndexing": 30, "StorageImageArrayDynamicIndexing": 31, "ClipDistance": 32,
    "CullDistance": 33, "SampleRateShading": 35, "MinLod": 42, "Int64Atomics": 12, "AtomicFloat32AddEXT": 6033,
    "StorageTexelBufferArrayDynamicIndexing": 5305, "SignedZeroInfNanPreserve": 4466, "DenormPreserve": 4464,
    "DotProduct": 6019, "DotProductInput4x8Bit": 6016, "DotProductInputAll": 6017, "DotProductInput4x8BitPacked": 6018,
    "TerminateInvocation": 4066, "ShaderClockKHR": 5055, "FragmentShaderSampleInterlockEXT": 5363,
}
MISC = {
    "AddrModel": {"Logical": 0, "Physical32": 1, "Physical64": 2, "PhysicalStorageBuffer64": 5348},
    "MemModel": {"Simple": 0, "GLSL450": 1, "OpenCL": 2, "Vulkan": 3},
    "Scope": {"CrossDevice": 0, "Device": 1, "Workgroup": 2, "Subgroup": 3, "Invocation": 4, "QueueFamily": 5},
    "ExecMode": {"Invocations": 0, "SpacingEqual": 1, "OriginUpperLeft": 7, "OriginLowerLeft": 8, "EarlyFragmentTests": 9,
                 "DepthReplacing": 12, "LocalSize": 17, "LocalSizeId": 38, "DepthGreater": 14, "DepthLess": 15,
                 "DepthUnchanged": 16, "PixelCenterInteger": 6, "DemoteToHelperInvocation": 5379},
    "Sem": {"None": 0, "Acquire": 2, "Release": 4, "AcquireRelease": 8, "SequentiallyConsistent": 16,
            "UniformMemory": 64, "SubgroupMemory": 128, "WorkgroupMemory": 256, "CrossWorkgroupMemory": 512,
            "ImageMemory": 2048, "OutputMemory": 4096, "MakeAvailable": 8192, "MakeVisible": 16384, "Volatile": 32768},
    "MemAccess": {"None": 0, "Volatile": 1, "Aligned": 2, "Nontemporal": 4, "MakePointerAvailable": 8,
                  "MakePointerVisible": 16, "NonPrivatePointer": 32},
    "FnControl": {"None": 0, "Inline": 1, "DontInline": 2, "Pure": 4, "Const": 8},
    "ImgOp": {"None": 0, "Bias": 1, "Lod": 2, "Grad": 4, "ConstOffset": 8, "Offset": 16, "Sample": 64, "MinLod": 128},
    "SelCtl": {"None": 0, "Flatten": 1, "DontFlatten": 2},
    "LoopCtl": {"None": 0, "Unroll": 1, "DontUnroll": 2},
    "Group": {"Reduce": 0, "InclusiveScan": 1, "ExclusiveScan": 2, "ClusteredReduce": 3},
}
NOTYPE = {"Label", "ExtInstImport", "String", "DecorationGroup"} | {"Type" + n for n in (
    "Void", "Bool", "Int", "Float", "Vector", "Matrix", "Image", "Sampler", "SampledImage", "Array", "RuntimeArray",
    "Struct", "Opaque", "Pointer", "Function", "Event", "DeviceEvent", "ReserveId", "Queue", "Pipe")}
NORESULT = {
    "Nop", "SourceContinued", "Source", "SourceExtension", "Name", "MemberName", "Line", "NoLine", "Extension",
    "MemoryModel", "EntryPoint", "ExecutionMode", "ExecutionModeId", "Capability", "Decorate", "MemberDecorate",
    "DecorateId", "GroupDecorate", "GroupMemberDecorate", "Store", "CopyMemory", "CopyMemorySized", "FunctionEnd",
    "ImageWrite", "EmitVertex", "EndPrimitive", "ControlBarrier", "MemoryBarrier", "LoopMerge", "SelectionMerge",
    "Branch", "BranchConditional", "Switch", "Kill", "Return", "ReturnValue", "Unreachable", "TerminateInvocation",
    "DemoteToHelperInvocation", "TypeForwardPointer", "AtomicStore", "ExecutionMode", "DecorateString",
    "MemberDecorateString", "ModuleProcessed",
}


def load_ops():
    ops = {}
    text = ENUM_FILE.read_text()
    m = re.search(r"enum SpvOp[^{]*\{(.*?)\n\};", text, re.S)
    for name, val in re.findall(r"Op(\w+)\s*=\s*(\d+)", m.group(1)):
        ops[name] = int(val)
    return ops


def load_enum(cls):
    text = ENUM_FILE.read_text()
    m = re.search(r"enum class %s\b[^{]*\{(.*?)\n\};" % cls, text, re.S)
    return {n: int(v) for n, v in re.findall(r"(\w+)\s*=\s*(\d+)", m.group(1))} if m else {}


ENUMS = {"Cap": CAPS, "Storage": load_enum("Storage"), "Deco": load_enum("Deco"), "BuiltIn": load_enum("BuiltIn"),
         "Dim": load_enum("Dim"), "Model": load_enum("Model"), "ImgFmt": load_enum("ImgFmt")}
ENUMS.update(MISC)
OPS = load_ops()


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f16bits(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def tokens(line):
    return re.findall(r'"(?:[^"\\]|\\.)*"|\S+', line)


def assemble(src):
    ids = {}
    words = []
    version = 0x00010300

    def idof(name):
        if name not in ids:
            ids[name] = len(ids) + 1
        return ids[name]

    def operand(tok):
        if tok.startswith('"'):
            b = bytes(tok[1:-1], "utf-8").decode("unicode_escape").encode("latin1") + b"\0"
            b += b"\0" * (-len(b) % 4)
            return list(struct.unpack("<%dI" % (len(b) // 4), b))
        if tok.startswith("%"):
            return [idof(tok[1:])]
        if tok.startswith("f32:"):
            return [f32bits(float(tok[4:]))]
        if tok.startswith("f16:"):
            return [f16bits(float(tok[4:]))]
        if tok.startswith(("u64:", "s64:")):
            v = int(tok[4:], 0) & ((1 << 64) - 1)
            return [v & 0xFFFFFFFF, v >> 32]
        out = 0
        for part in tok.split("|"):
            if ":" in part:
                cls, name = part.split(":", 1)
                out |= ENUMS[cls][name]
            else:
                out |= int(part, 0) & 0xFFFFFFFF
        return [out]

    for raw in src.splitlines():
        m = re.match(r"\s*;\s*version\s+(\d+)\.(\d+)", raw)
        if m:
            version = int(m.group(1)) << 16 | int(m.group(2)) << 8
        toks = tokens(raw)
        for i, t in enumerate(toks):
            if t.startswith(";"):
                toks = toks[:i]
                break
        if not toks:
            continue
        result = None
        if len(toks) >= 2 and toks[1] == "=":
            result = toks[0]
            toks = toks[2:]
        name = toks[0]
        assert name.startswith("Op"), line
        name = name[2:]
        opcode = OPS[name]
        args = toks[1:]
        ws = []
        if result is not None:
            if name in NOTYPE:
                ws += [idof(result[1:])]
            else:
                ws += [idof(args[0][1:]), idof(result[1:])]
                args = args[1:]
        elif name not in NORESULT and name not in NOTYPE:
            raise SystemExit("missing result: " + line)
        for a in args:
            ws += operand(a)
        words.append(((len(ws) + 1) << 16) | opcode)
        words.extend(ws)
    header = [0x07230203, version, 0, len(ids) + 1, 0]
    return struct.pack("<%dI" % (len(header) + len(words)), *(header + words))


if __name__ == "__main__":
    Path(sys.argv[2]).write_bytes(assemble(Path(sys.argv[1]).read_text()))
