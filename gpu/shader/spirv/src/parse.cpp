// SPIR-V reader, first half: one pass over the words into the tables of module.h, with the checks
// that keep the lowering from reading past an instruction.
#include <cstring>

#include "module.h"
#include "vpvk/spirv_frontend.h"

namespace spirv {

using namespace spv;

std::string literalString(const SpirvModule& m, const Instr& in, uint32_t firstWord, uint32_t* wordsUsed) {
    std::string s;
    uint32_t i = firstWord;
    for (; i < in.count; ++i) {
        uint32_t word = m.words[in.at + i];
        bool end = false;
        for (int b = 0; b < 4; ++b) {
            char c = char((word >> (8 * b)) & 0xFF);
            if (!c) { end = true; break; }
            s += c;
        }
        if (end) { ++i; break; }
    }
    if (wordsUsed) *wordsUsed = i - firstWord;
    return s;
}

namespace {

struct Parser {
    SpirvModule& m;
    std::string& error;
    FunctionDef* fn = nullptr;
    Block* block = nullptr;
    std::vector<Instr> pendingModes;

    bool fail(const std::string& why) {
        error = why;
        return false;
    }
    uint32_t w(const Instr& in, uint32_t i) const { return m.words[in.at + i]; }

    bool need(const Instr& in, uint32_t n) {
        if (in.count < n) return fail("instruction " + std::to_string(in.op) + " is too short");
        return true;
    }

    uint64_t constBits(const Instr& in, uint32_t type) {
        uint64_t v = w(in, 3);
        auto t = m.type(type);
        if (t && t->width == 64 && in.count > 4) v |= uint64_t(w(in, 4)) << 32;
        return v;
    }

    bool typeDecl(const Instr& in) {
        uint32_t id = w(in, 1);
        TypeInfo t;
        switch (in.op) {
            case OpTypeVoid: t.kind = TypeKind::Void; break;
            case OpTypeBool: t.kind = TypeKind::Bool; break;
            case OpTypeInt:
                if (!need(in, 4)) return false;
                t.kind = TypeKind::Int, t.width = w(in, 2), t.sign = w(in, 3) != 0;
                break;
            case OpTypeFloat:
                if (!need(in, 3)) return false;
                t.kind = TypeKind::Float, t.width = w(in, 2);
                break;
            case OpTypeVector:
            case OpTypeMatrix:
                if (!need(in, 4)) return false;
                t.kind = in.op == OpTypeVector ? TypeKind::Vector : TypeKind::Matrix;
                t.elem = w(in, 2), t.count = w(in, 3);
                break;
            case OpTypeArray: {
                if (!need(in, 4)) return false;
                t.kind = TypeKind::Array, t.elem = w(in, 2), t.lengthId = w(in, 3);
                auto c = m.constants.find(t.lengthId);
                t.count = c == m.constants.end() ? 0 : uint32_t(c->second.bits);
                break;
            }
            case OpTypeRuntimeArray:
                if (!need(in, 3)) return false;
                t.kind = TypeKind::RuntimeArray, t.elem = w(in, 2);
                break;
            case OpTypeStruct:
                t.kind = TypeKind::Struct;
                for (uint32_t i = 2; i < in.count; ++i) t.members.push_back(w(in, i));
                break;
            case OpTypePointer:
                if (!need(in, 4)) return false;
                t.kind = TypeKind::Pointer, t.storage = Storage(w(in, 2)), t.elem = w(in, 3);
                break;
            case OpTypeFunction: t.kind = TypeKind::Function; break;
            case OpTypeImage:
                if (!need(in, 9)) return false;
                t.kind = TypeKind::Image;
                t.sampledType = w(in, 2), t.dim = Dim(w(in, 3)), t.depth = w(in, 4), t.arrayed = w(in, 5);
                t.ms = w(in, 6), t.sampled = w(in, 7), t.format = ImgFmt(w(in, 8));
                break;
            case OpTypeSampler: t.kind = TypeKind::Sampler; break;
            case OpTypeSampledImage:
                if (!need(in, 3)) return false;
                t.kind = TypeKind::SampledImage, t.elem = w(in, 2);
                break;
            default: t.kind = TypeKind::Other; break;
        }
        m.types[id] = std::move(t);
        return true;
    }

    bool constantDecl(const Instr& in) {
        if (!need(in, 3)) return false;
        ConstantDef c;
        c.type = w(in, 1);
        uint32_t id = w(in, 2);
        m.resultType[id] = c.type;
        switch (in.op) {
            case OpConstantTrue: c.bits = 1; break;
            case OpConstantFalse: c.bits = 0; break;
            case OpSpecConstantTrue: c.bits = 1, c.spec = true; break;
            case OpSpecConstantFalse: c.bits = 0, c.spec = true; break;
            case OpConstant:
            case OpSpecConstant:
                if (!need(in, 4)) return false;
                c.bits = constBits(in, c.type);
                c.spec = in.op == OpSpecConstant;
                break;
            case OpConstantComposite:
            case OpSpecConstantComposite:
                c.composite = true;
                c.spec = in.op == OpSpecConstantComposite;
                for (uint32_t i = 3; i < in.count; ++i) c.parts.push_back(w(in, i));
                break;
            case OpConstantNull: c.isNull = true; break;
            case OpSpecConstantOp:
                if (!need(in, 4)) return false;
                c.spec = true, c.specOp = in;
                break;
            default: return fail("unsupported constant instruction " + std::to_string(in.op));
        }
        m.constants[id] = std::move(c);
        return true;
    }

    bool decorate(const Instr& in) {
        if (in.op == OpDecorate || in.op == OpDecorateId) {
            if (!need(in, 3)) return false;
            auto& d = m.decorations[w(in, 1)].items[w(in, 2)];
            d.clear();
            for (uint32_t i = 3; i < in.count; ++i) d.push_back(w(in, i));
        } else if (in.op == OpMemberDecorate) {
            if (!need(in, 4)) return false;
            auto& d = m.memberDecorations[{w(in, 1), w(in, 2)}].items[w(in, 3)];
            d.clear();
            for (uint32_t i = 4; i < in.count; ++i) d.push_back(w(in, i));
        }
        return true;
    }

    bool instruction(const Instr& in) {
        const uint16_t op = in.op;
        switch (op) {
            case OpCapability:
            case OpExtension:
            case OpMemoryModel:
            case OpSource:
            case OpSourceContinued:
            case OpSourceExtension:
            case OpString:
            case OpLine:
            case OpNoLine:
            case OpModuleProcessed:
            case OpMemberName:
            case OpDecorationGroup:
            case OpGroupDecorate:
            case OpGroupMemberDecorate:
            case OpDecorateString:
            case OpMemberDecorateString:
                return true;
            case OpName:
                if (in.count >= 3) m.names[w(in, 1)] = literalString(m, in, 2);
                return true;
            case OpExtInstImport:
                if (!need(in, 3)) return false;
                if (literalString(m, in, 2) == "GLSL.std.450") m.glslExt = w(in, 1);
                return true;
            case OpEntryPoint: {
                if (!need(in, 4)) return false;
                EntryPoint e;
                e.model = Model(w(in, 1));
                e.function = w(in, 2);
                uint32_t used = 0;
                e.name = literalString(m, in, 3, &used);
                for (uint32_t i = 3 + used; i < in.count; ++i) e.interface.push_back(w(in, i));
                m.entries.push_back(std::move(e));
                return true;
            }
            case OpExecutionMode:
            case OpExecutionModeId:
                if (!need(in, 3)) return false;
                pendingModes.push_back(in);
                return true;
            case OpDecorate:
            case OpDecorateId:
            case OpMemberDecorate: return decorate(in);
            case OpTypeVoid: case OpTypeBool: case OpTypeInt: case OpTypeFloat: case OpTypeVector:
            case OpTypeMatrix: case OpTypeArray: case OpTypeRuntimeArray: case OpTypeStruct:
            case OpTypePointer: case OpTypeFunction: case OpTypeImage: case OpTypeSampler:
            case OpTypeSampledImage: case OpTypeForwardPointer: case OpTypeOpaque: case OpTypeEvent:
            case OpTypeAccelerationStructureKHR:
                if (!need(in, 2)) return false;
                return op == OpTypeForwardPointer ? true : typeDecl(in);
            case OpConstantTrue: case OpConstantFalse: case OpConstant: case OpConstantComposite:
            case OpConstantNull: case OpSpecConstantTrue: case OpSpecConstantFalse: case OpSpecConstant:
            case OpSpecConstantComposite: case OpSpecConstantOp:
                if (fn) return fail("constant inside a function");
                return constantDecl(in);
            case OpUndef:
                if (!need(in, 3)) return false;
                m.resultType[w(in, 2)] = w(in, 1);
                if (!fn) {
                    ConstantDef c;
                    c.type = w(in, 1), c.isNull = true;
                    m.constants[w(in, 2)] = c;
                    return true;
                }
                break;  // a function-level undef is an ordinary instruction
            case OpVariable:
                if (!fn) {
                    if (!need(in, 4)) return false;
                    GlobalVar g;
                    g.type = w(in, 1), g.id = w(in, 2), g.storage = Storage(w(in, 3));
                    if (in.count > 4) g.initializer = w(in, 4);
                    m.resultType[g.id] = g.type;
                    m.globals[g.id] = g;
                    m.globalOrder.push_back(g.id);
                    return true;
                }
                break;
            case OpFunction: {
                if (!need(in, 5)) return false;
                if (fn) return fail("nested OpFunction");
                FunctionDef f;
                f.resultType = w(in, 1), f.id = w(in, 2);
                fn = &(m.functions[f.id] = std::move(f));
                return true;
            }
            case OpFunctionParameter:
                if (!fn || !need(in, 3)) return fail("stray OpFunctionParameter");
                fn->params.push_back(w(in, 2));
                fn->paramTypes.push_back(w(in, 1));
                m.resultType[w(in, 2)] = w(in, 1);
                return true;
            case OpFunctionEnd:
                if (!fn) return fail("stray OpFunctionEnd");
                if (block) return fail("function " + m.nameOf(fn->id) + " ends inside a block");
                fn = nullptr;
                return true;
            case OpLabel:
                if (!fn || !need(in, 2)) return fail("label outside a function");
                if (block) return fail("block without a terminator");
                fn->blockOf[w(in, 1)] = fn->blocks.size();
                fn->blocks.push_back(Block{});
                block = &fn->blocks.back();
                block->label = w(in, 1);
                return true;
            case OpSelectionMerge:
            case OpLoopMerge:
                if (!block) return fail("merge outside a block");
                block->merge = in;
                return true;
            case OpBranch: case OpBranchConditional: case OpSwitch: case OpReturn: case OpReturnValue:
            case OpKill: case OpUnreachable: case OpTerminateInvocation:
                if (!block) return fail("terminator outside a block");
                block->terminator = in;
                block = nullptr;
                return true;
            default: break;
        }
        if (!fn || !block) {
            // Unknown module-level instruction: skipped, it cannot affect the lowering.
            return true;
        }
        block->body.push_back(in);
        if (in.count >= 3 && hasTypedResult(in.op)) m.resultType[w(in, 2)] = w(in, 1);
        return true;
    }

    // Instructions with a result type and result id: (type, id) in words 1 and 2.
    static bool hasTypedResult(uint16_t op) {
        switch (op) {
            case OpStore: case OpCopyMemory: case OpCopyMemorySized: case OpControlBarrier:
            case OpMemoryBarrier: case OpImageWrite: case OpAtomicStore: case OpEmitVertex:
            case OpEndPrimitive: case OpLifetimeStart: case OpLifetimeStop: case OpBeginInvocationInterlockEXT:
            case OpEndInvocationInterlockEXT: case OpDemoteToHelperInvocation: case OpLine: case OpNoLine:
                return false;
            default: return true;
        }
    }

    bool run(const uint32_t* words, size_t count) {
        if (count < 5) return fail("module shorter than its header");
        if (words[0] != kMagic) return fail("not SPIR-V (bad magic)");
        m.words.assign(words, words + count);
        m.bound = words[3];
        for (size_t at = 5; at < count;) {
            uint32_t first = words[at];
            Instr in;
            in.op = uint16_t(first & 0xFFFF);
            in.count = uint16_t(first >> 16);
            in.at = uint32_t(at);
            if (in.count == 0 || at + in.count > count) return fail("truncated instruction at word " + std::to_string(at));
            if (!instruction(in)) return false;
            at += in.count;
        }
        if (fn) return fail("missing OpFunctionEnd");
        for (const Instr& in : pendingModes) {
            for (auto& e : m.entries)
                if (e.function == w(in, 1)) e.modes.push_back(in);
        }
        for (auto& [id, c] : m.constants) {
            if (auto d = m.deco(id); d && d->has(Deco::SpecId)) c.specId = d->get(Deco::SpecId);
        }
        // Array lengths declared before their constant was seen cannot happen (SPIR-V orders
        // declarations), but spec-constant lengths are resolved again at lowering.
        if (m.entries.empty()) return fail("module has no entry point");
        return true;
    }
};

}  // namespace

SpirvModule* parse(const uint32_t* words, size_t wordCount, std::string& error) {
    auto* m = new SpirvModule;
    Parser p{*m, error, nullptr, nullptr, {}};
    if (!p.run(words, wordCount)) {
        delete m;
        return nullptr;
    }
    return m;
}

void release(SpirvModule* m) { delete m; }

}  // namespace spirv
