// SPIR-V reader, first half: one pass over the words into the tables of module.h, with the checks
// that keep the lowering from reading past an instruction.
#include <cstring>
#include <unordered_set>

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
    std::unordered_set<uint32_t> forwardPointers;  // OpTypeForwardPointer ids whose OpTypePointer has not come yet

    bool fail(const std::string& why) {
        error = why;
        return false;
    }
    uint32_t w(const Instr& in, uint32_t i) const { return m.w(in, i); }

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

    // A type id a composite type is built from: SPIR-V declares it before the composite (only a pointer's
    // pointee may come later, through OpTypeForwardPointer), which also rules out types that contain
    // themselves (the lowering recurses over them).
    bool declared(uint32_t id) {
        if (m.types.count(id)) return true;
        return fail("type %" + std::to_string(id) + " is used before it is declared");
    }

    bool typeDecl(const Instr& in) {
        uint32_t id = w(in, 1);
        if (m.types.count(id) && !forwardPointers.erase(id)) return fail("type %" + std::to_string(id) + " declared twice");
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
                if (!declared(t.elem)) return false;
                if (t.count == 0 || t.count > (in.op == OpTypeVector ? 16u : 4u))
                    return fail("vector or matrix of " + std::to_string(t.count) + " components");
                break;
            case OpTypeArray: {
                if (!need(in, 4)) return false;
                t.kind = TypeKind::Array, t.elem = w(in, 2), t.lengthId = w(in, 3);
                if (!declared(t.elem)) return false;
                auto c = m.constants.find(t.lengthId);
                t.count = c == m.constants.end() ? 0 : uint32_t(c->second.bits);
                break;
            }
            case OpTypeRuntimeArray:
                if (!need(in, 3)) return false;
                t.kind = TypeKind::RuntimeArray, t.elem = w(in, 2);
                if (!declared(t.elem)) return false;
                break;
            case OpTypeStruct:
                t.kind = TypeKind::Struct;
                for (uint32_t i = 2; i < in.count; ++i) {
                    if (!declared(w(in, i))) return false;
                    t.members.push_back(w(in, i));
                }
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
                if (!declared(t.elem)) return false;
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
        if (m.constants.count(id)) return fail("constant %" + std::to_string(id) + " defined twice");
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
                for (uint32_t i = 3; i < in.count; ++i) {
                    // Constituents are defined before the composite (no constant contains itself).
                    if (!m.constants.count(w(in, i))) return fail("constant %" + std::to_string(w(in, i)) + " is used before it is defined");
                    c.parts.push_back(w(in, i));
                }
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
                if (const std::string name = literalString(m, in, 2); name == "GLSL.std.450") m.glslExt = w(in, 1);
                else if (name.rfind("NonSemantic.", 0) == 0) m.nonSemanticExt.push_back(w(in, 1));
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
                if (op == OpTypeForwardPointer) {
                    // A struct may hold a pointer to itself (buffer_reference lists): the pointer is named here, before the struct
                    // that contains it, and defined by its OpTypePointer after. Until then it is a pointer to nothing.
                    if (!m.types.count(w(in, 1))) {
                        TypeInfo t;
                        t.kind = TypeKind::Pointer, t.storage = Storage(w(in, 2));
                        m.types[w(in, 1)] = t;
                        forwardPointers.insert(w(in, 1));
                    }
                    return true;
                }
                return typeDecl(in);
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
                if (m.functions.count(w(in, 2))) return fail("function %" + std::to_string(w(in, 2)) + " defined twice");
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
                if (fn->blockOf.count(w(in, 1))) return fail("label %" + std::to_string(w(in, 1)) + " defined twice");
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

    // ---- Structural checks -------------------------------------------------------------------------
    // What the lowering takes for granted and a compiler's output always satisfies (it is not spirv-val: no types of
    // operands, no capabilities): every id a declaration defines is below the bound and defined once; branch and merge
    // targets are labels of the same function; calls go to defined functions with the right number of arguments and
    // never close a cycle (Vulkan forbids recursion); the entry point is a function without parameters that has a
    // body; pointer and constant types are declared. A module that fails is refused at parse time, with a message,
    // instead of reaching the lowering with an id that means nothing.
    bool validate() {
        std::unordered_set<uint32_t> defined;
        auto define = [&](uint32_t id, const char* what) {
            if (id == 0 || id >= m.bound) return fail(std::string(what) + " %" + std::to_string(id) + " is not below the id bound " + std::to_string(m.bound));
            if (!defined.insert(id).second) return fail("id %" + std::to_string(id) + " is defined twice (" + what + ")");
            return true;
        };
        for (const auto& [id, t] : m.types) {
            if (!define(id, "type")) return false;
            if (t.kind == TypeKind::Pointer && !m.types.count(t.elem)) return fail("pointer type %" + std::to_string(id) + " has no declared pointee");
        }
        for (const auto& [id, c] : m.constants) {
            if (!define(id, "constant")) return false;
            if (!m.types.count(c.type)) return fail("constant %" + std::to_string(id) + " has an undeclared type");
        }
        for (const auto& [id, g] : m.globals) {
            if (!define(id, "variable")) return false;
            const TypeInfo* pt = m.type(g.type);
            if (!pt || pt->kind != TypeKind::Pointer) return fail("variable %" + std::to_string(id) + " is not of a pointer type");
        }
        for (const auto& [id, f] : m.functions) {
            if (!define(id, "function")) return false;
            for (uint32_t p : f.params)
                if (!define(p, "parameter")) return false;
            for (const Block& b : f.blocks)
                if (!define(b.label, "label")) return false;
        }
        for (const auto& [id, f] : m.functions) {
            auto target = [&](uint32_t label, const char* what) {
                if (!f.blockOf.count(label)) return fail(std::string(what) + " of function " + m.nameOf(id) + " is %" + std::to_string(label) + ", not one of its labels");
                return true;
            };
            for (const Block& b : f.blocks) {
                if (b.merge.op != 0) {
                    if (!target(w(b.merge, 1), "merge block")) return false;
                    if (b.merge.op == OpLoopMerge && !target(w(b.merge, 2), "continue target")) return false;
                }
                const Instr& t = b.terminator;
                if (t.op == OpBranch) {
                    if (!target(w(t, 1), "branch target")) return false;
                } else if (t.op == OpBranchConditional) {
                    if (!target(w(t, 2), "branch target") || !target(w(t, 3), "branch target")) return false;
                } else if (t.op == OpSwitch) {
                    if (!target(w(t, 2), "switch default")) return false;
                    // Case literals take as many words as the selector (1 or 2); with an unknown selector the cases are left alone.
                    auto sel = m.resultType.find(w(t, 1));
                    const TypeInfo* st = sel == m.resultType.end() ? nullptr : m.type(sel->second);
                    if (st && (st->width == 32 || st->width == 64)) {
                        const uint32_t lit = st->width / 32;
                        for (uint32_t i = 3; i + lit < t.count; i += lit + 1)
                            if (!target(w(t, i + lit), "switch target")) return false;
                    }
                }
                for (const Instr& in : b.body) {
                    if (in.op != OpFunctionCall) continue;
                    auto callee = m.functions.find(w(in, 3));
                    if (in.count < 4 || callee == m.functions.end()) return fail("function " + m.nameOf(id) + " calls %" + std::to_string(w(in, 3)) + ", which is not a function");
                    if (size_t(in.count - 4) != callee->second.params.size())
                        return fail("call of " + m.nameOf(callee->first) + " passes " + std::to_string(in.count - 4) + " arguments, expected " + std::to_string(callee->second.params.size()));
                }
            }
        }
        // Calls never close a cycle: depth-first over the call graph with an explicit stack.
        std::unordered_map<uint32_t, int> state;  // 1 on the path, 2 finished
        for (const auto& [root, rf] : m.functions) {
            if (state[root]) continue;
            struct Frame {
                const FunctionDef* f;
                uint32_t id;
                std::vector<uint32_t> callees;
                size_t next = 0;
            };
            auto calleesOf = [&](const FunctionDef& f) {
                std::vector<uint32_t> c;
                for (const Block& b : f.blocks)
                    for (const Instr& in : b.body)
                        if (in.op == OpFunctionCall) c.push_back(w(in, 3));
                return c;
            };
            std::vector<Frame> stack;
            state[root] = 1;
            stack.push_back({&rf, root, calleesOf(rf)});
            while (!stack.empty()) {
                Frame& fr = stack.back();
                if (fr.next == fr.callees.size()) {
                    state[fr.id] = 2;
                    stack.pop_back();
                    continue;
                }
                const uint32_t c = fr.callees[fr.next++];
                if (state[c] == 1) return fail("function " + m.nameOf(c) + " calls itself (recursion)");
                if (state[c] == 2) continue;
                state[c] = 1;
                stack.push_back({&m.functions.at(c), c, calleesOf(m.functions.at(c))});
            }
        }
        for (const EntryPoint& e : m.entries) {
            auto f = m.functions.find(e.function);
            if (f == m.functions.end()) return fail("entry point \"" + e.name + "\" names %" + std::to_string(e.function) + ", which is not a function");
            if (!f->second.params.empty()) return fail("entry point \"" + e.name + "\" has parameters");
            for (uint32_t v : e.interface)
                if (!m.globals.count(v)) return fail("entry point \"" + e.name + "\" lists %" + std::to_string(v) + " in its interface, which is not a variable");
        }
        return true;
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
        return validate();
    }
};

}  // namespace

SpirvModule* parse(const uint32_t* words, size_t wordCount, std::string& error) {
    auto* m = new SpirvModule;
    Parser p{*m, error, nullptr, nullptr, {}, {}};
    if (!p.run(words, wordCount)) {
        delete m;
        return nullptr;
    }
    return m;
}

void release(SpirvModule* m) { delete m; }

bool declaresLayer(const SpirvModule& m) {
    auto isLayer = [](const Decorations& d) {
        return d.has(spv::Deco::BuiltIn) && d.get(spv::Deco::BuiltIn) == uint32_t(spv::BuiltIn::Layer);
    };
    for (const auto& [id, d] : m.decorations)
        if (isLayer(d)) return true;
    for (const auto& [member, d] : m.memberDecorations)
        if (isLayer(d)) return true;
    return false;
}

bool declaresSampleBuiltin(const SpirvModule& m) {
    auto isSample = [](const Decorations& d) {
        if (!d.has(spv::Deco::BuiltIn)) return false;
        const uint32_t b = d.get(spv::Deco::BuiltIn);
        return b == uint32_t(spv::BuiltIn::SampleId) || b == uint32_t(spv::BuiltIn::SampleMask);
    };
    for (const auto& [id, d] : m.decorations)
        if (isSample(d)) return true;
    for (const auto& [member, d] : m.memberDecorations)
        if (isSample(d)) return true;
    return false;
}

}  // namespace spirv
