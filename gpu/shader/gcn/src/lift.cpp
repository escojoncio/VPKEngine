// GCN 2 -> VPIR lifter. See vpgcn/lift.h for the execution model.
#include "vpgcn/lift.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <unordered_map>

#include "vpgcn/gcn.h"
#include "vpir/builder.h"

namespace vpgcn {
namespace {

namespace V = vpir;
using G = vpgcn::Op;
using V::ValueId;

constexpr uint16_t kVcc = 106, kExec = 126, kM0 = 124;

struct Decoded {
    uint32_t pc = 0;  // dword offset in its program
    Instruction in;
    bool fetch = false;  // from the fetch subroutine (inlined at s_swappc_b64)
};

// Basic block: [first, last] in `prog`, then its terminator.
struct BasicBlock {
    size_t first = 0, last = 0;
    enum Term : uint8_t { Fall, Jump, Cond, End } term = Fall;
    G cond = G::Count;       // the s_cbranch_* opcode
    int taken = -1, next = -1;  // successor blocks (-1 = none)
};

// Where an SGPR's value came from, for descriptor tracing.
struct Origin {
    // User: user data SGPR `sgpr`. Load: the dword at `path` from the pointer in user SGPRs
    // sgpr/sgpr+1 (see DescriptorSource::path).
    // Code / CodeHi: low / high dword of an address into the program image (path[0] = byte offset).
    // LoadHi: the high dword of a loaded 64-bit address with flag bits ORed in (a V# built from a
    // pointer: base address + stride in dword 1).
    enum Kind : uint8_t { Unknown, User, Load, Code, CodeHi, LoadHi } kind = Unknown;
    uint16_t sgpr = 0;
    std::vector<uint32_t> path;
};

}  // namespace

struct Lifter {
    const LiftInput& in;
    LiftResult& out;
    V::Module& m;
    V::FunctionBuilder* b = nullptr;

    std::vector<Decoded> prog;
    std::vector<BasicBlock> blocks;
    std::set<std::string> errorSet, noteSet;

    // Variables, created on first use.
    std::unordered_map<uint32_t, uint32_t> sgprVar;  // operand code -> var (u32)
    std::unordered_map<uint32_t, uint32_t> maskVar;  // pair code -> var (bool)
    std::array<int64_t, 256> vgprVar;
    int64_t m0Var = -1, sccVar = -1, execVar = -1, vccVar = -1;
    ValueId laneIndex = V::kNoValue;  // compute: lane within its 64-lane wave
    ValueId quadLane = V::kNoValue;   // pixel / compute: lane within its 2x2 quad (x + 2y)
    std::set<uint32_t> maskPairs;  // SGPR pair codes holding lane masks
    bool writesExec = false;       // VGPR writes must honour EXEC

    // Descriptor tracing: per instruction (index in prog), the source of its descriptor operands.
    std::unordered_map<size_t, DescriptorSource> rsrcSource, sampSource;
    std::unordered_map<size_t, DescriptorSource> pointerSource;  // s_load_*: the pointer's source
    std::set<size_t> rawBuffer;  // buffer accesses whose V# is built from an address in memory

    // Resource slots.
    std::map<std::pair<int, DescriptorSource>, uint32_t> slots;
    std::optional<uint32_t> userSlot;
    std::set<DescriptorSource> storageUses;  // T# sources written by the program

    // Pixel: barycentric VGPRs and pending interpolation.
    struct Bary {
        V::Interp interp;
        V::Sampling sampling;
        bool j;
    };
    std::unordered_map<unsigned, Bary> bary;
    std::map<std::pair<unsigned, int>, uint32_t> inputs;  // (attr, mode key) -> input index
    std::map<unsigned, uint32_t> paramOutputs, mrtOutputs;
    int64_t positionOut = -1, depthOut = -1;
    uint32_t perKindCount[5] = {};

    Lifter(const LiftInput& i, LiftResult& o) : in(i), out(o), m(o.module) { vgprVar.fill(-1); }

    void error(std::string s) { errorSet.insert(std::move(s)); }
    void note(std::string s) { noteSet.insert(std::move(s)); }

    // ---- Decoding and the CFG -----------------------------------------------------------------

    bool decodeProgram(std::span<const uint32_t> code, bool fetch, std::vector<Decoded>& dst) {
        // Every instruction reachable from the entry (branch targets included).
        std::set<uint32_t> todo{0}, done;
        std::map<uint32_t, Decoded> found;
        while (!todo.empty()) {
            uint32_t pc = *todo.begin();
            todo.erase(todo.begin());
            while (pc < code.size() && !done.count(pc)) {
                done.insert(pc);
                Decoded d;
                d.pc = pc;
                d.fetch = fetch;
                if (!decode(code.subspan(pc), d.in)) {
                    char buf[64];
                    std::snprintf(buf, sizeof buf, "undecodable word 0x%08x at dword %u", code[pc], pc);
                    error(buf);
                    return false;
                }
                found[pc] = d;
                G op = d.in.op;
                uint32_t next = pc + d.in.size / 4;
                if (d.in.encoding == Encoding::SOPP && isBranch(op)) {
                    uint32_t target = uint32_t(int32_t(next) + int16_t(d.in.simm16));
                    todo.insert(target);
                    if (op == G::s_branch) break;
                }
                if (op == G::s_endpgm || (fetch && op == G::s_setpc_b64)) break;
                pc = next;
            }
        }
        for (auto& [pc, d] : found) dst.push_back(d);
        return true;
    }

    static bool isBranch(G op) {
        switch (op) {
            case G::s_branch:
            case G::s_cbranch_scc0:
            case G::s_cbranch_scc1:
            case G::s_cbranch_vccz:
            case G::s_cbranch_vccnz:
            case G::s_cbranch_execz:
            case G::s_cbranch_execnz: return true;
            default: return false;
        }
    }

    void buildBlocks() {
        std::set<uint32_t> leaders{0};
        for (const Decoded& d : prog) {
            uint32_t next = d.pc + d.in.size / 4;
            if (d.in.encoding == Encoding::SOPP && isBranch(d.in.op)) {
                leaders.insert(uint32_t(int32_t(next) + int16_t(d.in.simm16)));
                leaders.insert(next);
            }
            if (d.in.op == G::s_endpgm) leaders.insert(next);
        }
        std::unordered_map<uint32_t, int> blockAt;
        for (size_t i = 0; i < prog.size(); ++i) {
            bool lead = leaders.count(prog[i].pc) || i == 0 || prog[i].pc != prog[i - 1].pc + prog[i - 1].in.size / 4;
            if (lead) {
                blockAt[prog[i].pc] = int(blocks.size());
                blocks.push_back({i, i});
            } else {
                blocks.back().last = i;
            }
        }
        for (size_t k = 0; k < blocks.size(); ++k) {
            BasicBlock& bb = blocks[k];
            const Decoded& d = prog[bb.last];
            uint32_t next = d.pc + d.in.size / 4;
            auto at = [&](uint32_t pc) {
                auto it = blockAt.find(pc);
                return it == blockAt.end() ? -1 : it->second;
            };
            if (d.in.op == G::s_endpgm) {
                bb.term = BasicBlock::End;
            } else if (d.in.encoding == Encoding::SOPP && isBranch(d.in.op)) {
                bb.taken = at(uint32_t(int32_t(next) + int16_t(d.in.simm16)));
                if (d.in.op == G::s_branch) {
                    bb.term = BasicBlock::Jump;
                } else {
                    bb.term = BasicBlock::Cond;
                    bb.cond = d.in.op;
                    bb.next = at(next);
                }
            } else {
                bb.next = at(next);
                bb.term = bb.next < 0 ? BasicBlock::End : BasicBlock::Fall;
                if (bb.next < 0) error("the program runs past its end without s_endpgm");
            }
        }
    }

    // ---- Analyses -------------------------------------------------------------------------------

    static bool isPairOp(G op) {
        switch (op) {
            case G::s_mov_b64:
            case G::s_cmov_b64:
            case G::s_not_b64:
            case G::s_and_b64:
            case G::s_or_b64:
            case G::s_xor_b64:
            case G::s_andn2_b64:
            case G::s_orn2_b64:
            case G::s_nand_b64:
            case G::s_nor_b64:
            case G::s_xnor_b64:
            case G::s_cselect_b64:
            case G::s_and_saveexec_b64:
            case G::s_or_saveexec_b64:
            case G::s_xor_saveexec_b64:
            case G::s_andn2_saveexec_b64:
            case G::s_orn2_saveexec_b64:
            case G::s_nand_saveexec_b64:
            case G::s_nor_saveexec_b64:
            case G::s_xnor_saveexec_b64:
            case G::s_wqm_b64: return true;
            default: return false;
        }
    }
    static bool isSaveexec(G op) { return op >= G::s_and_saveexec_b64 && op <= G::s_xnor_saveexec_b64; }

    static bool isMaskCode(uint16_t code) { return code == kVcc || code == kExec; }

    // SGPR pairs that hold lane masks: written by compares/carries/saveexec, or combined with one.
    // Instructions in execution order: the fetch subroutine inlined at its s_swappc_b64.
    std::vector<size_t> order;
    void buildOrder() {
        for (size_t i = 0; i < mainCount; ++i) {
            order.push_back(i);
            if (prog[i].in.op == G::s_swappc_b64) order.insert(order.end(), fetchIndex.begin(), fetchIndex.end());
        }
    }
    size_t mainCount = 0;

    void findMasks() {
        bool changed = true;
        auto isMask = [&](const Operand& o) {
            return o.count == 2 && (isMaskCode(o.code) || maskPairs.count(o.code));
        };
        auto mark = [&](const Operand& o) {
            if (o.count == 2 && (o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp) && maskPairs.insert(o.code).second)
                changed = true;
        };
        for (size_t oi : order) {
            const Instruction& x = prog[oi].in;
            if (x.encoding == Encoding::VOP3 || x.encoding == Encoding::VOPC) mark(x.sdst);
            if (x.encoding == Encoding::VOP3 || x.encoding == Encoding::VOP2)
                for (const Operand& s : x.src)
                    if (s.count == 2 && (x.op == G::v_cndmask_b32 || x.op == G::v_addc_u32 || x.op == G::v_subb_u32 ||
                                         x.op == G::v_subbrev_u32 || x.op == G::v_div_fmas_f32))
                        mark(s);
            if (isSaveexec(x.op)) mark(x.dst);
            if (x.op == G::s_and_b64 || x.op == G::s_or_b64 || x.op == G::s_xor_b64 || x.op == G::s_andn2_b64 ||
                x.op == G::s_orn2_b64 || x.op == G::s_nand_b64 || x.op == G::s_nor_b64 || x.op == G::s_xnor_b64)
                if (isMaskCode(x.dst.code)) mark(x.src[0]), mark(x.src[1]);
            if (x.op == G::s_mov_b64 && isMaskCode(x.dst.code)) mark(x.src[0]);
        }
        while (changed) {
            changed = false;
            for (size_t oi : order) {
                const Instruction& x = prog[oi].in;
                if (!isPairOp(x.op)) continue;
                bool any = isMask(x.dst) || isMask(x.src[0]) || isMask(x.src[1]);
                if (!any) continue;
                mark(x.dst);
                mark(x.src[0]);
                mark(x.src[1]);
            }
        }
    }

    // Descriptor sources: a forward walk in program order (descriptors are set up before use, in
    // the entry blocks, in every program seen; a later redefinition wins).
    void traceOrigins() {
        std::array<Origin, 140> o{};  // SGPR/ttmp codes (< 124) plus the widest descriptor
        for (unsigned i = 0; i < in.userSgprs && i < 104; ++i) o[i] = {Origin::User, uint16_t(i), {}};
        // The source of `n` consecutive SGPRs from `r`, if they are consecutive user SGPRs or
        // consecutive dwords loaded from one place.
        auto range = [&](const Operand& r, unsigned n) -> std::optional<DescriptorSource> {
            if (r.kind != OperandKind::Sgpr && r.kind != OperandKind::Ttmp) return std::nullopt;
            const Origin& f = o[r.code];
            if (f.kind == Origin::Unknown || f.kind == Origin::CodeHi || f.kind == Origin::LoadHi) return std::nullopt;
            if (f.kind == Origin::Load && n >= 2 && o[r.code + 1].kind == Origin::LoadHi) {
                // V# = {loaded address, its high dword | stride, size, format}: memory at the address
                // stored at f.path (the caller marks it raw).
                const Origin& h = o[r.code + 1];
                if (h.sgpr == f.sgpr && h.path.size() == f.path.size() && h.path.back() == f.path.back() + 4)
                    return DescriptorSource{DescriptorSource::Pointer, f.sgpr, f.path};
                return std::nullopt;
            }
            if (f.kind == Origin::Code) {
                // A descriptor whose base is an address in the program image (dwords 2-3 are its
                // size and format, set by the program).
                if (n < 2 || o[r.code + 1].kind != Origin::CodeHi) return std::nullopt;
                return DescriptorSource{DescriptorSource::Code, 0, f.path};
            }
            for (unsigned k = 1; f.kind == Origin::Load && k < n; ++k) {
                const Origin& x = o[r.code + k];
                if (x.kind != f.kind || x.sgpr != f.sgpr) return std::nullopt;
                {
                    if (x.path.size() != f.path.size() || x.path.back() != f.path.back() + 4 * k ||
                        !std::equal(f.path.begin(), f.path.end() - 1, x.path.begin()))
                        return std::nullopt;
                }
            }
            if (f.kind == Origin::User) {
                // (User origins are per register: consecutive registers must be consecutive user SGPRs.)
                for (unsigned k = 1; k < n; ++k)
                    if (o[r.code + k].kind != Origin::User || o[r.code + k].sgpr != f.sgpr + k) return std::nullopt;
                return DescriptorSource{DescriptorSource::UserData, f.sgpr, {}};
            }
            return DescriptorSource{DescriptorSource::Pointer, f.sgpr, f.path};
        };
        // A pointer pair's origin as the start of a load path: user SGPRs (path empty) or loaded.
        auto pointerOf = [&](const Operand& r) -> std::optional<Origin> {
            if (r.kind != OperandKind::Sgpr && r.kind != OperandKind::Ttmp) return std::nullopt;
            const Origin& lo = o[r.code];
            const Origin& hi = o[r.code + 1];
            if (lo.kind == Origin::User && hi.kind == Origin::User && hi.sgpr == lo.sgpr + 1) return Origin{Origin::Load, lo.sgpr, {}};
            if (lo.kind == Origin::Load && hi.kind == Origin::Load && hi.sgpr == lo.sgpr && hi.path.size() == lo.path.size() &&
                hi.path.back() == lo.path.back() + 4 && std::equal(lo.path.begin(), lo.path.end() - 1, hi.path.begin()))
                return lo;  // its path ends at the pointer's own offset
            return std::nullopt;
        };
        auto clobber = [&](const Operand& d) {
            if (d.kind == OperandKind::Sgpr || d.kind == OperandKind::Ttmp)
                for (unsigned k = 0; k < d.count; ++k) o[d.code + k] = {};
        };
        for (size_t i : order) {
            const Instruction& x = prog[i].in;
            switch (x.encoding) {
                case Encoding::SMRD: {
                    bool buffer = x.op >= G::s_buffer_load_dword && x.op <= G::s_buffer_load_dwordx16;
                    if (x.dst.kind == OperandKind::None) break;
                    // Read the base's origin before the destination (which may overlap it) is written.
                    std::optional<Origin> ptr;
                    if (buffer) {
                        if (auto src = range(x.src[0], 4)) rsrcSource[i] = *src;
                    } else if ((ptr = pointerOf(x.src[0]))) {
                        DescriptorSource ps{DescriptorSource::Pointer, ptr->sgpr, ptr->path};
                        if (ptr->path.empty()) ps = {DescriptorSource::UserData, ptr->sgpr, {}};
                        pointerSource[i] = ps;
                    }
                    clobber(x.dst);
                    // A load at a constant offset from a traced pointer: its dwords keep that origin.
                    if (ptr && x.src[1].kind != OperandKind::Sgpr && x.src[1].kind != OperandKind::Special) {
                        uint32_t off = x.src[1].value * 4;  // immediate and literal offsets are in dwords
                        for (unsigned k = 0; k < x.dst.count; ++k) {
                            if (x.dst.code + k >= o.size()) break;
                            Origin d{Origin::Load, ptr->sgpr, ptr->path};
                            d.path.push_back(off + 4 * k);
                            o[x.dst.code + k] = d;
                        }
                    }
                    break;
                }
                case Encoding::SOP2:
                    // Address arithmetic on an s_getpc_b64 result: s_add_u32 lo, imm / s_addc_u32 hi, 0.
                    if (x.dst.kind == OperandKind::Sgpr && (x.op == G::s_add_u32 || x.op == G::s_addc_u32 || x.op == G::s_or_b32)) {
                        const Operand* reg = nullptr;
                        const Operand* imm = nullptr;
                        for (unsigned k = 0; k < 2; ++k) {
                            const Operand& a2 = x.src[k];
                            if (a2.kind == OperandKind::Sgpr) reg = &a2;
                            else if (a2.kind == OperandKind::Const || a2.kind == OperandKind::Literal) imm = &a2;
                        }
                        if (reg && imm && x.op == G::s_or_b32 && o[reg->code].kind == Origin::Load) {
                            Origin src = o[reg->code];
                            src.kind = Origin::LoadHi;
                            o[x.dst.code] = src;
                            break;
                        }
                        if (reg && imm) {
                            Origin src = o[reg->code];
                            if (x.op == G::s_add_u32 && src.kind == Origin::Code) {
                                src.path[0] += imm->value;
                                o[x.dst.code] = src;
                                break;
                            }
                            if (x.op == G::s_addc_u32 && src.kind == Origin::CodeHi && imm->value == 0) {
                                o[x.dst.code] = src;
                                break;
                            }
                        }
                    }
                    clobber(x.dst);
                    break;
                case Encoding::SOP1:
                    if (x.op == G::s_getpc_b64 && x.dst.kind == OperandKind::Sgpr) {
                        uint32_t next = (prog[i].pc + 1) * 4;  // the address of the next instruction
                        o[x.dst.code] = {Origin::Code, 0, {next}};
                        o[x.dst.code + 1] = {Origin::CodeHi, 0, {}};
                        break;
                    }
                    if ((x.op == G::s_mov_b32 || x.op == G::s_mov_b64) &&
                        (x.dst.kind == OperandKind::Sgpr || x.dst.kind == OperandKind::Ttmp)) {
                        std::array<Origin, 2> tmp{};
                        if (x.src[0].kind == OperandKind::Sgpr || x.src[0].kind == OperandKind::Ttmp)
                            for (unsigned k = 0; k < x.src[0].count; ++k) tmp[k] = o[x.src[0].code + k];
                        for (unsigned k = 0; k < x.dst.count; ++k) o[x.dst.code + k] = tmp[k];
                        break;
                    }
                    clobber(x.dst);
                    break;
                case Encoding::MUBUF:
                case Encoding::MTBUF:
                    if (auto s2 = range(x.src[2], 4)) {
                        rsrcSource[i] = *s2;
                        if (o[x.src[2].code].kind == Origin::Load && o[x.src[2].code + 1].kind == Origin::LoadHi) rawBuffer.insert(i);
                    }
                    break;
                case Encoding::MIMG:
                    if (auto s2 = range(x.src[2], x.src[2].count)) rsrcSource[i] = *s2;
                    if (x.src[3].kind != OperandKind::None)
                        if (auto s2 = range(x.src[3], 4)) sampSource[i] = *s2;
                    break;
                default:
                    clobber(x.dst);
                    if (x.encoding == Encoding::VOP3 || x.encoding == Encoding::VOPC) clobber(x.sdst);
                    break;
            }
        }
    }

    // ---- Variables and operands -------------------------------------------------------------

    uint32_t var(int64_t& slot, V::Type t) {
        if (slot < 0) slot = b->variable(t);
        return uint32_t(slot);
    }
    uint32_t sVar(uint32_t code) {
        auto it = sgprVar.find(code);
        if (it != sgprVar.end()) return it->second;
        return sgprVar[code] = b->variable(V::kU32);
    }
    uint32_t mVar(uint32_t pair) {
        auto it = maskVar.find(pair);
        if (it != maskVar.end()) return it->second;
        return maskVar[pair] = b->variable(V::kBool);
    }
    uint32_t vVar(unsigned i) {
        if (i >= 256) {
            error("VGPR index past v255");
            i = 255;
        }
        return var(vgprVar[i], V::kU32);
    }
    uint32_t execV() { return var(execVar, V::kBool); }
    uint32_t vccV() { return var(vccVar, V::kBool); }
    uint32_t sccV() { return var(sccVar, V::kBool); }
    uint32_t m0V() { return var(m0Var, V::kU32); }

    ValueId u(uint32_t v) { return b->constU32(v); }
    ValueId f(float v) { return b->constF32(v); }
    ValueId bit(ValueId x, V::Type t) { return b->unary(V::Op::Bitcast, x, t); }
    ValueId asF(ValueId x) { return bit(x, V::kF32); }
    ValueId asU(ValueId x, V::Type from) { return from == V::kU32 ? x : bit(x, V::kU32); }
    ValueId asI(ValueId x) { return bit(x, V::kI32); }
    ValueId bin(V::Op op, ValueId a, ValueId c, V::Type t) { return b->binary(op, a, c, t); }
    ValueId sel(ValueId c, ValueId a, ValueId x, V::Type t) { return b->ternary(V::Op::Select, c, a, x, t); }
    ValueId b2u(ValueId c) { return sel(c, u(1), u(0), V::kU32); }
    ValueId nz(ValueId x) { return bin(V::Op::Ne, x, u(0), V::kBool); }
    ValueId notB(ValueId c) { return b->unary(V::Op::Not, c, V::kBool); }

    bool isPairMask(const Operand& o) {
        return o.count == 2 && (isMaskCode(o.code) || maskPairs.count(o.code));
    }

    // A lane mask operand as this lane's bit.
    // VCC and SGPR pairs used as lane masks keep two forms: the dwords as data (u32 variables, as
    // for any SGPR) and this lane's bit (a bool). A write in one form updates the other: a mask
    // write sets the dwords to (bit, 0); a dword write sets the bit to (lo | hi) != 0. Compilers
    // (Sony's among them) also use vcc_lo/vcc_hi and mask pairs as plain scalar registers.
    bool hasMaskForm(uint32_t pair) { return pair == kVcc || maskPairs.count(pair); }
    uint32_t maskBool(uint32_t pair) { return pair == kVcc ? vccV() : mVar(pair); }

    // A lane mask operand as this lane's bit.
    ValueId readMask(const Operand& o) {
        if (o.kind == OperandKind::Const || o.kind == OperandKind::Literal) {
            if (o.value == 0) return b->constBool(false);
            if (int32_t(o.value) == -1) return b->constBool(true);
            note("a lane mask constant other than 0 / -1 is read as all lanes");
            return b->constBool(true);
        }
        if (o.code == kExec) return b->loadVar(execV());
        if (hasMaskForm(o.code)) return b->loadVar(maskBool(o.code));
        if (o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp) {
            ValueId lo = b->loadVar(sVar(o.code)), hi = b->loadVar(sVar(o.code + 1));
            return nz(bin(V::Op::Or, lo, hi, V::kU32));
        }
        error("unsupported lane mask operand");
        return b->constBool(false);
    }
    void writeMask(const Operand& o, ValueId v) {
        if (o.code == kExec) {
            b->storeVar(execV(), v);
            return;
        }
        if (o.code == kVcc || o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp) {
            if (o.code != kVcc) maskPairs.insert(o.code);  // (findMasks has them all already)
            b->storeVar(maskBool(o.code), v);
            b->storeVar(sVar(o.code), b2u(v));
            b->storeVar(sVar(o.code + 1), u(0));
            return;
        }
        error("unsupported lane mask destination");
    }

    // A 32-bit scalar source (SGPR, special, inline constant, literal).
    ValueId readS(const Operand& o, unsigned half = 0) {
        switch (o.kind) {
            case OperandKind::Const:
            case OperandKind::Literal:
                if (half) return u(o.kind == OperandKind::Const && int32_t(o.value) < 0 && !o.isFloat ? 0xffffffffu : 0);
                return u(o.value);
            case OperandKind::Imm: return u(o.value);
            case OperandKind::Vgpr: return b->loadVar(vVar(o.index + half));
            case OperandKind::Sgpr:
            case OperandKind::Ttmp: return b->loadVar(sVar(o.code + half));
            case OperandKind::Special: {
                uint32_t code = o.code + half;
                switch (code) {
                    case kM0: return b->loadVar(m0V());
                    case kVcc:
                    case kVcc + 1: return b->loadVar(sVar(code));
                    case kExec: return b2u(b->loadVar(execV()));
                    case kExec + 1: return u(0);
                    case 251: return b2u(notB(b->loadVar(vccV())));
                    case 252: return b2u(notB(b->loadVar(execV())));
                    case 253: return b2u(b->loadVar(sccV()));
                    default: error("unsupported special source (flat_scratch, tba/tma or lds_direct)"); return u(0);
                }
            }
            default: return u(0);
        }
    }
    ValueId readV(const Operand& o) { return readS(o); }

    void writeS(const Operand& o, ValueId v, unsigned half = 0) {
        uint32_t code = o.code + half;
        bool sgpr = o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp;
        if (sgpr || code == kVcc || code == kVcc + 1) {
            b->storeVar(sVar(code), v);
            uint32_t pair = code & ~1u;
            if (hasMaskForm(pair)) {
                ValueId lo = b->loadVar(sVar(pair)), hi = b->loadVar(sVar(pair + 1));
                b->storeVar(maskBool(pair), nz(bin(V::Op::Or, lo, hi, V::kU32)));
            }
            return;
        }
        if (code == kM0) {
            b->storeVar(m0V(), v);
            return;
        }
        if (code == kExec) {
            note("a 32-bit write to exec_lo becomes this lane's bit");
            writesExec = true;
            b->storeVar(execV(), nz(v));
            return;
        }
        if (code == kExec + 1) return;  // lanes 32-63
        error("unsupported scalar destination");
    }

    void writeV(unsigned index, ValueId v) {
        if (writesExec) v = sel(b->loadVar(execV()), v, b->loadVar(vVar(index)), V::kU32);
        b->storeVar(vVar(index), v);
    }

    void setScc(ValueId c) { b->storeVar(sccV(), c); }

    // ---- Resources -----------------------------------------------------------------------------

    uint32_t resource(V::ResourceKind kind, const DescriptorSource& s, uint8_t dwords, V::Access access = V::Access::ReadWrite,
                      const std::string& prefix = "r") {
        auto key = std::make_pair(int(kind) * 256 + dwords, s);
        auto it = slots.find(key);
        if (it != slots.end()) {
            V::Resource& r = m.resources[it->second];
            if (r.access != access && kind == V::ResourceKind::Buffer) r.access = V::Access::ReadWrite;
            return it->second;
        }
        V::Resource r;
        r.kind = kind;
        r.index = perKindCount[int(kind)]++;
        r.name = prefix + std::to_string(r.index);
        r.access = access;
        if (kind == V::ResourceKind::Texture || kind == V::ResourceKind::StorageTexture) {
            if (in.describe)
                if (auto d = in.describe(s)) {
                    r.dim = d->dim;
                    r.elem = d->elem;
                    r.depth = d->depth && kind == V::ResourceKind::Texture;
                }
        }
        uint32_t slot = uint32_t(m.resources.size());
        m.resources.push_back(r);
        slots[key] = slot;
        out.bindings.push_back({kind, slot, s, dwords});
        return slot;
    }

    uint32_t userData() {
        if (!userSlot) {
            V::Resource r;
            r.kind = V::ResourceKind::ConstantBuffer;
            r.index = perKindCount[int(V::ResourceKind::ConstantBuffer)]++;
            r.name = "user_data";
            r.access = V::Access::Read;
            userSlot = uint32_t(m.resources.size());
            m.resources.push_back(r);
            out.userDataSlot = userSlot;
        }
        return *userSlot;
    }

    const DescriptorSource* sourceOf(const std::unordered_map<size_t, DescriptorSource>& map, size_t i, const char* what) {
        auto it = map.find(i);
        if (it != map.end()) return &it->second;
        error(std::string("cannot trace the ") + what + " descriptor of " + std::string(mnemonic(prog[i].in.op)) +
              " back to user data");
        return nullptr;
    }

    // ---- Instruction semantics ------------------------------------------------------------------

    // Float source i with VOP3 input modifiers.
    ValueId srcF(const Instruction& x, unsigned i) {
        ValueId v = asF(readV(x.src[i]));
        if (x.abs >> i & 1) v = b->unary(V::Op::Abs, v, V::kF32);
        if (x.neg >> i & 1) v = b->unary(V::Op::Neg, v, V::kF32);
        return v;
    }
    ValueId srcU(const Instruction& x, unsigned i) { return readV(x.src[i]); }
    ValueId srcI(const Instruction& x, unsigned i) { return asI(readV(x.src[i])); }

    void resultF(const Instruction& x, ValueId v) {
        if (x.omod == 1) v = bin(V::Op::Mul, v, f(2.0f), V::kF32);
        if (x.omod == 2) v = bin(V::Op::Mul, v, f(4.0f), V::kF32);
        if (x.omod == 3) v = bin(V::Op::Mul, v, f(0.5f), V::kF32);
        if (x.clamp) v = b->unary(V::Op::Saturate, v, V::kF32);
        writeV(x.dst.index, bit(v, V::kU32));
    }
    void resultU(const Instruction& x, ValueId v) { writeV(x.dst.index, v); }
    void resultI(const Instruction& x, ValueId v) { writeV(x.dst.index, bit(v, V::kU32)); }

    ValueId shiftAmount(ValueId v) { return bin(V::Op::And, v, u(31), V::kU32); }

    // High 32 bits of a 32x32 multiply, from 16-bit halves.
    ValueId mulHiU(ValueId a, ValueId c) {
        ValueId a0 = bin(V::Op::And, a, u(0xffff), V::kU32), a1 = bin(V::Op::Shr, a, u(16), V::kU32);
        ValueId c0 = bin(V::Op::And, c, u(0xffff), V::kU32), c1 = bin(V::Op::Shr, c, u(16), V::kU32);
        ValueId p00 = bin(V::Op::Mul, a0, c0, V::kU32), p01 = bin(V::Op::Mul, a0, c1, V::kU32);
        ValueId p10 = bin(V::Op::Mul, a1, c0, V::kU32), p11 = bin(V::Op::Mul, a1, c1, V::kU32);
        ValueId mid = bin(V::Op::Add, bin(V::Op::Shr, p00, u(16), V::kU32), bin(V::Op::And, p01, u(0xffff), V::kU32), V::kU32);
        mid = bin(V::Op::Add, mid, bin(V::Op::And, p10, u(0xffff), V::kU32), V::kU32);
        ValueId hi = bin(V::Op::Add, p11, bin(V::Op::Shr, p01, u(16), V::kU32), V::kU32);
        hi = bin(V::Op::Add, hi, bin(V::Op::Shr, p10, u(16), V::kU32), V::kU32);
        return bin(V::Op::Add, hi, bin(V::Op::Shr, mid, u(16), V::kU32), V::kU32);
    }
    ValueId mulHiI(ValueId a, ValueId c) {
        // signed hi = unsigned hi - (a < 0 ? c : 0) - (c < 0 ? a : 0)
        ValueId hi = mulHiU(a, c);
        ValueId an = bin(V::Op::Lt, asI(a), b->constI32(0), V::kBool), cn = bin(V::Op::Lt, asI(c), b->constI32(0), V::kBool);
        hi = bin(V::Op::Sub, hi, sel(an, c, u(0), V::kU32), V::kU32);
        return bin(V::Op::Sub, hi, sel(cn, a, u(0), V::kU32), V::kU32);
    }

    // f32 -> f16 bits in the low half of a u32, and back.
    ValueId halfBits(ValueId fv) {
        ValueId h = b->unary(V::Op::Convert, fv, V::kF16);
        ValueId pair = b->construct(std::array<ValueId, 2>{h, b->constant(V::kF16, {0})}, V::vec(V::kF16, 2));
        return bit(pair, V::kU32);
    }
    // f32 -> f16 bits rounded toward zero: round to nearest, then one step back toward zero when
    // that went past the value (also turns an overflow to infinity into the largest finite).
    ValueId halfRtz(ValueId fv) {
        auto U = V::kU32;
        ValueId h = halfBits(fv);
        ValueId back = halfToF(h, 0);
        ValueId past = bin(V::Op::Gt, b->unary(V::Op::Abs, back, V::kF32), b->unary(V::Op::Abs, fv, V::kF32), V::kBool);
        past = bin(V::Op::And, past, notB(b->unary(V::Op::IsNan, fv, V::kBool)), V::kBool);
        return sel(past, bin(V::Op::Sub, h, u(1), U), h, U);
    }
    ValueId halfToF(ValueId bits, unsigned hi) {
        ValueId pair = bit(bits, V::vec(V::kF16, 2));
        return b->unary(V::Op::Convert, b->extract(pair, hi, V::kF16), V::kF32);
    }

    // GCN bit-field extract, defined for every offset/width (offset + width may pass 32, where
    // Metal's extract_bits is undefined): (v >> off) masked to `width` bits, sign-extended from
    // bit width-1 when signed; width 0 gives 0. Vector ops pass src[4:0]; widths of 32+ keep all.
    ValueId bfeWide(ValueId v, ValueId off, ValueId width, bool sign) {
        auto U = V::kU32;
        // (The signed form shifts arithmetically: past bit 31 come copies of the sign.)
        ValueId shifted = sign ? bit(bin(V::Op::Shr, asI(v), asI(off), V::kI32), U) : bin(V::Op::Shr, v, off, U);
        ValueId full = bin(V::Op::Ge, width, u(32), V::kBool);
        ValueId w = bin(V::Op::And, width, u(31), U);
        ValueId mask = bin(V::Op::Sub, bin(V::Op::Shl, u(1), w, U), u(1), U);
        ValueId field = sel(full, shifted, bin(V::Op::And, shifted, mask, U), U);
        if (sign) {
            ValueId up = bin(V::Op::And, bin(V::Op::Sub, u(32), w, U), u(31), U);
            ValueId ext = bit(bin(V::Op::Shr, asI(bin(V::Op::Shl, field, up, U)), asI(up), V::kI32), U);
            field = sel(full, field, ext, U);
        }
        return sel(bin(V::Op::Eq, width, u(0), V::kBool), u(0), field, U);
    }
    // f32 -> i32 as the hardware converts: truncate, saturate, NaN gives 0.
    ValueId toI32(ValueId v) {
        auto F = V::kF32;
        ValueId big = bin(V::Op::Ge, v, f(2147483648.0f), V::kBool);
        ValueId c = bit(b->unary(V::Op::Convert, b->ternary(V::Op::Clamp, v, f(-2147483648.0f), f(2147483520.0f), F), V::kI32), V::kU32);
        c = sel(big, u(0x7fffffffu), c, V::kU32);
        return sel(b->unary(V::Op::IsNan, v, V::kBool), u(0), c, V::kU32);
    }
    ValueId bfe(ValueId v, ValueId off, ValueId cnt, bool sign) {
        return bfeWide(v, bin(V::Op::And, off, u(31), V::kU32), bin(V::Op::And, cnt, u(31), V::kU32), sign);
    }

    ValueId cmpF(G op, ValueId a, ValueId c) {
        std::string_view n = mnemonic(op);
        // v_cmp[x|s|sx]_<cond>_f32
        size_t p = n.find('_', 2);  // end of v_cmp / v_cmpx / v_cmps / v_cmpsx
        std::string_view cond = n.substr(p + 1, n.rfind('_') - p - 1);
        ValueId nan = bin(V::Op::Or, b->unary(V::Op::IsNan, a, V::kBool), b->unary(V::Op::IsNan, c, V::kBool), V::kBool);
        auto ord = [&](V::Op o) { return bin(o, a, c, V::kBool); };
        auto unord = [&](V::Op o) { return bin(V::Op::Or, nan, notB(bin(o, a, c, V::kBool)), V::kBool); };
        if (cond == "f") return b->constBool(false);
        if (cond == "tru") return b->constBool(true);
        if (cond == "lt") return ord(V::Op::Lt);
        if (cond == "eq") return ord(V::Op::Eq);
        if (cond == "le") return ord(V::Op::Le);
        if (cond == "gt") return ord(V::Op::Gt);
        if (cond == "lg") return bin(V::Op::And, notB(nan), ord(V::Op::Ne), V::kBool);
        if (cond == "ge") return ord(V::Op::Ge);
        if (cond == "o") return notB(nan);
        if (cond == "u") return nan;
        if (cond == "nge") return unord(V::Op::Ge);
        if (cond == "nlg") return bin(V::Op::Or, nan, ord(V::Op::Eq), V::kBool);
        if (cond == "ngt") return unord(V::Op::Gt);
        if (cond == "nle") return unord(V::Op::Le);
        if (cond == "neq") return bin(V::Op::Or, nan, ord(V::Op::Ne), V::kBool);
        if (cond == "nlt") return unord(V::Op::Lt);
        error("unknown float compare " + std::string(n));
        return b->constBool(false);
    }
    ValueId cmpInt(G op, ValueId a, ValueId c, bool sign) {
        std::string_view n = mnemonic(op);
        size_t p = n.find('_', 2);  // end of v_cmp / v_cmpx / v_cmps / v_cmpsx
        std::string_view cond = n.substr(p + 1, n.rfind('_') - p - 1);
        if (sign) a = asI(a), c = asI(c);
        V::Type t = sign ? V::kI32 : V::kU32;
        (void)t;
        if (cond == "f") return b->constBool(false);
        if (cond == "t") return b->constBool(true);
        if (cond == "lt") return bin(V::Op::Lt, a, c, V::kBool);
        if (cond == "eq") return bin(V::Op::Eq, a, c, V::kBool);
        if (cond == "le") return bin(V::Op::Le, a, c, V::kBool);
        if (cond == "gt") return bin(V::Op::Gt, a, c, V::kBool);
        if (cond == "ne" || cond == "lg") return bin(V::Op::Ne, a, c, V::kBool);
        if (cond == "ge") return bin(V::Op::Ge, a, c, V::kBool);
        error("unknown integer compare " + std::string(n));
        return b->constBool(false);
    }
    // 64-bit integer compares. Lane masks (VCC, EXEC, mask pairs, compared with each other or with
    // 0 / -1) compare as this lane's bit, as everything else per lane; register pairs compare by
    // dwords.
    ValueId cmp64(G op, const Instruction& x, bool sign) {
        std::string_view n = mnemonic(op);
        size_t p = n.find('_', 2);
        std::string_view cond = n.substr(p + 1, n.rfind('_') - p - 1);
        auto maskLike = [&](const Operand& o) {
            return (o.kind == OperandKind::Special && (o.code == kVcc || o.code == kExec)) ||
                   ((o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp) && maskPairs.count(o.code));
        };
        auto constMask = [&](const Operand& o) {
            return o.kind == OperandKind::Const && (o.value == 0 || int32_t(o.value) == -1);
        };
        const Operand &a = x.src[0], &c = x.src[1];
        if ((maskLike(a) || maskLike(c)) && (maskLike(a) || constMask(a)) && (maskLike(c) || constMask(c))) {
            ValueId ba = readMask(a), bc = readMask(c);
            if (cond == "eq") return bin(V::Op::Eq, ba, bc, V::kBool);
            if (cond == "ne") return bin(V::Op::Ne, ba, bc, V::kBool);
            if (cond == "f") return b->constBool(false);
            if (cond == "t") return b->constBool(true);
            note("an ordered 64-bit compare of lane masks uses this lane's bits");
            ValueId ua = b2u(ba), uc = b2u(bc);
            return cmpInt(op, ua, uc, false);
        }
        ValueId alo = readS(a, 0), ahi = readS(a, 1), clo = readS(c, 0), chi = readS(c, 1);
        ValueId hiEq = bin(V::Op::Eq, ahi, chi, V::kBool);
        if (cond == "f") return b->constBool(false);
        if (cond == "t") return b->constBool(true);
        if (cond == "eq") return bin(V::Op::And, hiEq, bin(V::Op::Eq, alo, clo, V::kBool), V::kBool);
        if (cond == "ne") return notB(bin(V::Op::And, hiEq, bin(V::Op::Eq, alo, clo, V::kBool), V::kBool));
        auto hiCmp = [&](V::Op o) { return sign ? bin(o, asI(ahi), asI(chi), V::kBool) : bin(o, ahi, chi, V::kBool); };
        auto lex = [&](V::Op strict, V::Op loOp) {
            return bin(V::Op::Or, hiCmp(strict), bin(V::Op::And, hiEq, bin(loOp, alo, clo, V::kBool), V::kBool), V::kBool);
        };
        if (cond == "lt") return lex(V::Op::Lt, V::Op::Lt);
        if (cond == "le") return lex(V::Op::Lt, V::Op::Le);
        if (cond == "gt") return lex(V::Op::Gt, V::Op::Gt);
        if (cond == "ge") return lex(V::Op::Gt, V::Op::Ge);
        error("unknown 64-bit compare " + std::string(n));
        return b->constBool(false);
    }

    // v_cmp_class_f32: bit per class (0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0,
    // 7 +denormal, 8 +normal, 9 +inf).
    ValueId cmpClass(ValueId fv, ValueId mask) {
        ValueId bits = bit(fv, V::kU32);
        ValueId neg = nz(bin(V::Op::And, bits, u(0x80000000u), V::kU32));
        ValueId mag = bin(V::Op::And, bits, u(0x7fffffffu), V::kU32);
        ValueId isNan = bin(V::Op::Gt, mag, u(0x7f800000u), V::kBool);
        ValueId quiet = nz(bin(V::Op::And, bits, u(0x00400000u), V::kU32));
        ValueId isInf = bin(V::Op::Eq, mag, u(0x7f800000u), V::kBool);
        ValueId isZero = bin(V::Op::Eq, mag, u(0), V::kBool);
        ValueId isDen = bin(V::Op::And, bin(V::Op::Lt, mag, u(0x00800000u), V::kBool), notB(isZero), V::kBool);
        ValueId isNorm = bin(V::Op::And, bin(V::Op::Ge, mag, u(0x00800000u), V::kBool),
                             bin(V::Op::Lt, mag, u(0x7f800000u), V::kBool), V::kBool);
        auto and3 = [&](ValueId p, ValueId q) { return bin(V::Op::And, p, q, V::kBool); };
        ValueId cls[10] = {and3(isNan, notB(quiet)), and3(isNan, quiet), and3(isInf, neg), and3(isNorm, neg),
                           and3(isDen, neg), and3(isZero, neg), and3(isZero, notB(neg)), and3(isDen, notB(neg)),
                           and3(isNorm, notB(neg)), and3(isInf, notB(neg))};
        ValueId r = b->constBool(false);
        for (unsigned i = 0; i < 10; ++i)
            r = bin(V::Op::Or, r, and3(cls[i], nz(bin(V::Op::And, mask, u(1u << i), V::kU32))), V::kBool);
        return r;
    }

    void liftVop(size_t idx) {
        const Instruction& x = prog[idx].in;
        G op = x.op;
        V::Op Add = V::Op::Add, Sub = V::Op::Sub, Mul = V::Op::Mul;
        auto F = V::kF32;
        auto U = V::kU32;
        auto I = V::kI32;
        std::string_view name = mnemonic(op);
        if (name.starts_with("v_cmp")) {
            ValueId r;
            if (name.find("_class_") != std::string_view::npos) {
                if (name.ends_with("f64")) return error("64-bit floats are not supported (" + std::string(name) + ")");
                r = cmpClass(srcF(x, 0), srcU(x, 1));
            } else if (name.ends_with("_f32")) {
                r = cmpF(op, srcF(x, 0), srcF(x, 1));
            } else if (name.ends_with("_i32") || name.ends_with("_u32")) {
                r = cmpInt(op, srcU(x, 0), srcU(x, 1), name.ends_with("_i32"));
            } else if (name.ends_with("_f64")) {
                return error("64-bit float compares are not supported (" + std::string(name) + ")");
            } else {
                r = cmp64(op, x, name.ends_with("_i64"));
            }
            // Inactive lanes compare as 0 into the mask (and, for cmpx, into EXEC).
            ValueId exec = b->loadVar(execV());
            r = bin(V::Op::And, r, exec, V::kBool);
            writeMask(x.sdst, r);
            if (name.starts_with("v_cmpx") || name.starts_with("v_cmpsx")) {
                writesExec = true;
                b->storeVar(execV(), r);
            }
            return;
        }
        switch (op) {
            case G::v_nop: return;
            case G::v_mov_b32:
                if (x.abs || x.neg) return resultF(x, srcF(x, 0));  // a float move with |x| / -x
                return resultU(x, srcU(x, 0));
            case G::v_readfirstlane_b32:
            case G::v_readlane_b32:
                if (op == G::v_readlane_b32) note("v_readlane_b32 reads this lane's own value");
                writeS(x.dst, srcU(x, 0));
                return;
            case G::v_writelane_b32:
                note("v_writelane_b32 writes every lane");
                return resultU(x, srcU(x, 0));
            case G::v_cndmask_b32: {
                ValueId c = readMask(x.src[2]);
                if (x.abs || x.neg) return resultF(x, sel(c, srcF(x, 1), srcF(x, 0), F));
                return resultU(x, sel(c, srcU(x, 1), srcU(x, 0), U));
            }
            case G::v_add_f32: return resultF(x, bin(Add, srcF(x, 0), srcF(x, 1), F));
            case G::v_sub_f32: return resultF(x, bin(Sub, srcF(x, 0), srcF(x, 1), F));
            case G::v_subrev_f32: return resultF(x, bin(Sub, srcF(x, 1), srcF(x, 0), F));
            case G::v_mul_f32: return resultF(x, bin(Mul, srcF(x, 0), srcF(x, 1), F));
            case G::v_mul_legacy_f32: {
                ValueId a = srcF(x, 0), c = srcF(x, 1);
                ValueId zero = bin(V::Op::Or, bin(V::Op::Eq, a, f(0), V::kBool), bin(V::Op::Eq, c, f(0), V::kBool), V::kBool);
                return resultF(x, sel(zero, f(0), bin(Mul, a, c, F), F));
            }
            case G::v_mac_legacy_f32:
            case G::v_mac_f32: {
                ValueId acc = asF(b->loadVar(vVar(x.dst.index)));
                ValueId a = srcF(x, 0), c = srcF(x, 1);
                ValueId p = bin(Mul, a, c, F);
                if (op == G::v_mac_legacy_f32) {
                    ValueId zero = bin(V::Op::Or, bin(V::Op::Eq, a, f(0), V::kBool), bin(V::Op::Eq, c, f(0), V::kBool), V::kBool);
                    p = sel(zero, f(0), p, F);
                }
                return resultF(x, bin(Add, p, acc, F));
            }
            case G::v_mad_f32:
            case G::v_mad_legacy_f32: {
                ValueId a = srcF(x, 0), c = srcF(x, 1), d = srcF(x, 2);
                ValueId p = bin(Mul, a, c, F);
                if (op == G::v_mad_legacy_f32) {
                    ValueId zero = bin(V::Op::Or, bin(V::Op::Eq, a, f(0), V::kBool), bin(V::Op::Eq, c, f(0), V::kBool), V::kBool);
                    p = sel(zero, f(0), p, F);
                }
                return resultF(x, bin(Add, p, d, F));
            }
            case G::v_fma_f32: return resultF(x, b->ternary(V::Op::Fma, srcF(x, 0), srcF(x, 1), srcF(x, 2), F));
            case G::v_madmk_f32: return resultF(x, bin(Add, bin(Mul, srcF(x, 0), srcF(x, 1), F), srcF(x, 2), F));
            case G::v_madak_f32: return resultF(x, bin(Add, bin(Mul, srcF(x, 0), srcF(x, 1), F), srcF(x, 2), F));
            case G::v_min_f32: return resultF(x, bin(V::Op::Min, srcF(x, 0), srcF(x, 1), F));
            case G::v_max_f32: return resultF(x, bin(V::Op::Max, srcF(x, 0), srcF(x, 1), F));
            case G::v_min_legacy_f32:
            case G::v_max_legacy_f32: {
                // D3D9 semantics: (a < b) ? a : b and (a >= b) ? a : b (a NaN anywhere gives b).
                ValueId a = srcF(x, 0), c = srcF(x, 1);
                V::Op cmp = op == G::v_min_legacy_f32 ? V::Op::Lt : V::Op::Ge;
                return resultF(x, sel(bin(cmp, a, c, V::kBool), a, c, F));
            }
            case G::v_min3_f32: return resultF(x, bin(V::Op::Min, bin(V::Op::Min, srcF(x, 0), srcF(x, 1), F), srcF(x, 2), F));
            case G::v_max3_f32: return resultF(x, bin(V::Op::Max, bin(V::Op::Max, srcF(x, 0), srcF(x, 1), F), srcF(x, 2), F));
            case G::v_med3_f32: {
                // With a NaN operand the hardware returns min3.
                ValueId a = srcF(x, 0), c = srcF(x, 1), d = srcF(x, 2);
                ValueId lo = bin(V::Op::Min, a, c, F), hi = bin(V::Op::Max, a, c, F);
                ValueId med = bin(V::Op::Max, lo, bin(V::Op::Min, hi, d, F), F);
                ValueId nan = bin(V::Op::Or, b->unary(V::Op::IsNan, a, V::kBool),
                                  bin(V::Op::Or, b->unary(V::Op::IsNan, c, V::kBool), b->unary(V::Op::IsNan, d, V::kBool), V::kBool), V::kBool);
                return resultF(x, sel(nan, bin(V::Op::Min, lo, d, F), med, F));
            }
            case G::v_min_i32: return resultI(x, bin(V::Op::Min, srcI(x, 0), srcI(x, 1), I));
            case G::v_max_i32: return resultI(x, bin(V::Op::Max, srcI(x, 0), srcI(x, 1), I));
            case G::v_min_u32: return resultU(x, bin(V::Op::Min, srcU(x, 0), srcU(x, 1), U));
            case G::v_max_u32: return resultU(x, bin(V::Op::Max, srcU(x, 0), srcU(x, 1), U));
            case G::v_min3_i32: return resultI(x, bin(V::Op::Min, bin(V::Op::Min, srcI(x, 0), srcI(x, 1), I), srcI(x, 2), I));
            case G::v_max3_i32: return resultI(x, bin(V::Op::Max, bin(V::Op::Max, srcI(x, 0), srcI(x, 1), I), srcI(x, 2), I));
            case G::v_min3_u32: return resultU(x, bin(V::Op::Min, bin(V::Op::Min, srcU(x, 0), srcU(x, 1), U), srcU(x, 2), U));
            case G::v_max3_u32: return resultU(x, bin(V::Op::Max, bin(V::Op::Max, srcU(x, 0), srcU(x, 1), U), srcU(x, 2), U));
            case G::v_med3_i32: {
                ValueId a = srcI(x, 0), c = srcI(x, 1), d = srcI(x, 2);
                return resultI(x, bin(V::Op::Max, bin(V::Op::Min, a, c, I), bin(V::Op::Min, bin(V::Op::Max, a, c, I), d, I), I));
            }
            case G::v_med3_u32: {
                ValueId a = srcU(x, 0), c = srcU(x, 1), d = srcU(x, 2);
                return resultU(x, bin(V::Op::Max, bin(V::Op::Min, a, c, U), bin(V::Op::Min, bin(V::Op::Max, a, c, U), d, U), U));
            }
            case G::v_mul_i32_i24: {
                ValueId a = bfe(srcU(x, 0), u(0), u(24), true), c = bfe(srcU(x, 1), u(0), u(24), true);
                return resultU(x, bin(Mul, a, c, U));
            }
            case G::v_mul_hi_i32_i24: {
                ValueId a = bfe(srcU(x, 0), u(0), u(24), true), c = bfe(srcU(x, 1), u(0), u(24), true);
                return resultU(x, mulHiI(a, c));
            }
            case G::v_mul_u32_u24: {
                ValueId a = bin(V::Op::And, srcU(x, 0), u(0xffffff), U), c = bin(V::Op::And, srcU(x, 1), u(0xffffff), U);
                return resultU(x, bin(Mul, a, c, U));
            }
            case G::v_mul_hi_u32_u24: {
                ValueId a = bin(V::Op::And, srcU(x, 0), u(0xffffff), U), c = bin(V::Op::And, srcU(x, 1), u(0xffffff), U);
                return resultU(x, mulHiU(a, c));
            }
            case G::v_mad_i32_i24: {
                ValueId a = bfe(srcU(x, 0), u(0), u(24), true), c = bfe(srcU(x, 1), u(0), u(24), true);
                return resultU(x, bin(Add, bin(Mul, a, c, U), srcU(x, 2), U));
            }
            case G::v_mad_u32_u24: {
                ValueId a = bin(V::Op::And, srcU(x, 0), u(0xffffff), U), c = bin(V::Op::And, srcU(x, 1), u(0xffffff), U);
                return resultU(x, bin(Add, bin(Mul, a, c, U), srcU(x, 2), U));
            }
            case G::v_mul_lo_u32:
            case G::v_mul_lo_i32: return resultU(x, bin(Mul, srcU(x, 0), srcU(x, 1), U));
            case G::v_mul_hi_u32: return resultU(x, mulHiU(srcU(x, 0), srcU(x, 1)));
            case G::v_mul_hi_i32: return resultU(x, mulHiI(srcU(x, 0), srcU(x, 1)));
            case G::v_lshr_b32: return resultU(x, bin(V::Op::Shr, srcU(x, 0), shiftAmount(srcU(x, 1)), U));
            case G::v_lshrrev_b32: return resultU(x, bin(V::Op::Shr, srcU(x, 1), shiftAmount(srcU(x, 0)), U));
            case G::v_ashr_i32:
                return resultI(x, bin(V::Op::Shr, srcI(x, 0), asI(shiftAmount(srcU(x, 1))), I));
            case G::v_ashrrev_i32:
                return resultI(x, bin(V::Op::Shr, srcI(x, 1), asI(shiftAmount(srcU(x, 0))), I));
            case G::v_lshl_b32: return resultU(x, bin(V::Op::Shl, srcU(x, 0), shiftAmount(srcU(x, 1)), U));
            case G::v_lshlrev_b32: return resultU(x, bin(V::Op::Shl, srcU(x, 1), shiftAmount(srcU(x, 0)), U));
            case G::v_and_b32: return resultU(x, bin(V::Op::And, srcU(x, 0), srcU(x, 1), U));
            case G::v_or_b32: return resultU(x, bin(V::Op::Or, srcU(x, 0), srcU(x, 1), U));
            case G::v_xor_b32: return resultU(x, bin(V::Op::Xor, srcU(x, 0), srcU(x, 1), U));
            case G::v_not_b32: return resultU(x, b->unary(V::Op::Not, srcU(x, 0), U));
            case G::v_bfrev_b32: return resultU(x, b->unary(V::Op::BitReverse, srcU(x, 0), U));
            case G::v_bcnt_u32_b32: return resultU(x, bin(Add, b->unary(V::Op::BitCount, srcU(x, 0), U), srcU(x, 1), U));
            case G::v_ffbl_b32: {
                ValueId v = srcU(x, 0);
                return resultU(x, sel(nz(v), b->unary(V::Op::FindLsb, v, U), u(0xffffffffu), U));
            }
            case G::v_ffbh_u32: {
                // Leading zeros from the MSB; -1 for 0.
                ValueId v = srcU(x, 0);
                ValueId msb = b->unary(V::Op::FindMsb, v, U);
                return resultU(x, sel(nz(v), bin(Sub, u(31), msb, U), u(0xffffffffu), U));
            }
            case G::v_ffbh_i32: {
                // Bits equal to the sign from the MSB; -1 when all bits are equal.
                ValueId v = srcU(x, 0);
                ValueId flipped = sel(bin(V::Op::Lt, asI(v), b->constI32(0), V::kBool), b->unary(V::Op::Not, v, U), v, U);
                ValueId msb = b->unary(V::Op::FindMsb, flipped, U);
                return resultU(x, sel(nz(flipped), bin(Sub, u(31), msb, U), u(0xffffffffu), U));
            }
            case G::v_bfm_b32: {
                // ((1 << width) - 1) << offset, width/offset in [4:0].
                ValueId w = shiftAmount(srcU(x, 0)), o = shiftAmount(srcU(x, 1));
                ValueId mask = bin(Sub, bin(V::Op::Shl, u(1), w, U), u(1), U);
                return resultU(x, bin(V::Op::Shl, mask, o, U));
            }
            case G::v_bfe_u32: return resultU(x, bfe(srcU(x, 0), srcU(x, 1), srcU(x, 2), false));
            case G::v_bfe_i32: return resultU(x, bfe(srcU(x, 0), srcU(x, 1), srcU(x, 2), true));
            case G::v_bfi_b32: {
                ValueId mask = srcU(x, 0);
                return resultU(x, bin(V::Op::Or, bin(V::Op::And, mask, srcU(x, 1), U),
                                      bin(V::Op::And, b->unary(V::Op::Not, mask, U), srcU(x, 2), U), U));
            }
            case G::v_alignbit_b32:
            case G::v_alignbyte_b32: {
                // ({src0, src1} >> shift)[31:0]
                ValueId hi = srcU(x, 0), lo = srcU(x, 1), sh = srcU(x, 2);
                sh = op == G::v_alignbyte_b32 ? bin(V::Op::Shl, bin(V::Op::And, sh, u(3), U), u(3), U) : shiftAmount(sh);
                ValueId r = bin(V::Op::Or, bin(V::Op::Shr, lo, sh, U),
                                bin(V::Op::Shl, hi, bin(V::Op::And, bin(Sub, u(32), sh, U), u(31), U), U), U);
                return resultU(x, sel(nz(sh), r, lo, U));
            }
            case G::v_add_i32:
            case G::v_sub_i32:
            case G::v_subrev_i32: {
                ValueId a = srcU(x, 0), c = srcU(x, 1);
                if (op == G::v_subrev_i32) std::swap(a, c);
                ValueId r = bin(op == G::v_add_i32 ? Add : Sub, a, c, U);
                ValueId carry = op == G::v_add_i32 ? bin(V::Op::Lt, r, a, V::kBool) : bin(V::Op::Lt, a, c, V::kBool);
                writeMask(x.sdst, bin(V::Op::And, carry, b->loadVar(execV()), V::kBool));
                return resultU(x, r);
            }
            case G::v_addc_u32:
            case G::v_subb_u32:
            case G::v_subbrev_u32: {
                ValueId a = srcU(x, 0), c = srcU(x, 1), cin = readMask(x.src[2]);
                if (op == G::v_subbrev_u32) std::swap(a, c);
                ValueId ci = b2u(cin);
                ValueId r, carry;
                if (op == G::v_addc_u32) {
                    ValueId s = bin(Add, a, c, U);
                    r = bin(Add, s, ci, U);
                    carry = bin(V::Op::Or, bin(V::Op::Lt, s, a, V::kBool), bin(V::Op::Lt, r, s, V::kBool), V::kBool);
                } else {
                    ValueId s = bin(Sub, a, c, U);
                    r = bin(Sub, s, ci, U);
                    carry = bin(V::Op::Or, bin(V::Op::Lt, a, c, V::kBool), bin(V::Op::Lt, s, ci, V::kBool), V::kBool);
                }
                writeMask(x.sdst, bin(V::Op::And, carry, b->loadVar(execV()), V::kBool));
                return resultU(x, r);
            }
            case G::v_ldexp_f32: {
                // a * 2^n, the power of two built from its bits ((e + 127) << 23: exact, where a fast
                // exp2 may not be). In steps of at most 2^127 / 2^-126, so no factor overflows (0 * inf) while the
                // result is representable.
                ValueId n = b->ternary(V::Op::Clamp, srcI(x, 1), b->constI32(-400), b->constI32(400), I);
                ValueId r = srcF(x, 0);
                for (int k = 0; k < 4; ++k) {
                    ValueId e = b->ternary(V::Op::Clamp, n, b->constI32(-126), b->constI32(127), I);
                    ValueId p = bin(V::Op::Shl, bit(bin(V::Op::Add, e, b->constI32(127), I), U), u(23), U);
                    r = bin(Mul, r, bit(p, F), F);
                    n = bin(V::Op::Sub, n, e, I);
                }
                return resultF(x, r);
            }
            case G::v_cvt_pkrtz_f16_f32: return resultU(x, bin(V::Op::Or, halfRtz(srcF(x, 0)), bin(V::Op::Shl, halfRtz(srcF(x, 1)), u(16), U), U));
            case G::v_cvt_pknorm_i16_f32:
            case G::v_cvt_pknorm_u16_f32: {
                bool s = op == G::v_cvt_pknorm_i16_f32;
                auto pack = [&](ValueId v) {
                    v = sel(b->unary(V::Op::IsNan, v, V::kBool), f(0), v, F);  // NaN packs as 0
                    ValueId c = s ? b->ternary(V::Op::Clamp, v, f(-1), f(1), F) : b->unary(V::Op::Saturate, v, F);
                    ValueId r = b->unary(V::Op::Round, bin(Mul, c, f(s ? 32767.0f : 65535.0f), F), F);
                    ValueId i = b->unary(V::Op::Convert, r, I);
                    return bin(V::Op::And, bit(i, U), u(0xffff), U);
                };
                return resultU(x, bin(V::Op::Or, pack(srcF(x, 0)), bin(V::Op::Shl, pack(srcF(x, 1)), u(16), U), U));
            }
            case G::v_cvt_pk_u16_u32:
            case G::v_cvt_pk_i16_i32: {
                // Saturated to 16 bits.
                auto sat = [&](unsigned i) {
                    if (op == G::v_cvt_pk_u16_u32) return bin(V::Op::Min, srcU(x, i), u(0xffff), U);
                    ValueId c = b->ternary(V::Op::Clamp, srcI(x, i), b->constI32(-32768), b->constI32(32767), I);
                    return bin(V::Op::And, bit(c, U), u(0xffff), U);
                };
                return resultU(x, bin(V::Op::Or, sat(0), bin(V::Op::Shl, sat(1), u(16), U), U));
            }
            case G::v_cvt_pk_u8_f32: {
                // byte src1[1:0] of src2 <- u8(src0)
                ValueId v = b->unary(V::Op::Convert, b->ternary(V::Op::Clamp, b->unary(V::Op::Round, srcF(x, 0), F), f(0), f(255), F), U);
                ValueId sh = bin(V::Op::Shl, bin(V::Op::And, srcU(x, 1), u(3), U), u(3), U);
                ValueId keep = b->unary(V::Op::Not, bin(V::Op::Shl, u(0xff), sh, U), U);
                return resultU(x, bin(V::Op::Or, bin(V::Op::And, srcU(x, 2), keep, U), bin(V::Op::Shl, v, sh, U), U));
            }
            case G::v_cvt_f32_i32: return resultF(x, b->unary(V::Op::Convert, srcI(x, 0), F));
            case G::v_cvt_f32_u32: return resultF(x, b->unary(V::Op::Convert, srcU(x, 0), F));
            case G::v_cvt_u32_f32: {
                // Saturating; NaN gives 0. (From 2^32 up it is 0xffffffff; the largest f32 below is 2^32 - 256.)
                ValueId v = srcF(x, 0);
                ValueId big = bin(V::Op::Ge, v, f(4294967296.0f), V::kBool);
                ValueId c = b->unary(V::Op::Convert, b->ternary(V::Op::Clamp, v, f(0), f(4294967040.0f), F), U);
                c = sel(big, u(0xffffffffu), c, U);
                return resultU(x, sel(b->unary(V::Op::IsNan, v, V::kBool), u(0), c, U));
            }
            case G::v_cvt_i32_f32: return resultU(x, toI32(srcF(x, 0)));
            case G::v_cvt_rpi_i32_f32: return resultU(x, toI32(b->unary(V::Op::Floor, bin(Add, srcF(x, 0), f(0.5f), F), F)));
            case G::v_cvt_flr_i32_f32: return resultU(x, toI32(b->unary(V::Op::Floor, srcF(x, 0), F)));
            case G::v_cvt_f16_f32: return resultU(x, halfBits(srcF(x, 0)));
            case G::v_cvt_f32_f16: {
                ValueId v = halfToF(srcU(x, 0), 0);
                if (x.abs & 1) v = b->unary(V::Op::Abs, v, F);
                if (x.neg & 1) v = b->unary(V::Op::Neg, v, F);
                return resultF(x, v);
            }
            case G::v_cvt_f32_ubyte0:
            case G::v_cvt_f32_ubyte1:
            case G::v_cvt_f32_ubyte2:
            case G::v_cvt_f32_ubyte3: {
                unsigned k = unsigned(op) - unsigned(G::v_cvt_f32_ubyte0);
                ValueId v = bin(V::Op::And, bin(V::Op::Shr, srcU(x, 0), u(8 * k), U), u(0xff), U);
                return resultF(x, b->unary(V::Op::Convert, v, F));
            }
            case G::v_cvt_off_f32_i4: {
                // 4-bit signed in [3:0] -> value / 16
                ValueId v = bfe(srcU(x, 0), u(0), u(4), true);
                return resultF(x, bin(Mul, b->unary(V::Op::Convert, asI(v), F), f(1.0f / 16.0f), F));
            }
            case G::v_fract_f32: return resultF(x, b->unary(V::Op::Fract, srcF(x, 0), F));
            case G::v_trunc_f32: return resultF(x, b->unary(V::Op::Trunc, srcF(x, 0), F));
            case G::v_ceil_f32: return resultF(x, b->unary(V::Op::Ceil, srcF(x, 0), F));
            case G::v_rndne_f32: return resultF(x, b->unary(V::Op::Round, srcF(x, 0), F));
            case G::v_floor_f32: return resultF(x, b->unary(V::Op::Floor, srcF(x, 0), F));
            case G::v_exp_f32:
            case G::v_exp_legacy_f32: return resultF(x, b->unary(V::Op::Exp2, srcF(x, 0), F));
            case G::v_log_f32:
            case G::v_log_legacy_f32: return resultF(x, b->unary(V::Op::Log2, srcF(x, 0), F));
            case G::v_rcp_f32:
            case G::v_rcp_iflag_f32:
            case G::v_rcp_legacy_f32: return resultF(x, b->unary(V::Op::Rcp, srcF(x, 0), F));
            case G::v_rsq_f32:
            case G::v_rsq_legacy_f32: return resultF(x, b->unary(V::Op::Rsq, srcF(x, 0), F));
            case G::v_log_clamp_f32:
            case G::v_rcp_clamp_f32:
            case G::v_rsq_clamp_f32: {
                // Infinite results become +-FLT_MAX.
                V::Op o = op == G::v_log_clamp_f32 ? V::Op::Log2 : op == G::v_rcp_clamp_f32 ? V::Op::Rcp : V::Op::Rsq;
                ValueId r = b->unary(o, srcF(x, 0), F);
                return resultF(x, b->ternary(V::Op::Clamp, r, f(-3.402823466e38f), f(3.402823466e38f), F));
            }
            case G::v_sqrt_f32: return resultF(x, b->unary(V::Op::Sqrt, srcF(x, 0), F));
            case G::v_sin_f32:
            case G::v_cos_f32: {
                // The argument is in revolutions.
                ValueId a = bin(Mul, srcF(x, 0), f(6.28318530717958647692f), F);
                return resultF(x, b->unary(op == G::v_sin_f32 ? V::Op::Sin : V::Op::Cos, a, F));
            }
            case G::v_frexp_mant_f32:
            case G::v_frexp_exp_i32_f32: {
                // v = mant * 2^exp, |mant| in [0.5, 1); 0, inf and NaN: exp 0, mant = v. Denormals
                // are normalised from their highest set bit.
                ValueId v = srcF(x, 0);
                ValueId bits = bit(v, U);
                ValueId e = bin(V::Op::And, bin(V::Op::Shr, bits, u(23), U), u(0xff), U);
                ValueId frac = bin(V::Op::And, bits, u(0x7fffff), U);
                ValueId zero = bin(V::Op::Eq, bin(V::Op::And, bits, u(0x7fffffffu), U), u(0), V::kBool);
                ValueId special = bin(V::Op::Or, zero, bin(V::Op::Eq, e, u(0xff), V::kBool), V::kBool);
                ValueId denorm = bin(V::Op::And, bin(V::Op::Eq, e, u(0), V::kBool), notB(zero), V::kBool);
                ValueId msb = b->unary(V::Op::FindMsb, frac, U);                       // denormals
                ValueId expNorm = bin(Sub, e, u(126), U);
                ValueId expDen = bin(Sub, msb, u(148), U);
                ValueId ex = sel(denorm, expDen, expNorm, U);
                if (op == G::v_frexp_exp_i32_f32) return resultU(x, sel(special, u(0), ex, U));
                // Mantissa: a normal one keeps its fraction with exponent -1; a denormal's fraction
                // is shifted up so its top bit becomes the implicit one.
                ValueId shiftUp = bin(Sub, u(23), msb, U);
                ValueId fracDen = bin(V::Op::And, bin(V::Op::Shl, frac, bin(V::Op::And, shiftUp, u(31), U), U), u(0x7fffff), U);
                ValueId f2 = sel(denorm, fracDen, frac, U);
                ValueId mant = bin(V::Op::Or, bin(V::Op::Or, bin(V::Op::And, bits, u(0x80000000u), U), f2, U), u(0x3f000000u), U);
                return resultF(x, sel(special, v, asF(mant), F));
            }
            case G::v_cubeid_f32:
            case G::v_cubesc_f32:
            case G::v_cubetc_f32:
            case G::v_cubema_f32: return cube(x);
            case G::v_sad_u32: {
                ValueId a = srcU(x, 0), c = srcU(x, 1);
                ValueId d = sel(bin(V::Op::Gt, a, c, V::kBool), bin(Sub, a, c, U), bin(Sub, c, a, U), U);
                return resultU(x, bin(Add, d, srcU(x, 2), U));
            }
            case G::v_lerp_u8: {
                ValueId a = srcU(x, 0), c = srcU(x, 1), r = srcU(x, 2);
                ValueId acc = u(0);
                for (unsigned k = 0; k < 4; ++k) {
                    auto byteOf = [&](ValueId v) { return bin(V::Op::And, bin(V::Op::Shr, v, u(8 * k), U), u(0xff), U); };
                    ValueId s = bin(Add, bin(Add, byteOf(a), byteOf(c), U), bin(V::Op::And, bin(V::Op::Shr, r, u(8 * k), U), u(1), U), U);
                    acc = bin(V::Op::Or, acc, bin(V::Op::Shl, bin(V::Op::Shr, s, u(1), U), u(8 * k), U), U);
                }
                return resultU(x, acc);
            }
            case G::v_mad_u64_u32: {
                // {dst+1, dst} = src0 * src1 + {src2+1, src2}; carry-out of the 64-bit add to sdst.
                ValueId a = srcU(x, 0), c = srcU(x, 1);
                ValueId plo = bin(Mul, a, c, U), phi = mulHiU(a, c);
                ValueId clo = readS(x.src[2], 0), chi = readS(x.src[2], 1);
                ValueId lo = bin(Add, plo, clo, U);
                ValueId carryLo = b2u(bin(V::Op::Lt, lo, plo, V::kBool));
                ValueId hi1 = bin(Add, phi, chi, U), hi = bin(Add, hi1, carryLo, U);
                ValueId carry = bin(V::Op::Or, bin(V::Op::Lt, hi1, phi, V::kBool), bin(V::Op::Lt, hi, hi1, V::kBool), V::kBool);
                writeMask(x.sdst, bin(V::Op::And, carry, b->loadVar(execV()), V::kBool));
                writeV(x.dst.index, lo);
                writeV(x.dst.index + 1, hi);
                return;
            }
            case G::v_mbcnt_lo_u32_b32:
            case G::v_mbcnt_hi_u32_b32: {
                // Lanes below this one in the mask. With the mask all lanes (the lane-index idiom
                // mbcnt_lo(-1, 0) + mbcnt_hi(-1, ...)) it is the lane index, known in compute.
                bool all = x.src[0].kind == OperandKind::Const && int32_t(x.src[0].value) == -1;
                if (!all || laneIndex == V::kNoValue) {
                    note("v_mbcnt with a partial mask or outside compute counts 0");
                    return resultU(x, srcU(x, 1));
                }
                ValueId part = op == G::v_mbcnt_lo_u32_b32 ? bin(V::Op::Min, laneIndex, u(32), U)
                                                           : bin(V::Op::Sub, bin(V::Op::Max, laneIndex, u(32), U), u(32), U);
                return resultU(x, bin(Add, part, srcU(x, 1), U));
            }
            case G::v_movrels_b32:
            case G::v_movreld_b32:
            case G::v_movrelsd_b32: return movrel(x);
            case G::v_div_scale_f32: {
                // Scaling only matters for extreme magnitudes: pass the value through, no scaling.
                note("v_div_scale/fmas/fixup lowered to a plain division");
                writeMask(x.sdst, b->constBool(false));
                return resultF(x, srcF(x, 0));
            }
            case G::v_div_fmas_f32: return resultF(x, b->ternary(V::Op::Fma, srcF(x, 0), srcF(x, 1), srcF(x, 2), F));
            case G::v_div_fixup_f32: return resultF(x, bin(V::Op::Div, srcF(x, 2), srcF(x, 1), F));
            default: break;
        }
        error("unsupported vector instruction " + std::string(name));
    }

    // Relative VGPR indexing by M0: movrels reads v[src + M0], movreld writes v[dst + M0], movrelsd
    // both. VGPRs are variables, so the index becomes a select chain (reads) or guarded stores
    // (writes) over the registers the program uses at or past the base.
    unsigned highestVgpr = 0;
    void movrel(const Instruction& x) {
        auto U = V::kU32;
        ValueId m0 = b->loadVar(m0V());
        bool relSrc = x.op != G::v_movreld_b32, relDst = x.op != G::v_movrels_b32;
        unsigned srcBase = x.src[0].kind == OperandKind::Vgpr ? x.src[0].index : 0;
        ValueId value;
        if (relSrc) {
            value = b->loadVar(vVar(srcBase));
            for (unsigned k = 1; srcBase + k <= highestVgpr; ++k)
                value = sel(bin(V::Op::Eq, m0, u(k), V::kBool), b->loadVar(vVar(srcBase + k)), value, U);
        } else {
            value = readV(x.src[0]);
        }
        if (!relDst) return writeV(x.dst.index, value);
        for (unsigned k = 0; x.dst.index + k <= highestVgpr; ++k) {
            b->beginIf(bin(V::Op::Eq, m0, u(k), V::kBool));
            writeV(x.dst.index + k, value);
            b->end();
        }
    }

    // Cube map coordinates (AMD definitions). sc/tc are scaled by the shader with 1 / |ma| and
    // offset by 1.5 before image_sample, which the sampler path turns back into a direction.
    void cube(const Instruction& x) {
        auto F = V::kF32;
        ValueId X = srcF(x, 0), Y = srcF(x, 1), Z = srcF(x, 2);
        ValueId ax = b->unary(V::Op::Abs, X, F), ay = b->unary(V::Op::Abs, Y, F), az = b->unary(V::Op::Abs, Z, F);
        ValueId zMajor = bin(V::Op::And, bin(V::Op::Ge, az, ax, V::kBool), bin(V::Op::Ge, az, ay, V::kBool), V::kBool);
        ValueId yMajor = bin(V::Op::And, notB(zMajor), bin(V::Op::Ge, ay, ax, V::kBool), V::kBool);
        ValueId xNeg = bin(V::Op::Lt, X, f(0), V::kBool), yNeg = bin(V::Op::Lt, Y, f(0), V::kBool),
                zNeg = bin(V::Op::Lt, Z, f(0), V::kBool);
        auto pick = [&](ValueId zv, ValueId yv, ValueId xv) { return sel(zMajor, zv, sel(yMajor, yv, xv, F), F); };
        ValueId negX = b->unary(V::Op::Neg, X, F), negY = b->unary(V::Op::Neg, Y, F), negZ = b->unary(V::Op::Neg, Z, F);
        ValueId r;
        switch (x.op) {
            case G::v_cubeid_f32:
                r = pick(sel(zNeg, f(5), f(4), F), sel(yNeg, f(3), f(2), F), sel(xNeg, f(1), f(0), F));
                break;
            case G::v_cubesc_f32: r = pick(sel(zNeg, negX, X, F), X, sel(xNeg, Z, negZ, F)); break;
            case G::v_cubetc_f32: r = pick(negY, sel(yNeg, negZ, Z, F), negY); break;
            default: r = bin(V::Op::Mul, f(2), pick(Z, Y, X), F); break;
        }
        resultF(x, r);
    }

    void liftScalar(size_t idx) {
        const Instruction& x = prog[idx].in;
        G op = x.op;
        auto U = V::kU32;
        auto I = V::kI32;
        auto s0 = [&](unsigned h = 0) { return readS(x.src[0], h); };
        auto s1 = [&](unsigned h = 0) { return readS(x.src[1], h); };
        auto put = [&](ValueId v, unsigned h = 0) { writeS(x.dst, v, h); };
        auto sccNz = [&](ValueId v) { setScc(nz(v)); };
        // 64-bit logical ops: lane masks as bools, data pairs dword by dword.
        auto logic64 = [&](V::Op o, bool invertA, bool invertB, bool invertR) {
            if (isPairMask(x.dst) || isPairMask(x.src[0]) || isPairMask(x.src[1])) {
                ValueId a = readMask(x.src[0]), c = readMask(x.src[1]);
                if (invertA) a = notB(a);
                if (invertB) c = notB(c);
                ValueId r = bin(o, a, c, V::kBool);
                if (invertR) r = notB(r);
                if (x.dst.code == kExec) writesExec = true;
                writeMask(x.dst, r);
                setScc(r);
                return;
            }
            ValueId any = b->constBool(false);
            for (unsigned h = 0; h < 2; ++h) {
                ValueId a = s0(h), c = s1(h);
                if (invertA) a = b->unary(V::Op::Not, a, U);
                if (invertB) c = b->unary(V::Op::Not, c, U);
                ValueId r = bin(o == V::Op::And ? V::Op::And : o == V::Op::Or ? V::Op::Or : V::Op::Xor, a, c, U);
                if (invertR) r = b->unary(V::Op::Not, r, U);
                put(r, h);
                any = bin(V::Op::Or, any, nz(r), V::kBool);
            }
            setScc(any);
        };
        auto saveexec = [&](V::Op o, bool invertExecIn, bool invertR) {
            ValueId exec = b->loadVar(execV());
            ValueId src = readMask(x.src[0]);
            writeMask(x.dst, exec);
            ValueId e = invertExecIn ? notB(exec) : exec;
            ValueId r = bin(o, src, e, V::kBool);
            if (invertR) r = notB(r);
            writesExec = true;
            b->storeVar(execV(), r);
            setScc(r);
        };
        switch (op) {
            // SOP2
            case G::s_add_u32: {
                ValueId a = s0(), c = s1(), r = bin(V::Op::Add, a, c, U);
                setScc(bin(V::Op::Lt, r, a, V::kBool));
                return put(r);
            }
            case G::s_sub_u32: {
                ValueId a = s0(), c = s1();
                setScc(bin(V::Op::Lt, a, c, V::kBool));
                return put(bin(V::Op::Sub, a, c, U));
            }
            case G::s_addc_u32: {
                ValueId a = s0(), c = s1(), ci = b2u(b->loadVar(sccV()));
                ValueId t = bin(V::Op::Add, a, c, U), r = bin(V::Op::Add, t, ci, U);
                setScc(bin(V::Op::Or, bin(V::Op::Lt, t, a, V::kBool), bin(V::Op::Lt, r, t, V::kBool), V::kBool));
                return put(r);
            }
            case G::s_subb_u32: {
                ValueId a = s0(), c = s1(), ci = b2u(b->loadVar(sccV()));
                ValueId t = bin(V::Op::Sub, a, c, U), r = bin(V::Op::Sub, t, ci, U);
                setScc(bin(V::Op::Or, bin(V::Op::Lt, a, c, V::kBool), bin(V::Op::Lt, t, ci, V::kBool), V::kBool));
                return put(r);
            }
            case G::s_add_i32:
            case G::s_sub_i32: {
                ValueId a = s0(), c = s1();
                ValueId r = bin(op == G::s_add_i32 ? V::Op::Add : V::Op::Sub, a, c, U);
                // Signed overflow: operands' signs (equal for add, different for sub) differ from the result's.
                ValueId sa = bin(V::Op::Shr, a, u(31), U), sc = bin(V::Op::Shr, c, u(31), U), sr = bin(V::Op::Shr, r, u(31), U);
                ValueId same = op == G::s_add_i32 ? bin(V::Op::Eq, sa, sc, V::kBool) : bin(V::Op::Ne, sa, sc, V::kBool);
                setScc(bin(V::Op::And, same, bin(V::Op::Ne, sa, sr, V::kBool), V::kBool));
                return put(r);
            }
            case G::s_min_i32:
            case G::s_max_i32: {
                ValueId a = asI(s0()), c = asI(s1());
                setScc(bin(op == G::s_min_i32 ? V::Op::Lt : V::Op::Gt, a, c, V::kBool));
                return put(bit(bin(op == G::s_min_i32 ? V::Op::Min : V::Op::Max, a, c, I), U));
            }
            case G::s_min_u32:
            case G::s_max_u32: {
                ValueId a = s0(), c = s1();
                setScc(bin(op == G::s_min_u32 ? V::Op::Lt : V::Op::Gt, a, c, V::kBool));
                return put(bin(op == G::s_min_u32 ? V::Op::Min : V::Op::Max, a, c, U));
            }
            case G::s_cselect_b32: return put(sel(b->loadVar(sccV()), s0(), s1(), U));
            case G::s_cselect_b64:
                if (isPairMask(x.dst)) return writeMask(x.dst, sel(b->loadVar(sccV()), readMask(x.src[0]), readMask(x.src[1]), V::kBool));
                for (unsigned h = 0; h < 2; ++h) put(sel(b->loadVar(sccV()), s0(h), s1(h), U), h);
                return;
            case G::s_and_b32: {
                ValueId r = bin(V::Op::And, s0(), s1(), U);
                sccNz(r);
                return put(r);
            }
            case G::s_or_b32: {
                ValueId r = bin(V::Op::Or, s0(), s1(), U);
                sccNz(r);
                return put(r);
            }
            case G::s_xor_b32: {
                ValueId r = bin(V::Op::Xor, s0(), s1(), U);
                sccNz(r);
                return put(r);
            }
            case G::s_andn2_b32: {
                ValueId r = bin(V::Op::And, s0(), b->unary(V::Op::Not, s1(), U), U);
                sccNz(r);
                return put(r);
            }
            case G::s_orn2_b32: {
                ValueId r = bin(V::Op::Or, s0(), b->unary(V::Op::Not, s1(), U), U);
                sccNz(r);
                return put(r);
            }
            case G::s_nand_b32:
            case G::s_nor_b32:
            case G::s_xnor_b32: {
                V::Op o = op == G::s_nand_b32 ? V::Op::And : op == G::s_nor_b32 ? V::Op::Or : V::Op::Xor;
                ValueId r = b->unary(V::Op::Not, bin(o, s0(), s1(), U), U);
                sccNz(r);
                return put(r);
            }
            case G::s_and_b64: return logic64(V::Op::And, false, false, false);
            case G::s_or_b64: return logic64(V::Op::Or, false, false, false);
            case G::s_xor_b64: return logic64(V::Op::Xor, false, false, false);
            case G::s_andn2_b64: return logic64(V::Op::And, false, true, false);
            case G::s_orn2_b64: return logic64(V::Op::Or, false, true, false);
            case G::s_nand_b64: return logic64(V::Op::And, false, false, true);
            case G::s_nor_b64: return logic64(V::Op::Or, false, false, true);
            case G::s_xnor_b64: return logic64(V::Op::Xor, false, false, true);
            case G::s_lshl_b32: {
                ValueId r = bin(V::Op::Shl, s0(), shiftAmount(s1()), U);
                sccNz(r);
                return put(r);
            }
            case G::s_lshr_b32: {
                ValueId r = bin(V::Op::Shr, s0(), shiftAmount(s1()), U);
                sccNz(r);
                return put(r);
            }
            case G::s_ashr_i32: {
                ValueId r = bit(bin(V::Op::Shr, asI(s0()), asI(shiftAmount(s1())), I), U);
                sccNz(r);
                return put(r);
            }
            case G::s_lshl_b64:
            case G::s_lshr_b64: {
                // 64-bit shift of a register pair by [5:0].
                ValueId lo = s0(0), hi = s0(1), sh = bin(V::Op::And, s1(), u(63), U);
                ValueId big = bin(V::Op::Ge, sh, u(32), V::kBool);
                ValueId s = bin(V::Op::And, sh, u(31), U), inv = bin(V::Op::And, bin(V::Op::Sub, u(32), s, U), u(31), U);
                ValueId zeroS = bin(V::Op::Eq, s, u(0), V::kBool);
                ValueId rlo, rhi;
                if (op == G::s_lshl_b64) {
                    ValueId carry = sel(zeroS, u(0), bin(V::Op::Shr, lo, inv, U), U);
                    rhi = sel(big, bin(V::Op::Shl, lo, s, U), bin(V::Op::Or, bin(V::Op::Shl, hi, s, U), carry, U), U);
                    rlo = sel(big, u(0), bin(V::Op::Shl, lo, s, U), U);
                } else {
                    ValueId carry = sel(zeroS, u(0), bin(V::Op::Shl, hi, inv, U), U);
                    rlo = sel(big, bin(V::Op::Shr, hi, s, U), bin(V::Op::Or, bin(V::Op::Shr, lo, s, U), carry, U), U);
                    rhi = sel(big, u(0), bin(V::Op::Shr, hi, s, U), U);
                }
                setScc(bin(V::Op::Or, nz(rlo), nz(rhi), V::kBool));
                put(rlo, 0);
                return put(rhi, 1);
            }
            case G::s_bfm_b32: {
                ValueId mask = bin(V::Op::Sub, bin(V::Op::Shl, u(1), shiftAmount(s0()), U), u(1), U);
                return put(bin(V::Op::Shl, mask, shiftAmount(s1()), U));
            }
            case G::s_mul_i32: return put(bin(V::Op::Mul, s0(), s1(), U));
            case G::s_bfe_u32:
            case G::s_bfe_i32: {
                ValueId c = s1();
                ValueId off = bin(V::Op::And, c, u(31), U), width = bin(V::Op::And, bin(V::Op::Shr, c, u(16), U), u(0x7f), U);
                ValueId r = bfeWide(s0(), off, width, op == G::s_bfe_i32);
                sccNz(r);
                return put(r);
            }
            case G::s_absdiff_i32: {
                ValueId a = asI(s0()), c = asI(s1());
                ValueId d = b->unary(V::Op::Abs, bin(V::Op::Sub, a, c, I), I);
                ValueId r = bit(d, U);
                sccNz(r);
                return put(r);
            }
            // SOP1
            case G::s_mov_b32: return put(s0());
            case G::s_mov_b64:
                if (isPairMask(x.dst) || isPairMask(x.src[0])) {
                    if (x.dst.code == kExec) writesExec = true;
                    return writeMask(x.dst, readMask(x.src[0]));
                }
                put(s0(0), 0);
                return put(s0(1), 1);
            case G::s_cmov_b32: return put(sel(b->loadVar(sccV()), s0(), readS(x.dst), U));
            case G::s_cmov_b64:
                if (isPairMask(x.dst)) return writeMask(x.dst, sel(b->loadVar(sccV()), readMask(x.src[0]), readMask(x.dst), V::kBool));
                for (unsigned h = 0; h < 2; ++h) put(sel(b->loadVar(sccV()), s0(h), readS(x.dst, h), U), h);
                return;
            case G::s_not_b32: {
                ValueId r = b->unary(V::Op::Not, s0(), U);
                sccNz(r);
                return put(r);
            }
            case G::s_not_b64:
                if (isPairMask(x.dst) || isPairMask(x.src[0])) {
                    ValueId r = notB(readMask(x.src[0]));
                    if (x.dst.code == kExec) writesExec = true;
                    writeMask(x.dst, r);
                    return setScc(r);
                }
                for (unsigned h = 0; h < 2; ++h) put(b->unary(V::Op::Not, s0(h), U), h);
                return;
            case G::s_wqm_b32:
            case G::s_wqm_b64:
                if (op == G::s_wqm_b64 && (isPairMask(x.dst) || isPairMask(x.src[0]))) {
                    // Whole quad mode: a lane is on if any lane of its quad is.
                    ValueId r = readMask(x.src[0]);
                    if (quadLane != V::kNoValue) r = b->emit(V::Inst{V::Op::QuadAny, V::kBool, V::kNoValue, false, {r}, 0, 0, {}, 0});
                    else note("s_wqm outside pixel/compute keeps this lane's mask");
                    if (x.dst.code == kExec) writesExec = true;
                    writeMask(x.dst, r);
                    return setScc(r);
                }
                return put(s0());
            case G::s_brev_b32: return put(b->unary(V::Op::BitReverse, s0(), U));
            case G::s_bcnt1_i32_b32: {
                ValueId r = b->unary(V::Op::BitCount, s0(), U);
                sccNz(r);
                return put(r);
            }
            case G::s_bcnt0_i32_b32: {
                ValueId r = b->unary(V::Op::BitCount, b->unary(V::Op::Not, s0(), U), U);
                sccNz(r);
                return put(r);
            }
            case G::s_bcnt1_i32_b64:
                if (isPairMask(x.src[0])) {
                    note("s_bcnt1 of a lane mask counts this lane only");
                    ValueId r = b2u(readMask(x.src[0]));
                    sccNz(r);
                    return put(r);
                }
                {
                    ValueId r = bin(V::Op::Add, b->unary(V::Op::BitCount, s0(0), U), b->unary(V::Op::BitCount, s0(1), U), U);
                    sccNz(r);
                    return put(r);
                }
            case G::s_ff1_i32_b32: {
                ValueId v = s0();
                return put(sel(nz(v), b->unary(V::Op::FindLsb, v, U), u(0xffffffffu), U));
            }
            case G::s_flbit_i32_b32: {
                ValueId v = s0();
                return put(sel(nz(v), bin(V::Op::Sub, u(31), b->unary(V::Op::FindMsb, v, U), U), u(0xffffffffu), U));
            }
            case G::s_sext_i32_i8: return put(bfe(s0(), u(0), u(8), true));
            case G::s_sext_i32_i16: return put(bfe(s0(), u(0), u(16), true));
            case G::s_bitset0_b32:
                return put(bin(V::Op::And, readS(x.dst), b->unary(V::Op::Not, bin(V::Op::Shl, u(1), shiftAmount(s0()), U), U), U));
            case G::s_bitset1_b32: return put(bin(V::Op::Or, readS(x.dst), bin(V::Op::Shl, u(1), shiftAmount(s0()), U), U));
            case G::s_abs_i32: {
                ValueId r = bit(b->unary(V::Op::Abs, asI(s0()), I), U);
                sccNz(r);
                return put(r);
            }
            case G::s_and_saveexec_b64: return saveexec(V::Op::And, false, false);
            case G::s_or_saveexec_b64: return saveexec(V::Op::Or, false, false);
            case G::s_xor_saveexec_b64: return saveexec(V::Op::Xor, false, false);
            case G::s_andn2_saveexec_b64: return saveexec(V::Op::And, true, false);
            case G::s_orn2_saveexec_b64: return saveexec(V::Op::Or, true, false);
            case G::s_nand_saveexec_b64: return saveexec(V::Op::And, false, true);
            case G::s_nor_saveexec_b64: return saveexec(V::Op::Or, false, true);
            case G::s_xnor_saveexec_b64: return saveexec(V::Op::Xor, false, true);
            // SOPC
            case G::s_cmp_eq_i32:
            case G::s_cmp_eq_u32: return setScc(bin(V::Op::Eq, s0(), s1(), V::kBool));
            case G::s_cmp_lg_i32:
            case G::s_cmp_lg_u32: return setScc(bin(V::Op::Ne, s0(), s1(), V::kBool));
            case G::s_cmp_gt_i32: return setScc(bin(V::Op::Gt, asI(s0()), asI(s1()), V::kBool));
            case G::s_cmp_ge_i32: return setScc(bin(V::Op::Ge, asI(s0()), asI(s1()), V::kBool));
            case G::s_cmp_lt_i32: return setScc(bin(V::Op::Lt, asI(s0()), asI(s1()), V::kBool));
            case G::s_cmp_le_i32: return setScc(bin(V::Op::Le, asI(s0()), asI(s1()), V::kBool));
            case G::s_cmp_gt_u32: return setScc(bin(V::Op::Gt, s0(), s1(), V::kBool));
            case G::s_cmp_ge_u32: return setScc(bin(V::Op::Ge, s0(), s1(), V::kBool));
            case G::s_cmp_lt_u32: return setScc(bin(V::Op::Lt, s0(), s1(), V::kBool));
            case G::s_cmp_le_u32: return setScc(bin(V::Op::Le, s0(), s1(), V::kBool));
            case G::s_bitcmp0_b32:
            case G::s_bitcmp1_b32: {
                ValueId v = nz(bin(V::Op::And, bin(V::Op::Shr, s0(), shiftAmount(s1()), U), u(1), U));
                return setScc(op == G::s_bitcmp1_b32 ? v : notB(v));
            }
            // SOPK: the 16-bit immediate, sign-extended (unsigned compares zero-extend).
            case G::s_movk_i32: return put(u(uint32_t(int32_t(int16_t(x.simm16)))));
            case G::s_cmovk_i32: return put(sel(b->loadVar(sccV()), u(uint32_t(int32_t(int16_t(x.simm16)))), readS(x.dst), U));
            case G::s_addk_i32: {
                ValueId a = readS(x.dst), c = u(uint32_t(int32_t(int16_t(x.simm16))));
                ValueId r = bin(V::Op::Add, a, c, U);
                ValueId sa = bin(V::Op::Shr, a, u(31), U), sc = bin(V::Op::Shr, c, u(31), U), sr = bin(V::Op::Shr, r, u(31), U);
                setScc(bin(V::Op::And, bin(V::Op::Eq, sa, sc, V::kBool), bin(V::Op::Ne, sa, sr, V::kBool), V::kBool));
                return put(r);
            }
            case G::s_mulk_i32: return put(bin(V::Op::Mul, readS(x.dst), u(uint32_t(int32_t(int16_t(x.simm16)))), U));
            case G::s_cmpk_eq_i32:
            case G::s_cmpk_lg_i32:
            case G::s_cmpk_gt_i32:
            case G::s_cmpk_ge_i32:
            case G::s_cmpk_lt_i32:
            case G::s_cmpk_le_i32: {
                ValueId a = asI(readS(x.dst)), c = b->constI32(int16_t(x.simm16));
                V::Op o[] = {V::Op::Eq, V::Op::Ne, V::Op::Gt, V::Op::Ge, V::Op::Lt, V::Op::Le};
                return setScc(bin(o[unsigned(op) - unsigned(G::s_cmpk_eq_i32)], a, c, V::kBool));
            }
            case G::s_cmpk_eq_u32:
            case G::s_cmpk_lg_u32:
            case G::s_cmpk_gt_u32:
            case G::s_cmpk_ge_u32:
            case G::s_cmpk_lt_u32:
            case G::s_cmpk_le_u32: {
                ValueId a = readS(x.dst), c = u(x.simm16);
                V::Op o[] = {V::Op::Eq, V::Op::Ne, V::Op::Gt, V::Op::Ge, V::Op::Lt, V::Op::Le};
                return setScc(bin(o[unsigned(op) - unsigned(G::s_cmpk_eq_u32)], a, c, V::kBool));
            }
            case G::s_getreg_b32:
                note("s_getreg_b32 reads 0 (hardware state is not modelled)");
                return put(u(0));
            case G::s_setreg_b32:
            case G::s_setreg_imm32_b32: note("s_setreg (rounding/denormal mode) is ignored"); return;
            default: break;
        }
        error("unsupported scalar instruction " + std::string(mnemonic(op)));
    }

    // SMRD: s_load from a pointer, s_buffer_load from a V#.
    void liftSmrd(size_t idx) {
        const Instruction& x = prog[idx].in;
        G op = x.op;
        if (op == G::s_dcache_inv || op == G::s_dcache_inv_vol) return;
        if (op == G::s_memtime) {
            note("s_memtime reads 0");
            writeS(x.dst, u(0), 0);
            writeS(x.dst, u(0), 1);
            return;
        }
        bool buffer = op >= G::s_buffer_load_dword;
        // Offset: an immediate or literal in dwords, an SGPR in bytes.
        ValueId offset = x.src[1].kind == OperandKind::Imm || x.src[1].kind == OperandKind::Literal ? u(x.src[1].value * 4)
                                                                                                     : readS(x.src[1]);
        uint32_t slot;
        if (buffer) {
            const DescriptorSource* s = sourceOf(rsrcSource, idx, "V#");
            if (!s) return;
            slot = resource(V::ResourceKind::ConstantBuffer, *s, 4, V::Access::Read, "cb");
        } else {
            const DescriptorSource* s = sourceOf(pointerSource, idx, "pointer");
            if (!s) return;
            // dwords = 0: the source holds an address (bind memory there), not a descriptor.
            slot = resource(V::ResourceKind::ConstantBuffer, *s, 0, V::Access::Read, "mem");
        }
        for (unsigned k = 0; k < x.dst.count; ++k) {
            ValueId a = bin(V::Op::Add, offset, u(4 * k), V::kU32);
            writeS(x.dst, b->load(slot, a, V::kU32), k);
        }
    }

    // ---- Buffers -------------------------------------------------------------------------------

    // Byte address of a MUBUF/MTBUF access inside its buffer (V# base = the slot's start).
    ValueId bufferAddress(const Instruction& x) {
        const MemoryFields& mf = x.mem;
        ValueId addr = bin(V::Op::Add, u(mf.offset), readS(x.src[3]), V::kU32);
        unsigned v = 0;
        if (mf.addr64) {
            note("addr64 buffer access uses the low dword of the 64-bit address");
            return bin(V::Op::Add, addr, b->loadVar(vVar(x.src[1].index)), V::kU32);
        }
        if (mf.idxen) {
            ValueId index = b->loadVar(vVar(x.src[1].index + v++));
            // Stride: V# dword 1 bits [29:16].
            ValueId w1 = readS(x.src[2], 1);
            ValueId stride = bin(V::Op::And, bin(V::Op::Shr, w1, u(16), V::kU32), u(0x3fff), V::kU32);
            addr = bin(V::Op::Add, addr, bin(V::Op::Mul, index, stride, V::kU32), V::kU32);
        }
        if (mf.offen) addr = bin(V::Op::Add, addr, b->loadVar(vVar(x.src[1].index + v)), V::kU32);
        return addr;
    }

    // Typed load: `n` components of format dfmt/nfmt at addr -> f32/u32 bit patterns.
    std::vector<ValueId> loadFormat(uint32_t slot, ValueId addr, unsigned dfmt, unsigned nfmt, unsigned n) {
        auto U = V::kU32;
        auto F = V::kF32;
        std::vector<ValueId> c;
        unsigned bits[16] = {0, 8, 16, 8, 32, 16, 0, 0, 0, 0, 8, 32, 16, 32, 32, 0};
        unsigned count[16] = {0, 1, 1, 2, 1, 2, 3, 3, 4, 4, 4, 2, 4, 3, 4, 0};
        if (!bits[dfmt] && dfmt != 6 && dfmt != 7 && dfmt != 8 && dfmt != 9) {
            error("unsupported buffer data format " + std::to_string(dfmt));
            return std::vector<ValueId>(n, u(0));
        }
        auto convert = [&](ValueId raw, unsigned width) -> ValueId {
            // raw: zero-extended unsigned field of `width` bits.
            float maxU = float((1ull << width) - 1), maxS = float((1ull << (width - 1)) - 1);
            switch (nfmt) {
                case 0: return bit(bin(V::Op::Div, b->unary(V::Op::Convert, raw, F), f(maxU), F), U);  // unorm
                case 1: {  // snorm
                    ValueId s = bfe(raw, u(0), u(width), true);
                    ValueId v = bin(V::Op::Div, b->unary(V::Op::Convert, asI(s), F), f(maxS), F);
                    return bit(bin(V::Op::Max, v, f(-1), F), U);
                }
                case 2: return bit(b->unary(V::Op::Convert, raw, F), U);                                     // uscaled
                case 3: return bit(b->unary(V::Op::Convert, asI(bfe(raw, u(0), u(width), true)), F), U);   // sscaled
                case 4: return raw;                                                                          // uint
                case 5: return bfe(raw, u(0), u(width), true);                                               // sint
                case 7:  // float
                    if (width == 32) return raw;
                    if (width == 16) return bit(halfToF(raw, 0), U);
                    [[fallthrough]];
                default: error("unsupported buffer number format " + std::to_string(nfmt) + " for width " + std::to_string(width));
                    return raw;
            }
        };
        if (dfmt == 4 || dfmt == 11 || dfmt == 13 || dfmt == 14) {  // 32-bit components
            for (unsigned k = 0; k < count[dfmt]; ++k) c.push_back(convert(b->load(slot, bin(V::Op::Add, addr, u(4 * k), U), U), 32));
        } else if (dfmt == 2) {  // one 16-bit component, at any 2-byte boundary
            ValueId w = b->load(slot, bin(V::Op::And, addr, u(~3u), U), U);
            w = bin(V::Op::Shr, w, bin(V::Op::Shl, bin(V::Op::And, addr, u(2), U), u(3), U), U);
            c.push_back(convert(bin(V::Op::And, w, u(0xffff), U), 16));
        } else if (dfmt == 5 || dfmt == 12) {  // 16-bit components
            for (unsigned k = 0; k < count[dfmt]; k += 2) {
                ValueId w = b->load(slot, bin(V::Op::Add, addr, u(2 * k), U), U);
                c.push_back(convert(bin(V::Op::And, w, u(0xffff), U), 16));
                if (k + 1 < count[dfmt]) c.push_back(convert(bin(V::Op::Shr, w, u(16), U), 16));
            }
        } else if (dfmt == 1 || dfmt == 3 || dfmt == 10) {  // 8-bit components
            ValueId w = b->load(slot, bin(V::Op::And, addr, u(~3u), U), U);
            if (dfmt != 10) w = bin(V::Op::Shr, w, bin(V::Op::Shl, bin(V::Op::And, addr, u(3), U), u(3), U), U);
            for (unsigned k = 0; k < count[dfmt]; ++k)
                c.push_back(convert(bin(V::Op::And, bin(V::Op::Shr, w, u(8 * k), U), u(0xff), U), 8));
        } else if (dfmt == 8 || dfmt == 9) {  // 10_10_10_2 / 2_10_10_10
            ValueId w = b->load(slot, addr, U);
            unsigned widths[4] = {10, 10, 10, 2}, shifts[4] = {0, 10, 20, 30};
            if (dfmt == 9) widths[0] = 2, widths[3] = 10, shifts[0] = 0, shifts[1] = 2, shifts[2] = 12, shifts[3] = 22;
            for (unsigned k = 0; k < 4; ++k)
                c.push_back(convert(b->ternary(V::Op::BitfieldExtract, w, u(shifts[k]), u(widths[k]), U), widths[k]));
        } else {  // 10_11_11 / 11_11_10 floats
            note("packed 11/10-bit float buffer formats read as 0");
            c.assign(3, u(0));
        }
        // Missing components: (0, 0, 0, 1) as float or integer.
        bool integer = nfmt == 4 || nfmt == 5;
        while (c.size() < n) c.push_back(c.size() == 3 ? u(integer ? 1u : 0x3f800000u) : u(0));
        c.resize(n);
        return c;
    }

    void liftBuffer(size_t idx) {
        const Instruction& x = prog[idx].in;
        G op = x.op;
        if (op == G::buffer_wbinvl1 || op == G::buffer_wbinvl1_vol) return;
        const DescriptorSource* s = sourceOf(rsrcSource, idx, "V#");
        if (!s) return;
        std::string_view name = mnemonic(op);
        bool store = name.find("_store_") != std::string_view::npos;
        bool atomic = name.find("_atomic_") != std::string_view::npos;
        if (x.mem.lds) return error("buffer loads into LDS are not supported");
        auto U = V::kU32;
        V::Access access = store ? V::Access::Write : atomic ? V::Access::ReadWrite : V::Access::Read;
        // (dwords 0: the source holds the buffer's address, the V# is built by the program.)
        uint32_t slot = resource(V::ResourceKind::Buffer, *s, rawBuffer.count(idx) ? 0 : 4, access, "buf");
        ValueId addr = bufferAddress(x);
        unsigned data = x.src[0].index;
        auto guarded = [&](auto&& body) {
            if (writesExec) b->beginIf(b->loadVar(execV()));
            body();
            if (writesExec) b->end();
        };
        if (atomic) {
            V::Op aop;
            if (name.find("_add") != std::string_view::npos) aop = V::Op::AtomicAdd;
            else if (name.find("_sub") != std::string_view::npos) aop = V::Op::AtomicSub;
            else if (name.find("_umin") != std::string_view::npos || name.find("_smin") != std::string_view::npos) aop = V::Op::AtomicMin;
            else if (name.find("_umax") != std::string_view::npos || name.find("_smax") != std::string_view::npos) aop = V::Op::AtomicMax;
            else if (name.find("_and") != std::string_view::npos) aop = V::Op::AtomicAnd;
            else if (name.find("_or") != std::string_view::npos) aop = V::Op::AtomicOr;
            else if (name.find("_xor") != std::string_view::npos) aop = V::Op::AtomicXor;
            else if (name.find("_swap") != std::string_view::npos && name.find("cmp") == std::string_view::npos) aop = V::Op::AtomicExchange;
            else if (name == "buffer_atomic_cmpswap") aop = V::Op::AtomicCompareExchange;
            else return error("unsupported buffer atomic " + std::string(name));
            if (name.ends_with("_x2")) return error("64-bit buffer atomics are not supported");
            bool sign = name.find("_smin") != std::string_view::npos || name.find("_smax") != std::string_view::npos;
            V::Type t = sign ? V::kI32 : U;
            auto body = [&] {
                std::vector<ValueId> vals;
                if (aop == V::Op::AtomicCompareExchange) {
                    // vdata = {swap value, compare value}
                    vals = {b->loadVar(vVar(data + 1)), b->loadVar(vVar(data))};
                } else {
                    vals = {b->loadVar(vVar(data))};
                }
                if (sign)
                    for (ValueId& v : vals) v = asI(v);
                ValueId r = b->atomic(aop, slot, addr, vals, t);
                if (x.mem.glc) b->storeVar(vVar(data), sign ? bit(r, U) : r);
            };
            guarded(body);
            return;
        }
        // Element size per op.
        unsigned n = 0, dfmt = 0, nfmt = 0;
        bool typed = x.encoding == Encoding::MTBUF || name.find("_format_") != std::string_view::npos;
        if (typed) {
            n = unsigned(name.size() - name.rfind('_') - 1);
            if (x.encoding == Encoding::MTBUF) {
                dfmt = x.mem.dfmt;
                nfmt = x.mem.nfmt;
            } else {
                std::optional<DescriptorInfo> d = in.describe ? in.describe(*s) : std::nullopt;
                if (d && d->dfmt) {
                    dfmt = d->dfmt;
                    nfmt = d->nfmt;
                } else {
                    // Unknown at build time: 32-bit floats, one per component.
                    unsigned byCount[5] = {0, 4, 11, 13, 14};
                    dfmt = byCount[n];
                    nfmt = 7;
                    note("typed buffer access without a known format: read as 32-bit floats");
                }
            }
        }
        if (store) {
            auto body = [&] {
                if (typed) {
                    if (!(dfmt == 4 || dfmt == 11 || dfmt == 13 || dfmt == 14) || !(nfmt == 4 || nfmt == 5 || nfmt == 7)) {
                        error("typed buffer stores support 32-bit formats only");
                        return;
                    }
                    for (unsigned k = 0; k < n; ++k)
                        b->store(slot, bin(V::Op::Add, addr, u(4 * k), U), b->loadVar(vVar(data + k)));
                    return;
                }
                if (name.ends_with("_byte") || name.ends_with("_short")) {
                    error("sub-dword buffer stores are not supported yet");
                    return;
                }
                unsigned dw = x.src[0].count;
                for (unsigned k = 0; k < dw; ++k) b->store(slot, bin(V::Op::Add, addr, u(4 * k), U), b->loadVar(vVar(data + k)));
            };
            guarded(body);
            return;
        }
        // Loads.
        std::vector<ValueId> vals;
        if (typed) {
            vals = loadFormat(slot, addr, dfmt, nfmt, n);
        } else if (name.ends_with("_ubyte") || name.ends_with("_sbyte") || name.ends_with("_ushort") || name.ends_with("_sshort")) {
            ValueId w = b->load(slot, bin(V::Op::And, addr, u(~3u), U), U);
            ValueId sh = bin(V::Op::Shl, bin(V::Op::And, addr, u(3), U), u(3), U);
            unsigned width = name.ends_with("byte") ? 8 : 16;
            ValueId field = bin(V::Op::Shr, w, sh, U);
            vals.push_back(bfe(field, u(0), u(width), name.ends_with("_sbyte") || name.ends_with("_sshort")));
        } else {
            unsigned dw = name.ends_with("x4") ? 4 : name.ends_with("x3") ? 3 : name.ends_with("x2") ? 2 : 1;
            for (unsigned k = 0; k < dw; ++k) vals.push_back(b->load(slot, bin(V::Op::Add, addr, u(4 * k), U), U));
        }
        for (unsigned k = 0; k < vals.size(); ++k) writeV(data + k, vals[k]);
        if (x.mem.tfe) writeV(data + unsigned(vals.size()), u(0));
    }

    // ---- Images --------------------------------------------------------------------------------

    void liftImage(size_t idx) {
        const Instruction& x = prog[idx].in;
        std::string_view name = mnemonic(x.op);
        const DescriptorSource* ts = sourceOf(rsrcSource, idx, "T#");
        if (!ts) return;
        auto U = V::kU32;
        auto F = V::kF32;
        bool sampleOp = name.starts_with("image_sample") || name.starts_with("image_gather4") || name == "image_get_lod";
        bool store = name.starts_with("image_store");
        bool atomic = name.starts_with("image_atomic");
        if (name == "image_get_lod") return error("image_get_lod is not supported yet");
        if (atomic) return error("image atomics are not supported yet");
        bool storage = store || storageUses.count(*ts);
        uint32_t tex = resource(storage ? V::ResourceKind::StorageTexture : V::ResourceKind::Texture, *ts,
                                uint8_t(x.mem.r128 ? 4 : 8), storage ? V::Access::ReadWrite : V::Access::Read, storage ? "img" : "tex");
        const V::Resource r = m.resources[tex];  // (a copy: more resources may be added below)
        if (r.elem == V::Scalar::F16) return error("f16 texel types are not supported by the lifter");
        unsigned cw = V::coordWidth(r.dim);
        bool arrayed = V::isArrayed(r.dim);
        bool cubeDim = r.dim == V::TexDim::Cube || r.dim == V::TexDim::CubeArray;
        unsigned a = x.src[1].index;  // next address VGPR
        auto addrF = [&] { return asF(b->loadVar(vVar(a++))); };
        auto addrU = [&] { return b->loadVar(vVar(a++)); };
        unsigned dmask = x.mem.dmask;
        unsigned dataReg = x.src[0].index;
        auto writeTexel = [&](ValueId texel, V::Type texelType) {
            unsigned k = 0;
            for (unsigned c = 0; c < 4; ++c) {
                if (!(dmask >> c & 1)) continue;
                ValueId comp = texelType.width == 1 ? texel : b->extract(texel, c, texelType.element());
                writeV(dataReg + k++, asU(comp, texelType.element()));
            }
            if (x.mem.tfe || x.mem.lwe) writeV(dataReg + k, u(0));
        };
        if (sampleOp) {
            const DescriptorSource* ss = sourceOf(sampSource, idx, "S#");
            if (!ss) return;
            uint32_t smp = resource(V::ResourceKind::Sampler, *ss, 4, V::Access::Read, "smp");
            bool gather = name.starts_with("image_gather4");
            std::string_view mods = name.substr(gather ? 13 : 12);  // "_c_b_cl_o" etc.
            auto has = [&](std::string_view m2) {
                for (size_t p = 0; p < mods.size();) {
                    size_t q = mods.find('_', p + 1);
                    std::string_view part = mods.substr(p + 1, (q == std::string_view::npos ? mods.size() : q) - p - 1);
                    if (part == m2) return true;
                    if (q == std::string_view::npos) break;
                    p = q;
                }
                return false;
            };
            // Address VGPRs: [offset] [bias] [compare] [derivatives] coordinates [lod | clamp].
            ValueId offsetPacked = has("o") ? addrU() : V::kNoValue;
            ValueId bias = has("b") ? addrF() : V::kNoValue;
            ValueId compare = has("c") ? addrF() : V::kNoValue;
            std::vector<ValueId> grads;
            unsigned gw = cubeDim ? 3 : cw;
            if (has("d") || has("cd")) {
                unsigned dcount = cubeDim ? 2 : cw;  // GCN takes 2D derivatives for cubes (face space)
                std::vector<ValueId> dx, dy;
                for (unsigned k = 0; k < dcount; ++k) dx.push_back(addrF());
                for (unsigned k = 0; k < dcount; ++k) dy.push_back(addrF());
                if (cubeDim) {
                    note("cube map gradients are approximated in face space");
                    dx.push_back(f(0));
                    dy.push_back(f(0));
                }
                grads = {gw == 1 ? dx[0] : b->construct(dx, V::vec(F, uint8_t(gw))), gw == 1 ? dy[0] : b->construct(dy, V::vec(F, uint8_t(gw)))};
            }
            std::vector<ValueId> coord;
            if (cubeDim) {
                // (s, t, face[, slice]) in [1, 2] -> direction.
                ValueId s = addrF(), t = addrF(), face = addrF();
                ValueId uu = bin(V::Op::Mul, bin(V::Op::Sub, s, f(1.5f), F), f(2), F);
                ValueId vv = bin(V::Op::Mul, bin(V::Op::Sub, t, f(1.5f), F), f(2), F);
                ValueId nu = b->unary(V::Op::Neg, uu, F), nv = b->unary(V::Op::Neg, vv, F);
                // The face VGPR holds slice * 8 + face for cube arrays.
                ValueId fs = b->unary(V::Op::Convert, b->unary(V::Op::Round, face, F), U);
                ValueId fi = bin(V::Op::And, fs, u(7), U);
                auto pick = [&](std::array<ValueId, 6> per) {
                    ValueId r2 = per[5];
                    for (int k = 4; k >= 0; --k) r2 = sel(bin(V::Op::Eq, fi, u(unsigned(k)), V::kBool), per[k], r2, F);
                    return r2;
                };
                ValueId one = f(1), mone = f(-1);
                coord.push_back(pick({one, mone, uu, uu, uu, nu}));
                coord.push_back(pick({nv, nv, one, mone, nv, nv}));
                coord.push_back(pick({nu, uu, vv, nv, one, mone}));
                if (arrayed) coord.push_back(b->unary(V::Op::Convert, bin(V::Op::Shr, fs, u(3), U), F));
            } else {
                for (unsigned k = 0; k < cw; ++k) coord.push_back(addrF());
                if (arrayed) coord.push_back(b->unary(V::Op::Round, addrF(), F));
                if (x.mem.unorm) {
                    note("unnormalized sampling coordinates are divided by the texture size");
                    ValueId size = b->emit(V::Inst{V::Op::ImageSize, V::vec(U, uint8_t(cw)), V::kNoValue, false, {}, tex, 0, {}, 0});
                    for (unsigned k = 0; k < cw; ++k) {
                        ValueId sk = cw == 1 ? size : b->extract(size, k, U);
                        coord[k] = bin(V::Op::Div, coord[k], b->unary(V::Op::Convert, sk, F), F);
                    }
                }
            }
            ValueId lod = V::kNoValue;
            if (has("l")) lod = addrF();
            if (has("lz")) lod = f(0);
            if (has("cl")) {
                addrF();
                note("sampling lod clamp (_cl) is ignored");
            }
            ValueId offsetVec = V::kNoValue;
            if (offsetPacked != V::kNoValue) {
                if (!cubeDim && cw > 1) {
                    // 6-bit signed per component at bits [5:0], [13:8], [21:16].
                    std::vector<ValueId> o;
                    for (unsigned k = 0; k < cw; ++k) o.push_back(asI(bfe(offsetPacked, u(8 * k), u(6), true)));
                    offsetVec = b->construct(o, V::vec(V::kI32, uint8_t(cw)));
                } else {
                    note("texel offsets on 1D/cube textures are ignored");
                }
            }
            if (compare != V::kNoValue) {
                if (!r.depth) {
                    m.resources[tex].depth = true;
                    note("a texture sampled with compare is declared as a depth texture");
                }
                if (bias != V::kNoValue || !grads.empty()) {
                    note("compare sampling drops bias/gradients");
                    bias = V::kNoValue;
                    grads.clear();
                }
            }
            if (gather) {
                if (lod != V::kNoValue || bias != V::kNoValue || !grads.empty()) note("gather ignores lod/bias");
                lod = bias = V::kNoValue;
                grads.clear();
            } else if (lod == V::kNoValue && grads.empty() && m.stage != V::Stage::Fragment) {
                if (bias != V::kNoValue) note("bias outside a pixel shader is dropped");
                bias = V::kNoValue;
                note("implicit-lod sampling outside a pixel shader uses lod 0");
                lod = f(0);
            }
            // Optional operands in ImageOperand bit order: lod, bias, grad (2), offset, compare.
            uint8_t ops = 0;
            std::vector<ValueId> extra;
            if (lod != V::kNoValue) ops |= V::kImageLod, extra.push_back(lod);
            if (bias != V::kNoValue) ops |= V::kImageBias, extra.push_back(bias);
            if (!grads.empty()) ops |= V::kImageGrad, extra.insert(extra.end(), grads.begin(), grads.end());
            if (offsetVec != V::kNoValue) ops |= V::kImageOffset, extra.push_back(offsetVec);
            if (compare != V::kNoValue) ops |= V::kImageCompare, extra.push_back(compare);
            V::Type ct = coord.size() == 1 ? F : V::vec(F, uint8_t(coord.size()));
            ValueId cv = coord.size() == 1 ? coord[0] : b->construct(coord, ct);
            const V::Resource rr = m.resources[tex];
            V::Type texel = rr.depth ? F : V::Type(rr.elem, 4);
            if (gather) {
                unsigned comp = dmask ? unsigned(std::countr_zero(dmask)) : 0;
                if (rr.depth) comp = 0;
                V::Inst g;
                g.op = V::Op::ImageGather;
                g.type = V::Type(rr.depth ? V::Scalar::F32 : rr.elem, 4);
                g.sym = tex;
                g.sampler = smp;
                g.lits = {comp};
                g.args = {cv};
                g.args.insert(g.args.end(), extra.begin(), extra.end());
                g.image = ops;
                ValueId res = b->emit(g);
                // Gather returns 4 texels of one component: all four VGPRs.
                for (unsigned k = 0; k < 4; ++k) writeV(dataReg + k, asU(b->extract(res, k, g.type.element()), g.type.element()));
                if (x.mem.tfe || x.mem.lwe) writeV(dataReg + 4, u(0));
                return;
            }
            ValueId res = b->imageSample(tex, smp, cv, ops, extra, texel);
            if (rr.depth) {
                // A depth sample: the value in the first selected component.
                writeV(dataReg, bit(res, U));
                return;
            }
            return writeTexel(res, texel);
        }
        // image_load / image_load_mip / image_store / image_get_resinfo: integer coordinates.
        if (name == "image_get_resinfo") {
            ValueId mip = addrU();
            unsigned sw = (r.dim == V::TexDim::D1) ? 1 : (r.dim == V::TexDim::D3 || arrayed) ? 3 : 2;
            V::Inst q{V::Op::ImageSize, V::vec(U, uint8_t(sw)), V::kNoValue, false, {mip}, tex, 0, {}, V::kImageLod};
            if (storage) q.args.clear(), q.image = 0;
            ValueId size = b->emit(q);
            ValueId levels = storage ? u(1) : b->emit(V::Inst{V::Op::ImageLevels, U, V::kNoValue, false, {}, tex, 0, {}, 0});
            std::vector<ValueId> comps;
            for (unsigned k = 0; k < 3; ++k) comps.push_back(k < sw ? (sw == 1 ? size : b->extract(size, k, U)) : u(1));
            comps.push_back(levels);
            unsigned k = 0;
            for (unsigned c = 0; c < 4; ++c)
                if (dmask >> c & 1) writeV(dataReg + k++, comps[c]);
            return;
        }
        std::vector<ValueId> coord;
        unsigned icw = cubeDim ? 3 : cw;
        for (unsigned k = 0; k < icw; ++k) coord.push_back(addrU());
        if (arrayed) coord.push_back(addrU());
        ValueId mip = name.find("_mip") != std::string_view::npos ? addrU() : V::kNoValue;
        V::Type ct = coord.size() == 1 ? U : V::vec(U, uint8_t(coord.size()));
        ValueId cv = coord.size() == 1 ? coord[0] : b->construct(coord, ct);
        V::Type texel(r.elem, 4);
        if (store) {
            std::vector<ValueId> comps;
            unsigned k = 0;
            for (unsigned c = 0; c < 4; ++c) {
                ValueId raw = (dmask >> c & 1) ? b->loadVar(vVar(dataReg + k++)) : u(c == 3 && r.elem == V::Scalar::F32 ? 0x3f800000u : 0);
                comps.push_back(r.elem == V::Scalar::U32 ? raw : bit(raw, texel.element()));
            }
            if (mip != V::kNoValue) note("image_store_mip writes mip 0");
            if (writesExec) b->beginIf(b->loadVar(execV()));
            b->imageStore(tex, cv, b->construct(comps, texel));
            if (writesExec) b->end();
            return;
        }
        ValueId res;
        if (storage) {
            res = b->imageFetch(tex, cv, texel);
        } else {
            V::Inst q{V::Op::ImageFetch, texel, V::kNoValue, false, {cv}, tex, 0, {}, 0};
            if (mip != V::kNoValue) q.args.push_back(mip), q.image = V::kImageLod;
            res = b->emit(q);
        }
        writeTexel(res, texel);
    }

    // ---- LDS -----------------------------------------------------------------------------------

    void liftDs(size_t idx) {
        const Instruction& x = prog[idx].in;
        std::string_view name = mnemonic(x.op);
        auto U = V::kU32;
        if (name == "ds_swizzle_b32") return swizzle(x);
        if (m.sharedBytes == 0) return error("LDS access in a program without LDS (" + std::string(name) + ")");
        if (x.mem.gds) return error("GDS access is not supported (" + std::string(name) + ")");
        uint32_t lds = V::kTargetShared;
        ValueId base = x.src[0].kind != OperandKind::None ? b->loadVar(vVar(x.src[0].index)) : u(0);
        auto at = [&](uint32_t off) { return bin(V::Op::Add, base, u(off), U); };
        auto reg = [&](const Operand& o, unsigned k) { return b->loadVar(vVar(o.index + k)); };
        bool two = name.find("2") != std::string_view::npos && (name.find("read2") != std::string_view::npos || name.find("write2") != std::string_view::npos);
        unsigned esz = name.find("_b64") != std::string_view::npos ? 8 : 4;
        unsigned mul = name.find("st64") != std::string_view::npos ? esz * 64 : esz;
        uint32_t off0 = x.mem.offset & 0xff, off1 = x.mem.offset >> 8;
        auto guarded = [&](auto&& body) {
            if (writesExec) b->beginIf(b->loadVar(execV()));
            body();
            if (writesExec) b->end();
        };
        if (name.starts_with("ds_read")) {
            std::vector<ValueId> vals;
            if (two) {
                for (uint32_t off : {off0 * mul, off1 * mul})
                    for (unsigned k = 0; k < esz / 4; ++k) vals.push_back(b->load(lds, at(off + 4 * k), U));
            } else if (name == "ds_read_b32" || name == "ds_read_b64" || name == "ds_read_b96" || name == "ds_read_b128") {
                unsigned n = x.dst.count;
                for (unsigned k = 0; k < n; ++k) vals.push_back(b->load(lds, at(x.mem.offset + 4 * k), U));
            } else {
                // i8/u8/i16/u16
                ValueId a = at(x.mem.offset);
                ValueId w = b->load(lds, bin(V::Op::And, a, u(~3u), U), U);
                ValueId f2 = bin(V::Op::Shr, w, bin(V::Op::Shl, bin(V::Op::And, a, u(3), U), u(3), U), U);
                unsigned width = name.ends_with("8") ? 8 : 16;
                vals.push_back(bfe(f2, u(0), u(width), name[8] == 'i'));
            }
            for (unsigned k = 0; k < vals.size(); ++k) writeV(x.dst.index + k, vals[k]);
            return;
        }
        if (name.starts_with("ds_write")) {
            if (name.find("src2") != std::string_view::npos) return error("ds_write_src2 is not supported");
            guarded([&] {
                if (two) {
                    for (unsigned k = 0; k < esz / 4; ++k) b->store(lds, at(off0 * mul + 4 * k), reg(x.src[1], k));
                    for (unsigned k = 0; k < esz / 4; ++k) b->store(lds, at(off1 * mul + 4 * k), reg(x.src[2], k));
                } else if (name == "ds_write_b8" || name == "ds_write_b16") {
                    error("sub-dword LDS writes are not supported yet");
                } else {
                    for (unsigned k = 0; k < x.src[1].count; ++k) b->store(lds, at(x.mem.offset + 4 * k), reg(x.src[1], k));
                }
            });
            return;
        }
        if (name == "ds_swizzle_b32") {
            note("ds_swizzle_b32 returns this lane's own value");
            return writeV(x.dst.index, reg(x.src[0], 0));
        }
        if (name.find("_b64") != std::string_view::npos || name.find("_u64") != std::string_view::npos ||
            name.find("_i64") != std::string_view::npos || name.find("_f") != std::string_view::npos)
            return error("unsupported LDS operation " + std::string(name));
        V::Op aop;
        if (name.starts_with("ds_add")) aop = V::Op::AtomicAdd;
        else if (name.starts_with("ds_sub")) aop = V::Op::AtomicSub;
        else if (name.starts_with("ds_min")) aop = V::Op::AtomicMin;
        else if (name.starts_with("ds_max")) aop = V::Op::AtomicMax;
        else if (name.starts_with("ds_and")) aop = V::Op::AtomicAnd;
        else if (name.starts_with("ds_or")) aop = V::Op::AtomicOr;
        else if (name.starts_with("ds_xor")) aop = V::Op::AtomicXor;
        else if (name.starts_with("ds_wrxchg_rtn")) aop = V::Op::AtomicExchange;
        else if (name.starts_with("ds_cmpst")) aop = V::Op::AtomicCompareExchange;
        else return error("unsupported LDS operation " + std::string(name));
        if (name.find("src2") != std::string_view::npos) return error("unsupported LDS operation " + std::string(name));
        bool sign = name.ends_with("_i32");
        V::Type t = sign ? V::kI32 : U;
        guarded([&] {
            std::vector<ValueId> vals;
            // DS_CMPST: data0 (DATA) is the comparand, data1 (DATA2) the new value.
            if (aop == V::Op::AtomicCompareExchange) vals = {reg(x.src[1], 0), reg(x.src[2], 0)};
            else vals = {reg(x.src[1], 0)};
            if (sign)
                for (ValueId& v : vals) v = asI(v);
            ValueId r = b->atomic(aop, lds, at(x.mem.offset), vals, t);
            if (x.dst.kind != OperandKind::None) b->storeVar(vVar(x.dst.index), sign ? bit(r, U) : r);
        });
    }

    // ds_swizzle_b32: lanes exchange values. Within a quad (QUAD_PERM, or a bit-mask pattern that
    // keeps lane-id bits 2-4) it is exact through quad_shuffle; wider patterns read the lane's own
    // value (noted).
    void swizzle(const Instruction& x) {
        auto U = V::kU32;
        ValueId v = b->loadVar(vVar(x.src[0].index));
        uint32_t off = x.mem.offset;
        ValueId lane = V::kNoValue;
        if (quadLane != V::kNoValue) {
            if ((off & 0xff00) == 0x8000) {
                // lane = pattern[2*q+1 : 2*q]
                ValueId shift = bin(V::Op::Shl, quadLane, u(1), U);
                lane = bin(V::Op::And, bin(V::Op::Shr, u(off & 0xff), shift, U), u(3), U);
            } else if (!(off & 0x8000)) {
                uint32_t andMask = off & 0x1f, orMask = (off >> 5) & 0x1f, xorMask = (off >> 10) & 0x1f;
                if ((andMask & 0x1c) == 0x1c && !(orMask & 0x1c) && !(xorMask & 0x1c)) {
                    ValueId l = bin(V::Op::And, quadLane, u(andMask & 3), U);
                    l = bin(V::Op::Or, l, u(orMask & 3), U);
                    lane = bin(V::Op::Xor, l, u(xorMask & 3), U);
                }
            }
        }
        if (lane == V::kNoValue) {
            note("a ds_swizzle_b32 pattern beyond the quad (or outside pixel/compute) returns this lane's own value");
            return writeV(x.dst.index, v);
        }
        V::Inst q{V::Op::QuadShuffle, U, V::kNoValue, false, {v, lane}, 0, 0, {}, 0};
        writeV(x.dst.index, b->emit(q));
    }

    // ---- Interpolation and exports -------------------------------------------------------------

    uint32_t inputFor(unsigned attr, V::Interp interp, V::Sampling sampling) {
        int key = int(interp) * 4 + int(sampling);
        auto it = inputs.find({attr, key});
        if (it != inputs.end()) return it->second;
        for (auto& [k, v] : inputs)
            if (k.first == attr) {
                note("an attribute read with two interpolation modes uses the first");
                return v;
            }
        V::IoVar io;
        io.name = "attr" + std::to_string(attr);
        io.type = V::vec(V::kF32, 4);
        io.location = attr;
        io.interp = interp;
        io.sampling = sampling;
        uint32_t index = uint32_t(m.inputs.size());
        m.inputs.push_back(io);
        inputs[{attr, key}] = index;
        return index;
    }

    void liftInterp(size_t idx) {
        const Instruction& x = prog[idx].in;
        unsigned attr = x.mem.attr, chan = x.mem.chan;
        if (x.op == G::v_interp_mov_f32) {
            if (x.src[0].value != 2) note("v_interp_mov from p10/p20 reads the provoking vertex");
            ValueId v = b->loadInput(inputFor(attr, V::Interp::Flat, V::Sampling::Center), V::vec(V::kF32, 4));
            return writeV(x.dst.index, bit(b->extract(v, chan, V::kF32), V::kU32));
        }
        auto it = bary.find(x.src[0].index);
        V::Interp interp = V::Interp::Perspective;
        V::Sampling sampling = V::Sampling::Center;
        if (it != bary.end()) interp = it->second.interp, sampling = it->second.sampling;
        else note("interpolation with a barycentric VGPR the inputs did not set: perspective centre");
        if (x.op == G::v_interp_p1_f32) return;  // the value comes whole at p2
        ValueId v = b->loadInput(inputFor(attr, interp, sampling), V::vec(V::kF32, 4));
        writeV(x.dst.index, bit(b->extract(v, chan, V::kF32), V::kU32));
    }

    uint32_t outputFor(std::map<unsigned, uint32_t>& map, unsigned index, const std::string& prefix) {
        auto it = map.find(index);
        if (it != map.end()) return it->second;
        V::IoVar io;
        io.name = prefix + std::to_string(index);
        io.type = V::vec(V::kF32, 4);
        io.location = index;
        uint32_t o = uint32_t(m.outputs.size());
        m.outputs.push_back(io);
        return map[index] = o;
    }

    void liftExport(size_t idx) {
        const Instruction& x = prog[idx].in;
        const MemoryFields& mf = x.mem;
        auto F = V::kF32;
        unsigned t = mf.target;
        if (t == 9) {
            if (mf.done && in.stage == ShaderStage::Pixel && writesExec) {
                b->beginIf(notB(b->loadVar(execV())));
                b->discard();
                b->end();
            }
            return;
        }
        // Components (f32), unexported ones 0 (alpha/w 1).
        std::array<ValueId, 4> c{};
        CompressedExport packing = t < 8 ? in.compressedExport[t] : CompressedExport::Fp16;
        auto unpack = [&](ValueId bits, unsigned hi) -> ValueId {
            if (packing == CompressedExport::Fp16) return halfToF(bits, hi);
            bool sign = packing == CompressedExport::Snorm16 || packing == CompressedExport::Sint16;
            ValueId v = bfe(bits, u(16 * hi), u(16), sign);
            ValueId fv = b->unary(V::Op::Convert, sign ? asI(v) : v, F);
            if (packing == CompressedExport::Unorm16) return bin(V::Op::Div, fv, f(65535.0f), F);
            if (packing == CompressedExport::Snorm16) return bin(V::Op::Max, bin(V::Op::Div, fv, f(32767.0f), F), f(-1), F);
            note("16-bit integer compressed exports become float values");
            return fv;
        };
        for (unsigned k = 0; k < 4; ++k) {
            if (mf.compr) {
                const Operand& o = x.src[k / 2];
                c[k] = o.kind == OperandKind::None ? f(k == 3 ? 1.0f : 0.0f) : unpack(b->loadVar(vVar(o.index)), k & 1);
            } else {
                const Operand& o = x.src[k];
                c[k] = o.kind == OperandKind::None ? f(k == 3 ? 1.0f : 0.0f) : asF(b->loadVar(vVar(o.index)));
            }
        }
        ValueId vec4 = b->construct(std::vector<ValueId>(c.begin(), c.end()), V::vec(F, 4));
        bool guard = writesExec && in.stage == ShaderStage::Pixel;
        if (guard) {
            b->beginIf(notB(b->loadVar(execV())));
            if (mf.done || mf.vm) b->discard();
            b->end();
        }
        if (t < 8) {
            if (in.stage != ShaderStage::Pixel) return error("colour export outside a pixel shader");
            return b->storeOutput(outputFor(mrtOutputs, t, "color"), vec4);
        }
        if (t == 8) {
            if (in.stage != ShaderStage::Pixel) return error("depth export outside a pixel shader");
            if (depthOut < 0) {
                V::IoVar io;
                io.name = "depth";
                io.type = F;
                io.builtin = V::Builtin::FragDepth;
                depthOut = int64_t(m.outputs.size());
                m.outputs.push_back(io);
            }
            if (mf.enable & 1) b->storeOutput(uint32_t(depthOut), c[0]);
            if (mf.enable & ~1u) note("stencil / sample mask exports are ignored");
            return;
        }
        if (t >= 12 && t < 16) {
            if (t != 12) {
                note("pos1-pos3 exports (point size, clip distances, layer) are ignored");
                return;
            }
            if (positionOut < 0) {
                V::IoVar io;
                io.name = "position";
                io.type = V::vec(F, 4);
                io.builtin = V::Builtin::Position;
                positionOut = int64_t(m.outputs.size());
                m.outputs.push_back(io);
            }
            return b->storeOutput(uint32_t(positionOut), vec4);
        }
        if (t >= 32) return b->storeOutput(outputFor(paramOutputs, t - 32, "param"), vec4);
        error("unsupported export target " + std::to_string(t));
    }

    // ---- One instruction -----------------------------------------------------------------------

    void liftOne(size_t idx) {
        const Instruction& x = prog[idx].in;
        switch (x.encoding) {
            case Encoding::SOP2:
            case Encoding::SOP1:
            case Encoding::SOPC:
            case Encoding::SOPK:
                if (x.op == G::s_getpc_b64 && !prog[idx].fetch) {
                    // An address in the program image, relative to it: only constant tables are
                    // reached this way, through the Code descriptor source (bound by the client).
                    writeS(x.dst, u((prog[idx].pc + 1) * 4), 0);
                    writeS(x.dst, u(0), 1);
                    return;
                }
                if (x.op == G::s_swappc_b64 || x.op == G::s_setpc_b64 || x.op == G::s_getpc_b64) {
                    if (x.op == G::s_swappc_b64) return liftFetchCall();
                    if (x.op == G::s_setpc_b64 && prog[idx].fetch) return;
                    return error("program counter manipulation (" + std::string(mnemonic(x.op)) + ") is not supported");
                }
                return liftScalar(idx);
            case Encoding::SOPP:
                switch (x.op) {
                    case G::s_barrier:
                        if (m.stage == V::Stage::Compute) {
                            // Lanes that left an EXEC-masked region would skip the barrier and hang
                            // the threadgroup: barriers inside EXEC branches are not supported.
                            if (writesExec) note("s_barrier in a program that changes EXEC: it must not sit in a divergent region");
                            b->barrier();
                        }
                        return;
                    case G::s_sendmsg:
                    case G::s_sendmsghalt: note("s_sendmsg is ignored"); return;
                    default: return;  // nop, waitcnt, endpgm (block terminator), branches, priorities
                }
            case Encoding::SMRD: return liftSmrd(idx);
            case Encoding::VOP1:
            case Encoding::VOP2:
            case Encoding::VOPC:
            case Encoding::VOP3: return liftVop(idx);
            case Encoding::VINTRP: return liftInterp(idx);
            case Encoding::MUBUF:
            case Encoding::MTBUF: return liftBuffer(idx);
            case Encoding::MIMG: return liftImage(idx);
            case Encoding::DS: return liftDs(idx);
            case Encoding::FLAT: return error("FLAT (raw pointer) memory access is not supported");
            case Encoding::EXP: return liftExport(idx);
        }
    }

    std::vector<Decoded> fetchProg;
    std::vector<size_t> fetchIndex;  // fetch instructions' positions in `prog` (appended at the end)

    void liftFetchCall() {
        if (fetchIndex.empty()) return error("s_swappc_b64 without a fetch subroutine");
        for (size_t i : fetchIndex) liftOne(i);
    }

    // ---- Control flow ----------------------------------------------------------------------------

    ValueId condition(const BasicBlock& bb) {
        switch (bb.cond) {
            case G::s_cbranch_scc0: return notB(b->loadVar(sccV()));
            case G::s_cbranch_scc1: return b->loadVar(sccV());
            case G::s_cbranch_vccz: return notB(b->loadVar(vccV()));
            case G::s_cbranch_vccnz: return b->loadVar(vccV());
            case G::s_cbranch_execz: return notB(b->loadVar(execV()));
            case G::s_cbranch_execnz: return b->loadVar(execV());
            default: return b->constBool(false);
        }
    }

    void liftBlockBody(const BasicBlock& bb) {
        for (size_t i = bb.first; i <= bb.last; ++i) liftOne(i);
    }

    // ---- Structured form: nested ifs and natural loops ------------------------------------------
    //
    // Loops are accepted in the shape compilers lay out: blocks h..l contiguous, one back edge l -> h,
    // entered only at h, and every edge leaving them going to one exit block E. Inside, an edge to
    // h is `continue`, an edge to E is `break`. Branch joins are immediate post-dominators on the
    // graph without back edges. Anything else uses the dispatch loop.

    struct Loop {
        int head = 0, latch = 0, exit = -1;
    };
    std::map<int, Loop> loops;  // by header
    std::set<std::pair<int, int>> backEdges;

    std::vector<int> successors(int k) const {
        std::vector<int> s2;
        const BasicBlock& bb = blocks[k];
        if (bb.taken >= 0) s2.push_back(bb.taken);
        if (bb.term != BasicBlock::Jump && bb.term != BasicBlock::End && bb.next >= 0) s2.push_back(bb.next);
        return s2;
    }

    bool findLoops() {
        int n = int(blocks.size());
        for (int k = 0; k < n; ++k)
            for (int t : successors(k))
                if (t <= k) {
                    if (loops.count(t)) return false;  // two back edges to one header
                    loops[t] = {t, k, -1};
                    backEdges.insert({k, t});
                }
        for (auto& [h, lp] : loops) {
            for (int k = 0; k < n; ++k)
                for (int t : successors(k)) {
                    bool from = k >= lp.head && k <= lp.latch, to = t >= lp.head && t <= lp.latch;
                    if (!from && to && t != lp.head) return false;  // entered past its header
                    if (from && !to) {
                        if (lp.exit >= 0 && lp.exit != t) return false;  // two exits
                        lp.exit = t;
                    }
                }
            if (lp.exit < 0) return false;  // only left through s_endpgm: keep it simple
            // Nested loops must nest as ranges.
            for (auto& [h2, l2] : loops)
                if (h2 != h && h2 > lp.head && h2 <= lp.latch && l2.latch > lp.latch) return false;
        }
        return true;
    }

    // DAG check (back edges removed).
    bool dagAcyclic() {
        std::vector<int> state(blocks.size(), 0);
        bool ok = true;
        std::function<void(int)> dfs = [&](int k) {
            if (k < 0 || !ok) return;
            if (state[k] == 1) {
                ok = false;
                return;
            }
            if (state[k] == 2) return;
            state[k] = 1;
            for (int t : successors(k))
                if (!backEdges.count({k, t})) dfs(t);
            state[k] = 2;
        };
        dfs(0);
        return ok;
    }

    // Immediate post-dominators (virtual exit = blocks.size()) on the graph without back edges.
    std::vector<int> postDominators() {
        int n = int(blocks.size()), exit = n;
        std::vector<std::vector<int>> succ(n + 1);
        for (int k = 0; k < n; ++k) {
            if (blocks[k].term == BasicBlock::End) succ[k].push_back(exit);
            for (int t : successors(k))
                if (!backEdges.count({k, t})) succ[k].push_back(t);
            if (succ[k].empty()) succ[k].push_back(exit);  // a latch whose only way on is back
        }
        std::vector<std::set<int>> pdom(n + 1);
        std::set<int> all;
        for (int k = 0; k <= n; ++k) all.insert(k);
        for (int k = 0; k < n; ++k) pdom[k] = all;
        pdom[exit] = {exit};
        bool changed = true;
        while (changed) {
            changed = false;
            for (int k = n - 1; k >= 0; --k) {
                std::set<int> acc;
                bool first = true;
                for (int s2 : succ[k]) {
                    if (first) acc = pdom[s2], first = false;
                    else {
                        std::set<int> t;
                        std::set_intersection(acc.begin(), acc.end(), pdom[s2].begin(), pdom[s2].end(), std::inserter(t, t.begin()));
                        acc = t;
                    }
                }
                acc.insert(k);
                if (acc != pdom[k]) pdom[k] = acc, changed = true;
            }
        }
        std::vector<int> ipdom(n, exit);
        for (int k = 0; k < n; ++k)
            for (int c : pdom[k]) {
                if (c == k) continue;
                bool imm = true;
                for (int o : pdom[k])
                    if (o != k && o != c && !pdom[c].count(o)) imm = false;
                if (imm) ipdom[k] = c;
            }
        return ipdom;
    }

    // One walk serves both the size check (dry) and the emission.
    struct Walk {
        bool dry = true;
        size_t count = 0, budget = 0;
        std::vector<const Loop*> loopStack;
        bool ok = true;
    };

    // Leaves the current loop iteration if `k` is its header (continue) or exit (break).
    bool loopEdge(int k, Walk& w) {
        if (w.loopStack.empty()) return false;
        const Loop* lp = w.loopStack.back();
        if (k == lp->head) {
            if (!w.dry) b->continueLoop();
            return true;
        }
        if (k == lp->exit) {
            if (!w.dry) b->breakLoop();
            return true;
        }
        for (size_t i = 0; i + 1 < w.loopStack.size(); ++i)
            if (k == w.loopStack[i]->head || k == w.loopStack[i]->exit) w.ok = false;  // a jump out of two loops
        return false;
    }

    void walk(int k, int stop, const std::vector<int>& ipdom, Walk& w, bool loopEntry = false) {
        int exit = int(blocks.size());
        while (w.ok && k >= 0 && k != stop && k != exit) {
            if (!loopEntry && loopEdge(k, w)) return;
            if (!loopEntry && loops.count(k) && (w.loopStack.empty() || w.loopStack.back()->head != k)) {
                const Loop& lp = loops[k];
                if (!w.dry) b->beginLoop();
                w.loopStack.push_back(&lp);
                walk(k, -2, ipdom, w, true);
                w.loopStack.pop_back();
                if (!w.dry) {
                    b->breakLoop();  // (reached only if the body falls out without break/continue)
                    b->end();
                }
                k = lp.exit;
                continue;
            }
            loopEntry = false;
            const BasicBlock& bb = blocks[k];
            w.count += bb.last - bb.first + 1;
            if (w.count > w.budget) {
                w.ok = false;
                return;
            }
            if (!w.dry) liftBlockBody(bb);
            switch (bb.term) {
                case BasicBlock::End:
                    if (!w.dry) b->ret();
                    return;
                case BasicBlock::Jump: k = bb.taken; break;
                case BasicBlock::Fall: k = bb.next; break;
                case BasicBlock::Cond: {
                    if (bb.taken < 0 || bb.next < 0) {
                        error("a branch leaves the program");
                        w.ok = false;
                        return;
                    }
                    int join = ipdom[k];
                    if (!w.dry) b->beginIf(condition(bb));
                    walk(bb.taken, join, ipdom, w);
                    if (!w.dry) b->beginElse();
                    walk(bb.next, join, ipdom, w);
                    if (!w.dry) b->end();
                    k = join;
                    break;
                }
            }
        }
        if (k >= 0 && k != stop && k != exit && w.ok) loopEdge(k, w);
    }

    void dispatchLoop() {
        uint32_t blk = b->variable(V::kU32);
        b->storeVar(blk, u(0));
        b->beginLoop();
        for (size_t k = 0; k < blocks.size(); ++k) {
            const BasicBlock& bb = blocks[k];
            b->beginIf(bin(V::Op::Eq, b->loadVar(blk), u(uint32_t(k)), V::kBool));
            liftBlockBody(bb);
            switch (bb.term) {
                case BasicBlock::End: b->ret(); break;
                case BasicBlock::Jump: b->storeVar(blk, u(uint32_t(bb.taken))); b->continueLoop(); break;
                case BasicBlock::Fall: b->storeVar(blk, u(uint32_t(bb.next))); b->continueLoop(); break;
                case BasicBlock::Cond:
                    b->storeVar(blk, sel(condition(bb), u(uint32_t(bb.taken)), u(uint32_t(bb.next)), V::kU32));
                    b->continueLoop();
                    break;
            }
            b->end();
        }
        b->breakLoop();
        b->end();
    }

    // ---- Entry state ---------------------------------------------------------------------------

    void entryState() {
        auto U = V::kU32;
        b->storeVar(execV(), b->constBool(true));
        b->storeVar(vccV(), b->constBool(false));
        b->storeVar(sccV(), b->constBool(false));
        b->storeVar(m0V(), u(0));
        unsigned s = 0;
        for (; s < in.userSgprs; ++s) b->storeVar(sVar(s), b->load(userData(), u(4 * s), U));
        auto builtinIn = [&](V::Builtin bi, V::Type t, const char* name) {
            V::IoVar io;
            io.name = name;
            io.type = t;
            io.builtin = bi;
            uint32_t index = uint32_t(m.inputs.size());
            m.inputs.push_back(io);
            return b->loadInput(index, t);
        };
        switch (in.stage) {
            case ShaderStage::Vertex: {
                ValueId vid = builtinIn(V::Builtin::VertexId, U, "vertex_id");
                b->storeVar(vVar(0), vid);
                if (in.vgprCompCnt >= 1) {
                    ValueId iid = builtinIn(V::Builtin::InstanceId, U, "instance_id");
                    for (unsigned v = 1; v <= in.vgprCompCnt && v < 4; ++v) b->storeVar(vVar(v), iid);
                    if (in.vgprCompCnt >= 1) note("instance step-rate VGPRs (v1, v2) hold the instance id");
                }
                break;
            }
            case ShaderStage::Pixel: {
                // VGPRs in SPI_PS_INPUT_ADDR order.
                unsigned v = 0;
                struct Item {
                    unsigned bitIndex, count;
                };
                const Item items[] = {{0, 2}, {1, 2}, {2, 2}, {3, 3}, {4, 2}, {5, 2}, {6, 2}, {7, 1},
                                      {8, 1}, {9, 1}, {10, 1}, {11, 1}, {12, 1}, {13, 1}, {14, 1}, {15, 1}};
                ValueId pos = V::kNoValue;
                auto position = [&] {
                    if (pos == V::kNoValue) pos = builtinIn(V::Builtin::Position, V::vec(V::kF32, 4), "frag_coord");
                    return pos;
                };
                {
                    // The quad lane: GCN and Metal number a pixel quad the same way (x + 2y).
                    ValueId px = b->unary(V::Op::Convert, b->extract(position(), 0, V::kF32), U);
                    ValueId py = b->unary(V::Op::Convert, b->extract(position(), 1, V::kF32), U);
                    quadLane = bin(V::Op::Add, bin(V::Op::And, px, u(1), U),
                                   bin(V::Op::Shl, bin(V::Op::And, py, u(1), U), u(1), U), U);
                }
                for (const Item& it : items) {
                    if (!(in.psInputAddr >> it.bitIndex & 1)) continue;
                    unsigned bi = it.bitIndex;
                    if (bi <= 6 && bi != 3) {
                        V::Interp ip = bi >= 4 ? V::Interp::Linear : V::Interp::Perspective;
                        unsigned mode = bi >= 4 ? bi - 4 : bi;
                        V::Sampling sm = mode == 0 ? V::Sampling::Sample : mode == 2 ? V::Sampling::Centroid : V::Sampling::Center;
                        bary[v] = {ip, sm, false};
                        bary[v + 1] = {ip, sm, true};
                        b->storeVar(vVar(v), u(0));
                        b->storeVar(vVar(v + 1), u(0));
                    } else if (bi == 3 || bi == 7 || bi == 15) {
                        for (unsigned k = 0; k < it.count; ++k) b->storeVar(vVar(v + k), u(0));
                        if (bi == 3) note("pull-model interpolation inputs are 0");
                    } else if (bi >= 8 && bi <= 11) {
                        ValueId c = b->extract(position(), bi - 8, V::kF32);
                        if (bi == 11) c = b->unary(V::Op::Rcp, c, V::kF32), note("POS_W input: 1 / w of the fragment position");
                        b->storeVar(vVar(v), bit(c, U));
                    } else if (bi == 12) {
                        ValueId ff = builtinIn(V::Builtin::FrontFacing, V::kBool, "front_facing");
                        // 1.0 front, 0 back: right whether the program tests it as a float (> 0) or
                        // as an integer (!= 0).
                        b->storeVar(vVar(v), sel(ff, u(0x3f800000u), u(0), U));
                    } else if (bi == 13) {
                        ValueId sid = builtinIn(V::Builtin::SampleId, U, "sample_id");
                        b->storeVar(vVar(v), bin(V::Op::Shl, sid, u(8), U));
                    } else if (bi == 14) {
                        b->storeVar(vVar(v), builtinIn(V::Builtin::SampleMask, U, "sample_mask"));
                    }
                    v += it.count;
                }
                // The primitive mask SGPR (for LDS parameter loads), unused here.
                b->storeVar(sVar(s++), u(0));
                break;
            }
            case ShaderStage::Compute: {
                ValueId tg = V::kNoValue;
                for (unsigned k = 0; k < 3; ++k)
                    if (in.tgidEnable[k]) {
                        if (tg == V::kNoValue) tg = builtinIn(V::Builtin::WorkgroupId, V::vec(U, 3), "workgroup_id");
                        b->storeVar(sVar(s++), b->extract(tg, k, U));
                    }
                if (in.tgSizeEnable) {
                    note("TG_SIZE SGPR holds the wave count only");
                    uint32_t lanes = in.workgroupSize[0] * in.workgroupSize[1] * in.workgroupSize[2];
                    b->storeVar(sVar(s++), u((lanes + 63) / 64));
                }
                ValueId lid = builtinIn(V::Builtin::LocalInvocationId, V::vec(U, 3), "local_id");
                laneIndex = bin(V::Op::And, builtinIn(V::Builtin::LocalInvocationIndex, U, "local_index"), u(63), U);
                quadLane = bin(V::Op::And, laneIndex, u(3), U);
                for (unsigned k = 0; k <= in.tidigCompCnt && k < 3; ++k) b->storeVar(vVar(k), b->extract(lid, k, U));
                break;
            }
        }
    }

    // ---- Driver --------------------------------------------------------------------------------

    void run() {
        m.stage = in.stage == ShaderStage::Vertex ? V::Stage::Vertex : in.stage == ShaderStage::Pixel ? V::Stage::Fragment : V::Stage::Compute;
        m.waveSize = 64;
        if (in.stage == ShaderStage::Compute) {
            for (unsigned k = 0; k < 3; ++k) m.workgroupSize[k] = std::max(1u, in.workgroupSize[k]);
            m.sharedBytes = in.ldsBytes;
        }
        if (in.code.empty()) return error("empty program");
        if (!decodeProgram(in.code, false, prog)) return;
        if (!in.fetch.empty()) {
            if (!decodeProgram(in.fetch, true, fetchProg)) return;
        }
        buildBlocks();
        mainCount = prog.size();
        for (Decoded& d : fetchProg) {
            fetchIndex.push_back(prog.size());
            prog.push_back(d);
        }
        buildOrder();
        for (const Decoded& d : prog) {
            auto see = [&](const Operand& o) {
                if (o.kind == OperandKind::Vgpr) highestVgpr = std::max(highestVgpr, unsigned(o.index + o.count - 1));
            };
            see(d.in.dst);
            for (const Operand& o : d.in.src) see(o);
        }
        highestVgpr = std::min(highestVgpr, 255u);
        findMasks();
        for (const Decoded& d : prog) {
            const Instruction& x = d.in;
            if ((x.dst.kind == OperandKind::Special && (x.dst.code == kExec || x.dst.code == kExec + 1)) ||
                (x.sdst.kind == OperandKind::Special && x.sdst.code == kExec) ||
                mnemonic(x.op).starts_with("v_cmpx") || mnemonic(x.op).starts_with("v_cmpsx") || isSaveexec(x.op))
                writesExec = true;
        }
        traceOrigins();
        // Images written anywhere are storage textures everywhere.
        for (size_t i = 0; i < prog.size(); ++i) {
            const Instruction& x = prog[i].in;
            std::string_view n = mnemonic(x.op);
            if (x.encoding == Encoding::MIMG && (n.starts_with("image_store") || n.starts_with("image_atomic")))
                if (auto it = rsrcSource.find(i); it != rsrcSource.end())
                    storageUses.insert(it->second);
        }
        bool structured = false;
        std::vector<int> ipdom;
        if (findLoops() && dagAcyclic()) {
            ipdom = postDominators();
            if (std::getenv("VPGCN_DEBUG"))
                for (size_t k = 0; k < blocks.size(); ++k)
                    std::fprintf(stderr, "block %zu [%u..%u] term %d taken %d next %d ipdom %d\n", k, prog[blocks[k].first].pc,
                                 prog[blocks[k].last].pc, int(blocks[k].term), blocks[k].taken, blocks[k].next, ipdom[k]);
            Walk dry;
            dry.budget = 4 * prog.size() + 64;
            walk(0, -2, ipdom, dry);
            structured = dry.ok;
            if (!structured) note("branch structure does not nest (or duplicates too much code): dispatch loop used");
        } else if (blocks.size() > 1) {
            note("irreducible or multi-exit loops: dispatch loop used");
        }
        V::FunctionBuilder fb(m, "main");
        b = &fb;
        entryState();
        if (structured) {
            Walk emit;
            emit.dry = false;
            emit.budget = SIZE_MAX;
            walk(0, -2, ipdom, emit);
        }
        else dispatchLoop();
        fb.finish();
        b = nullptr;
        out.structured = structured;
    }
};

LiftResult lift(const LiftInput& input) {
    LiftResult out;
    Lifter l(input, out);
    l.run();
    out.errors.assign(l.errorSet.begin(), l.errorSet.end());
    out.notes.assign(l.noteSet.begin(), l.noteSet.end());
    return out;
}

}  // namespace vpgcn
