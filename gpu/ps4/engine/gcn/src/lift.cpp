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
#include "vp4ir/builder.h"

namespace vpgcn {
namespace {

namespace V = vp4ir;
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
    bool usesVskip = false;        // the program has s_setvskip: vector instructions test VSKIP
    int64_t vskipVar = -1;
    // MODE register (rounding, denormals, DX10 clamp, IEEE): `modeVar` holds it at run time, `curMode` is
    // what the lifter knows statically (set from the entry state and s_setreg_imm32_b32 in lifting order).
    // `modeDynamic`: an s_setreg_b32 with a register source exists, so denormal flushing tests the variable.
    int64_t modeVar = -1;
    uint32_t curMode = 0x1c0;
    bool modeDynamic = false;
    bool flushOps = false;         // the op being lifted is a float ALU op subject to the denormal mode
    int64_t timeVar = -1;          // s_memtime: synthetic monotonic counter
    bool usesMode = false;         // s_getreg / s_setreg exist: MODE is kept in a variable
    bool usesTime = false;
    G curOp = G::Count;            // the instruction being lifted
    unsigned highestSgpr = 0;

    // Descriptor tracing: per instruction (index in prog), the source of its descriptor operands.
    std::unordered_map<size_t, DescriptorSource> rsrcSource, sampSource;
    std::unordered_map<size_t, DescriptorSource> pointerSource;  // s_load_*: the pointer's source
    std::set<size_t> rawBuffer;  // buffer accesses whose V# is built from an address in memory

    // Resource slots.
    std::map<std::pair<int, DescriptorSource>, uint32_t> slots;
    std::optional<uint32_t> userSlot;
    std::set<DescriptorSource> storageUses;  // T# sources written by the program
    std::map<DescriptorSource, bool> atomicUses;  // T# sources of image atomics -> any signed (smin/smax)

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

    // Wave mode: the workgroup is one wave (compute, at most 64 invocations, one per lane) and runs in lockstep, as the
    // hardware does. Lane masks have a real 64-bit dword form (a ballot), cross-lane ops exchange values, and every
    // branch on EXEC / VCC is wave-wide. Chosen by lift() when the first, per-lane pass met something that needs it
    // (wantsWave).
    bool waveMode = false;
    bool wantsWave = false;
    void needWave() { wantsWave = true; }

    Lifter(const LiftInput& i, LiftResult& o, bool wave = false) : in(i), out(o), m(o.module), waveMode(wave) { vgprVar.fill(-1); }

    std::set<uint32_t> padding;  // dwords where a path ran into zero padding (main program)

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
            std::map<uint16_t, uint32_t> konst;  // SGPR -> constant (program-image addresses), along this path
            while (pc < code.size() && !done.count(pc)) {
                done.insert(pc);
                Decoded d;
                d.pc = pc;
                d.fetch = fetch;
                if (!decode(code.subspan(pc), d.in)) {
                    // Zero words after the last instruction are the binary's alignment padding:
                    // a path that falls into them is one the program never takes (it runs past
                    // its end on the console too). End it there instead of rejecting the shader.
                    if (code[pc] == 0 && !found.empty() && !fetch) {
                        auto prev = found.lower_bound(pc);
                        std::string after = prev == found.begin() ? std::string("?")
                                                                  : std::string(mnemonic(std::prev(prev)->second.in.op));
                        note("a path falls into zero padding (after " + after + "): ended there");
                        padding.insert(pc);
                        break;
                    }
                    char buf[64];
                    std::snprintf(buf, sizeof buf, "undecodable word 0x%08x at dword %u", code[pc], pc);
                    error(buf);
                    return false;
                }
                found[pc] = d;
                G op = d.in.op;
                uint32_t next = pc + d.in.size / 4;
                if (!fetch) trackConstants(d, konst);
                if (!fetch && op == G::s_cbranch_g_fork) {
                    // The target pair was built from s_getpc_b64 and constants.
                    const Operand& t = d.in.src[1];
                    auto it = t.kind == OperandKind::Sgpr ? konst.find(t.code) : konst.end();
                    if (it == konst.end()) {
                        error("s_cbranch_g_fork with a target that is not a constant program address");
                        return false;
                    }
                    forkTarget[pc] = it->second / 4;
                }
                if (isBranchInstr(d)) {
                    uint32_t target = 0;
                    branchTarget(d, target);
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
            case G::s_cbranch_execnz:
            case G::s_cbranch_g_fork:
            case G::s_cbranch_i_fork: return true;
            default: return false;
        }
    }
    static bool isFork(G op) { return op == G::s_cbranch_g_fork || op == G::s_cbranch_i_fork; }

    // Targets of s_cbranch_g_fork (the address is in an SGPR pair): pc -> dword offset, found while decoding by
    // following the constants s_getpc_b64 / s_add_u32 / s_addc_u32 / s_mov build into the pair.
    std::map<uint32_t, uint32_t> forkTarget;

    // The dword a branch instruction jumps to, if it is one (a fork whose target is unknown: false).
    bool branchTarget(const Decoded& d, uint32_t& target) const {
        const Instruction& x = d.in;
        uint32_t next = d.pc + x.size / 4;
        if (x.encoding == Encoding::SOPP && isBranch(x.op)) {
            target = uint32_t(int32_t(next) + int16_t(x.simm16));
            return true;
        }
        if (x.op == G::s_cbranch_i_fork) {
            target = uint32_t(int32_t(next) + int16_t(x.simm16));
            return true;
        }
        if (x.op == G::s_cbranch_g_fork) {
            auto it = forkTarget.find(d.pc);
            if (it == forkTarget.end()) return false;
            target = it->second;
            return true;
        }
        return false;
    }
    bool isBranchInstr(const Decoded& d) const { return isBranch(d.in.op) && (d.in.encoding == Encoding::SOPP || isFork(d.in.op)); }

    // Constants held in SGPRs along a decoding path: s_getpc_b64 (the next instruction's byte address), moves
    // and adds of constants (what a fork's target is built from).
    void trackConstants(const Decoded& d, std::map<uint16_t, uint32_t>& konst) {
        const Instruction& x = d.in;
        if (x.dst.kind != OperandKind::Sgpr) return;
        auto drop = [&] {
            for (unsigned k = 0; k < std::max<unsigned>(1, x.dst.count); ++k) konst.erase(uint16_t(x.dst.code + k));
        };
        auto val = [&](const Operand& o, uint32_t& v) {
            if (o.kind == OperandKind::Const || o.kind == OperandKind::Literal) {
                v = o.value;
                return true;
            }
            auto it = o.kind == OperandKind::Sgpr ? konst.find(o.code) : konst.end();
            if (it == konst.end()) return false;
            v = it->second;
            return true;
        };
        if (x.op == G::s_getpc_b64) {
            konst[x.dst.code] = (d.pc + x.size / 4) * 4;
            konst[uint16_t(x.dst.code + 1)] = 0;
            return;
        }
        uint32_t a = 0, c = 0;
        switch (x.op) {
            case G::s_mov_b32:
                if (val(x.src[0], a)) konst[x.dst.code] = a;
                else drop();
                return;
            case G::s_mov_b64:
                if (val(x.src[0], a)) {
                    konst[x.dst.code] = a;
                    konst[uint16_t(x.dst.code + 1)] = 0;
                } else {
                    drop();
                }
                return;
            case G::s_add_u32:
            case G::s_sub_u32:
                if (val(x.src[0], a) && val(x.src[1], c)) konst[x.dst.code] = x.op == G::s_add_u32 ? a + c : a - c;
                else drop();
                return;
            case G::s_addc_u32:
                // (the carry of a program address plus a small offset is 0)
                if (val(x.src[0], a) && val(x.src[1], c)) konst[x.dst.code] = a + c;
                else drop();
                return;
            default: drop(); return;
        }
    }

    void buildBlocks() {
        std::set<uint32_t> leaders{0};
        for (const Decoded& d : prog) {
            uint32_t next = d.pc + d.in.size / 4;
            if (isBranchInstr(d)) {
                uint32_t t = 0;
                if (branchTarget(d, t)) leaders.insert(t);
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
            } else if (isBranchInstr(d)) {
                uint32_t t = 0;
                branchTarget(d, t);
                bb.taken = at(t);
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
                if (bb.next < 0 && !d.fetch && padding.count(next)) continue;  // see decodeProgram
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
            if (x.op == G::s_cbranch_g_fork) mark(x.src[0]);
            if (x.op == G::s_cbranch_i_fork) mark(x.dst);
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
        auto step = [&](size_t i) {
            // A revisit (see the block walk below) recomputes this instruction's sources.
            rsrcSource.erase(i);
            sampSource.erase(i);
            pointerSource.erase(i);
            rawBuffer.erase(i);
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
        };
        // Forward dataflow over the main program's blocks (the fetch subroutine runs inline at its
        // s_swappc_b64): a block starts from the meet of its predecessors' ends, so a branch target
        // sees the descriptors set up before the branch even when the fall-through path reuses
        // those SGPRs (a straight program-order walk would see the fall-through's clobbers).
        auto meet = [](std::array<Origin, 140>& a, const std::array<Origin, 140>& b2) {
            bool changed = false;
            for (size_t k = 0; k < a.size(); ++k)
                if (a[k].kind != Origin::Unknown &&
                    (a[k].kind != b2[k].kind || a[k].sgpr != b2[k].sgpr || a[k].path != b2[k].path)) {
                    a[k] = {};
                    changed = true;
                }
            return changed;
        };
        const std::array<Origin, 140> entry = o;
        const int nb = int(blocks.size());
        std::vector<std::optional<std::array<Origin, 140>>> blockIn(nb);
        std::vector<int> work;
        for (int k = 0; k < nb; ++k)
            if (blocks[k].first == 0) {
                blockIn[k] = entry;
                work.push_back(k);
            }
        if (work.empty()) {
            for (size_t i : order) step(i);
            return;
        }
        unsigned guard = 0;
        while (!work.empty() && guard++ < 100000) {
            int k = work.back();
            work.pop_back();
            o = *blockIn[k];
            for (size_t i = blocks[k].first; i <= blocks[k].last && i < mainCount; ++i) {
                step(i);
                if (prog[i].in.op == G::s_swappc_b64)
                    for (size_t f : fetchIndex) step(f);
            }
            for (int t : successors(k)) {
                if (t < 0 || t >= nb || blocks[t].first >= mainCount) continue;
                if (!blockIn[t]) {
                    blockIn[t] = o;
                    work.push_back(t);
                } else if (meet(*blockIn[t], o)) {
                    work.push_back(t);
                }
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
            if (waveMode) return laneBit(readS(o, 0), readS(o, 1));
            note("a lane mask constant other than 0 / -1 is read as all lanes");
            needWave();
            return b->constBool(true);
        }
        if (o.code == kExec) return b->loadVar(execV());
        if (hasMaskForm(o.code)) return b->loadVar(maskBool(o.code));
        if (o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp) {
            ValueId lo = b->loadVar(sVar(o.code)), hi = b->loadVar(sVar(o.code + 1));
            if (waveMode) return laneBit(lo, hi);
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
            if (waveMode) return;  // (the dwords are the ballot of the bit, see readS)
            b->storeVar(sVar(o.code), b2u(v));
            b->storeVar(sVar(o.code + 1), u(0));
            return;
        }
        error("unsupported lane mask destination");
    }

    // The 64 bits of a lane mask as two dwords (bit n = lane n): a ballot over the wave. Needs the wave to be
    // one 64-thread compute workgroup (waveMode); elsewhere a mask has no dword form and the draw is rejected.
    bool maskWords(const Operand& o, ValueId& lo, ValueId& hi) {
        if (!waveMode) {
            needWave();
            error(std::string(mnemonic(curOp)) + " of a lane mask is only supported in compute shaders of at most 64 threads");
            lo = hi = u(0);
            return false;
        }
        ballotWords(readMask(o), lo, hi);
        return true;
    }

    // ---- Wave mode primitives -------------------------------------------------------------------

    // The 64 lane bits of `cond` as two dwords (bit n = lane n's value; lanes that do not exist read 0).
    void ballotWords(ValueId cond, ValueId& lo, ValueId& hi) {
        ValueId ballot = b->emit(V::Inst{V::Op::SubgroupBallot, V::vec(V::kU32, 4), V::kNoValue, false, {cond}, 0, 0, {}, 0});
        lo = b->extract(ballot, 0, V::kU32);
        hi = b->extract(ballot, 1, V::kU32);
    }
    ValueId ballotWord(ValueId cond, unsigned half) {
        ValueId lo, hi;
        ballotWords(cond, lo, hi);
        return half ? hi : lo;
    }
    // Whether any lane has `cond` (a wave-wide condition: what an s_cbranch on VCC / EXEC tests).
    ValueId anyLane(ValueId cond) {
        return b->emit(V::Inst{V::Op::SubgroupAny, V::kBool, V::kNoValue, false, {cond}, 0, 0, {}, 0});
    }
    // This lane's bit of the 64-bit value {lo, hi}.
    ValueId laneBit(ValueId lo, ValueId hi) {
        ValueId lane = bin(V::Op::And, laneIndex, u(63), V::kU32);
        ValueId upper = bin(V::Op::Ge, lane, u(32), V::kBool);
        ValueId word = sel(upper, hi, lo, V::kU32);
        return nz(bin(V::Op::And, bin(V::Op::Shr, word, bin(V::Op::And, lane, u(31), V::kU32), V::kU32), u(1), V::kU32));
    }
    // SCC after an instruction that sets it to "the 64-bit result is nonzero": for a lane mask, any lane.
    void setSccMask(ValueId r) { setScc(waveMode ? anyLane(r) : r); }
    // A lane mask read as dwords by an instruction that treats it as data: true when it must be treated as a
    // per-lane bit instead (outside wave mode, with an approximation; it asks for wave mode on the way).
    bool maskApprox(const Operand& o) {
        if (!isPairMask(o)) return false;
        if (waveMode) return false;
        needWave();
        return true;
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
            case OperandKind::Ttmp:
                if (waveMode && hasMaskForm((o.code + half) & ~1u)) return ballotWord(b->loadVar(maskBool((o.code + half) & ~1u)), (o.code + half) & 1);
                return b->loadVar(sVar(o.code + half));
            case OperandKind::Special: {
                uint32_t code = o.code + half;
                switch (code) {
                    case kM0: return b->loadVar(m0V());
                    case kVcc:
                    case kVcc + 1:
                        if (waveMode) return ballotWord(b->loadVar(vccV()), code - kVcc);
                        return b->loadVar(sVar(code));
                    case kExec:
                    case kExec + 1:
                        if (waveMode) return ballotWord(b->loadVar(execV()), code - kExec);
                        return code == kExec ? b2u(b->loadVar(execV())) : u(0);
                    case 251:
                        if (waveMode) return b2u(notB(anyLane(b->loadVar(vccV()))));
                        return b2u(notB(b->loadVar(vccV())));
                    case 252:
                        if (waveMode) return b2u(notB(anyLane(b->loadVar(execV()))));
                        return b2u(notB(b->loadVar(execV())));
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
        if (waveMode && (sgpr || code == kVcc || code == kVcc + 1) && hasMaskForm(code & ~1u)) {
            // A dword of a lane mask: the other dword stays, the lane's bit is recomputed.
            uint32_t pair = code & ~1u;
            bool hiHalf = code & 1;
            ValueId keep = ballotWord(b->loadVar(maskBool(pair)), hiHalf ? 0 : 1);
            b->storeVar(maskBool(pair), hiHalf ? laneBit(keep, v) : laneBit(v, keep));
            return;
        }
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
        if (waveMode && (code == kExec || code == kExec + 1)) {
            writesExec = true;
            ValueId keep = ballotWord(b->loadVar(execV()), code == kExec ? 1 : 0);
            b->storeVar(execV(), code == kExec ? laneBit(v, keep) : laneBit(keep, v));
            return;
        }
        if (code == kExec) {
            note("a 32-bit write to exec_lo becomes this lane's bit");
            needWave();
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
            // Metal cannot write, or run atomics on, a cube texture: storage cubes are declared as 2D arrays of
            // their faces (layer = slice * 6 + face), which the client binds as an array view of the cube.
            if (kind == V::ResourceKind::StorageTexture && (r.dim == V::TexDim::Cube || r.dim == V::TexDim::CubeArray)) {
                note("a cube texture used with image load / store / atomics is declared as a 2D array of faces");
                r.dim = V::TexDim::D2Array;
            }
            // Image atomics are integer: the texture is declared with integer texels (R32 data keeps
            // its bits; loads and stores of it move them unchanged).
            if (auto at = atomicUses.find(s); at != atomicUses.end() && kind == V::ResourceKind::StorageTexture) {
                V::Scalar want = at->second ? V::Scalar::I32 : V::Scalar::U32;
                if (r.elem != want) note("a texture used with image atomics is declared with 32-bit integer texels");
                r.elem = want;
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

    // ---- The MODE register ----------------------------------------------------------------------
    uint32_t modeV() { return var(modeVar, V::kU32); }
    uint32_t vskipV() { return var(vskipVar, V::kBool); }

    // An f32 value with denormals flushed to a zero of the same sign when MODE's FP_DENORM bit `allowBit`
    // (0x10: inputs are allowed, 0x20: outputs are allowed) is clear. Known statically unless the program
    // writes MODE from a register.
    ValueId flushDenorm(ValueId v, uint32_t allowBit) {
        if (!modeDynamic && (curMode & allowBit)) return v;
        ValueId tiny = bin(V::Op::Lt, b->unary(V::Op::Abs, v, V::kF32), f(1.17549435e-38f), V::kBool);
        // (selected as integer bits: a float select may canonicalise a NaN's payload)
        ValueId raw = bit(v, V::kU32);
        ValueId flushed = sel(tiny, bin(V::Op::And, raw, u(0x80000000u), V::kU32), raw, V::kU32);
        if (!modeDynamic) return asF(flushed);
        ValueId allow = nz(bin(V::Op::And, b->loadVar(modeV()), u(allowBit), V::kU32));
        return asF(sel(allow, raw, flushed, V::kU32));
    }

    // Float source i with VOP3 input modifiers.
    ValueId srcF(const Instruction& x, unsigned i) {
        ValueId v = asF(readV(x.src[i]));
        if (x.abs >> i & 1) v = b->unary(V::Op::Abs, v, V::kF32);
        if (x.neg >> i & 1) v = b->unary(V::Op::Neg, v, V::kF32);
        if (flushOps) v = flushDenorm(v, 0x10);
        return v;
    }
    ValueId srcU(const Instruction& x, unsigned i) { return readV(x.src[i]); }
    ValueId srcI(const Instruction& x, unsigned i) { return asI(readV(x.src[i])); }

    void resultF(const Instruction& x, ValueId v) {
        if (flushOps) v = flushDenorm(v, 0x20);
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
        ValueId r = bit(pair, V::kU32);
        // A NaN keeps its sign and the top ten bits of its payload, quieted (the conversion's own NaN is the
        // canonical one on some targets).
        auto U = V::kU32;
        ValueId raw = bit(fv, U);
        ValueId nanBits = bin(V::Op::Or,
                              bin(V::Op::Or, bin(V::Op::And, bin(V::Op::Shr, raw, u(16), U), u(0x8000u), U), u(0x7e00u), U),
                              bin(V::Op::And, bin(V::Op::Shr, raw, u(13), U), u(0x3ffu), U), U);
        return sel(b->unary(V::Op::IsNan, fv, V::kBool), nanBits, r, U);
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
    // 64-bit shift of {hi, lo} by sh (0..63): kind 0 left, 1 logical right, 2 arithmetic right.
    std::pair<ValueId, ValueId> shift64(ValueId lo, ValueId hi, ValueId sh, int kind) {
        auto U = V::kU32;
        auto I = V::kI32;
        ValueId big = bin(V::Op::Ge, sh, u(32), V::kBool);
        ValueId s = bin(V::Op::And, sh, u(31), U), inv = bin(V::Op::And, bin(V::Op::Sub, u(32), s, U), u(31), U);
        ValueId zeroS = bin(V::Op::Eq, s, u(0), V::kBool);
        if (kind == 0) {
            ValueId carry = sel(zeroS, u(0), bin(V::Op::Shr, lo, inv, U), U);
            ValueId rhi = sel(big, bin(V::Op::Shl, lo, s, U), bin(V::Op::Or, bin(V::Op::Shl, hi, s, U), carry, U), U);
            return {sel(big, u(0), bin(V::Op::Shl, lo, s, U), U), rhi};
        }
        ValueId carry = sel(zeroS, u(0), bin(V::Op::Shl, hi, inv, U), U);
        auto shrHi = [&](ValueId amount) {
            return kind == 2 ? bit(bin(V::Op::Shr, bit(hi, I), bit(amount, I), I), U) : bin(V::Op::Shr, hi, amount, U);
        };
        ValueId fill = kind == 2 ? bit(bin(V::Op::Shr, bit(hi, I), b->constI32(31), I), U) : u(0);
        ValueId rlo = sel(big, shrHi(s), bin(V::Op::Or, bin(V::Op::Shr, lo, s, U), carry, U), U);
        return {rlo, sel(big, fill, shrHi(s), U)};
    }

    // A byte / short store into dword-addressed memory (buffer or LDS): two atomics on the dword,
    // clear the field then set it, so lanes writing other bytes of the same dword do not lose
    // theirs (a plain read-modify-write would race).
    void storeSubDword(uint32_t target, ValueId addr, ValueId value, unsigned width) {
        auto U = V::kU32;
        ValueId word = bin(V::Op::And, addr, u(~3u), U);
        ValueId sh = bin(V::Op::Shl, bin(V::Op::And, addr, u(3), U), u(3), U);
        uint32_t low = width == 8 ? 0xffu : 0xffffu;
        ValueId mask = bin(V::Op::Shl, u(low), sh, U);
        ValueId field = bin(V::Op::Shl, bin(V::Op::And, value, u(low), U), sh, U);
        ValueId keep[] = {b->unary(V::Op::Not, mask, U)};
        b->atomic(V::Op::AtomicAnd, target, word, keep, U);
        ValueId set[] = {field};
        b->atomic(V::Op::AtomicOr, target, word, set, U);
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

    // ---- Byte SADs, division helpers ---------------------------------------------------------------
    ValueId absDiffU(ValueId a, ValueId c) {
        auto U = V::kU32;
        return sel(bin(V::Op::Gt, a, c, V::kBool), bin(V::Op::Sub, a, c, U), bin(V::Op::Sub, c, a, U), U);
    }
    // Sum of the absolute differences of the four bytes of a and c; masked: bytes where c is 0 count 0.
    ValueId sad4(ValueId a, ValueId c, bool masked) {
        auto U = V::kU32;
        ValueId sum = u(0);
        for (unsigned k = 0; k < 4; ++k) {
            ValueId x = bin(V::Op::And, bin(V::Op::Shr, a, u(8 * k), U), u(0xff), U);
            ValueId y = bin(V::Op::And, bin(V::Op::Shr, c, u(8 * k), U), u(0xff), U);
            ValueId d = absDiffU(x, y);
            if (masked) d = sel(bin(V::Op::Eq, y, u(0), V::kBool), u(0), d, U);
            sum = bin(V::Op::Add, sum, d, U);
        }
        return sum;
    }
    ValueId isDenormalF(ValueId v) {
        auto U = V::kU32;
        ValueId bits = bit(v, U);
        return bin(V::Op::And, bin(V::Op::Eq, bin(V::Op::And, bits, u(0x7f800000u), U), u(0), V::kBool),
                   bin(V::Op::Ne, bin(V::Op::And, bits, u(0x7fffffu), U), u(0), V::kBool), V::kBool);
    }
    ValueId expField(ValueId v) { return bin(V::Op::And, bin(V::Op::Shr, bit(v, V::kU32), u(23), V::kU32), u(0xff), V::kU32); }
    // ldexp(v, 64) done on the bits, so denormal inputs (the case the division scaling exists for) are exact.
    ValueId ldexpUp64(ValueId v) {
        auto U = V::kU32;
        ValueId bits = bit(v, U);
        ValueId sign = bin(V::Op::And, bits, u(0x80000000u), U), mag = bin(V::Op::And, bits, u(0x7fffffffu), U);
        ValueId e = bin(V::Op::Shr, mag, u(23), U), frac = bin(V::Op::And, mag, u(0x7fffffu), U);
        // normal: exponent + 64, to infinity from 255 - 64; zero, infinity and NaN stay
        ValueId normal = sel(bin(V::Op::Ge, e, u(255 - 64), V::kBool), u(0x7f800000u), bin(V::Op::Add, mag, u(64u << 23), U), U);
        // denormal: normalised from the highest set bit p: biased exponent p + 42, fraction shifted up
        ValueId p = b->unary(V::Op::FindMsb, frac, U);
        ValueId denorm = bin(V::Op::Or, bin(V::Op::Shl, bin(V::Op::Add, p, u(42), U), u(23), U),
                             bin(V::Op::And, bin(V::Op::Shl, frac, bin(V::Op::And, bin(V::Op::Sub, u(23), p, U), u(31), U), U), u(0x7fffffu), U), U);
        ValueId special = bin(V::Op::Or, bin(V::Op::Eq, mag, u(0), V::kBool), bin(V::Op::Eq, e, u(255), V::kBool), V::kBool);
        ValueId r = sel(bin(V::Op::Eq, e, u(0), V::kBool), denorm, normal, U);
        return asF(bin(V::Op::Or, sign, sel(special, mag, r, U), U));
    }

    // v_div_scale_f32: D = src0 (the denominator src1 or the numerator src2), scaled by 2^+-64 when the
    // quotient or the reciprocal would be denormal or out of range; VCC says whether the operands were scaled
    // (v_div_fmas then rescales). Cases in the order of the ISA's pseudo code.
    void divScale(const Instruction& x) {
        auto F = V::kF32;
        auto I = V::kI32;
        ValueId s0 = srcF(x, 0), s1 = srcF(x, 1), s2 = srcF(x, 2);
        ValueId zero = bin(V::Op::Or, bin(V::Op::Eq, s2, f(0), V::kBool), bin(V::Op::Eq, s1, f(0), V::kBool), V::kBool);
        ValueId e1 = expField(s1), e2 = expField(s2);
        ValueId big = bin(V::Op::Ge, bin(V::Op::Sub, asI(e2), asI(e1), I), b->constI32(96), V::kBool);
        ValueId den1 = isDenormalF(s1);
        ValueId rdn = isDenormalF(bin(V::Op::Div, f(1.0f), s1, F)), qdn = isDenormalF(bin(V::Op::Div, s2, s1, F));
        ValueId tiny = bin(V::Op::Le, e2, u(23), V::kBool);
        ValueId eq1 = bin(V::Op::Eq, s0, s1, V::kBool), eq2 = bin(V::Op::Eq, s0, s2, V::kBool);
        ValueId up = ldexpUp64(s0), down = bin(V::Op::Mul, s0, f(5.421010862427522e-20f), F);  // 2^-64
        ValueId d = s0, vcc = b->constBool(false), open = b->constBool(true);
        auto branch = [&](ValueId cond, ValueId value, bool setsVcc) {
            ValueId hit = bin(V::Op::And, open, cond, V::kBool);
            d = sel(hit, value, d, F);
            if (setsVcc) vcc = bin(V::Op::Or, vcc, hit, V::kBool);
            open = bin(V::Op::And, open, notB(cond), V::kBool);
        };
        branch(zero, asF(u(0xffc00000u)), false);
        branch(big, sel(eq1, up, s0, F), true);
        branch(den1, up, false);
        branch(bin(V::Op::And, rdn, qdn, V::kBool), sel(eq1, up, s0, F), true);
        branch(rdn, down, false);
        branch(qdn, sel(eq2, up, s0, F), true);
        branch(tiny, up, false);
        // (a NaN or infinity operand matches none of these but the exponent difference ones, as on the hardware)
        writeMask(x.sdst, bin(V::Op::And, vcc, b->loadVar(execV()), V::kBool));
        resultF(x, d);
    }

    // v_div_fixup_f32: the final answer from the quotient estimate src0, denominator src1 and numerator src2:
    // NaN, infinity and zero cases of n / d, else |src0| with the sign of n / d.
    void divFixup(const Instruction& x) {
        auto U = V::kU32;
        auto F = V::kF32;
        ValueId q = srcF(x, 0), den = srcF(x, 1), num = srcF(x, 2);
        ValueId sign = bin(V::Op::And, bin(V::Op::Xor, bit(den, U), bit(num, U), U), u(0x80000000u), U);
        ValueId nanDen = b->unary(V::Op::IsNan, den, V::kBool), nanNum = b->unary(V::Op::IsNan, num, V::kBool);
        ValueId zeroDen = bin(V::Op::Eq, den, f(0), V::kBool), zeroNum = bin(V::Op::Eq, num, f(0), V::kBool);
        ValueId infDen = b->unary(V::Op::IsInf, den, V::kBool), infNum = b->unary(V::Op::IsInf, num, V::kBool);
        ValueId r = bin(V::Op::Or, bin(V::Op::And, bit(q, U), u(0x7fffffffu), U), sign, U);
        r = sel(bin(V::Op::Or, infDen, zeroNum, V::kBool), sign, r, U);                                   // +-0
        r = sel(bin(V::Op::Or, zeroDen, infNum, V::kBool), bin(V::Op::Or, sign, u(0x7f800000u), U), r, U);  // +-inf
        r = sel(bin(V::Op::And, infDen, infNum, V::kBool), u(0xffc00000u), r, U);
        r = sel(bin(V::Op::And, zeroDen, zeroNum, V::kBool), u(0xffc00000u), r, U);
        r = sel(nanDen, bin(V::Op::Or, bit(den, U), u(0x00400000u), U), r, U);
        r = sel(nanNum, bin(V::Op::Or, bit(num, U), u(0x00400000u), U), r, U);
        resultF(x, asF(r));
        (void)F;
    }

    void liftVop(size_t idx) {
        const Instruction& x = prog[idx].in;
        G op = x.op;
        V::Op Add = V::Op::Add, Sub = V::Op::Sub, Mul = V::Op::Mul;
        auto F = V::kF32;
        auto U = V::kU32;
        auto I = V::kI32;
        std::string_view name = mnemonic(op);
        // Float ALU ops obey the MODE register's f32 denormal handling (moves, selects and class tests do not).
        flushOps = name.ends_with("_f32") && name.find("class") == std::string_view::npos && !name.starts_with("v_div_");
        struct ResetFlush {
            bool& f;
            ~ResetFlush() { f = false; }
        } resetFlush{flushOps};
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
            case G::v_clrexcp: return;  // no exception state is kept
            case G::v_mov_fed_b32:  // (a move that skips error-detection on the hardware's data path)
            case G::v_mov_b32:
                if (x.abs || x.neg) return resultF(x, srcF(x, 0));  // a float move with |x| / -x
                return resultU(x, srcU(x, 0));
            case G::v_readfirstlane_b32:
            case G::v_readlane_b32: {
                needWave();
                if (waveMode) {
                    ValueId lane;
                    if (op == G::v_readlane_b32) {
                        lane = bin(V::Op::And, srcU(x, 1), u(63), U);
                    } else {
                        // The first lane with EXEC set (lane 0 when none is).
                        ValueId elo, ehi;
                        ballotWords(b->loadVar(execV()), elo, ehi);
                        ValueId inLo = b->unary(V::Op::FindLsb, elo, U);
                        ValueId inHi = bin(V::Op::Add, b->unary(V::Op::FindLsb, ehi, U), u(32), U);
                        lane = sel(nz(elo), inLo, sel(nz(ehi), inHi, u(0), U), U);
                    }
                    writeS(x.dst, b->emit(V::Inst{V::Op::SubgroupShuffle, U, V::kNoValue, false, {srcU(x, 0), lane}, 0, 0, {}, 0}));
                    return;
                }
                if (op == G::v_readlane_b32 && x.src[0].kind == OperandKind::Vgpr && spillBase.count(x.src[0].index)) {
                    // A register that holds spilled scalars (v_writelane): its lane k is slot k of its shadow array.
                    ValueId lane = bin(V::Op::And, srcU(x, 1), u(63), U);
                    ValueId at = bin(V::Op::Add, u(spillBase[x.src[0].index]), bin(V::Op::Shl, lane, u(2), U), U);
                    writeS(x.dst, b->load(V::kTargetPrivate, at, U));
                    return;
                }
                if (op == G::v_readlane_b32) note("v_readlane_b32 reads this lane's own value");
                writeS(x.dst, srcU(x, 0));
                return;
            }
            case G::v_writelane_b32: {
                needWave();
                if (waveMode) {
                    // One lane of the destination takes the scalar, whatever EXEC says; the others keep theirs.
                    ValueId target = bin(V::Op::And, srcU(x, 1), u(63), U);
                    ValueId mine = bin(V::Op::Eq, bin(V::Op::And, laneIndex, u(63), U), target, V::kBool);
                    b->storeVar(vVar(x.dst.index), sel(mine, srcU(x, 0), b->loadVar(vVar(x.dst.index)), U));
                    return;
                }
                if (spillBase.count(x.dst.index)) {
                    // Scalar spill to a VGPR lane (the value is wave-uniform, so each invocation keeps its own copy of
                    // the register's 64 lanes in private memory); EXEC does not apply to this instruction.
                    ValueId lane = bin(V::Op::And, srcU(x, 1), u(63), U);
                    ValueId at = bin(V::Op::Add, u(spillBase[x.dst.index]), bin(V::Op::Shl, lane, u(2), U), U);
                    b->store(V::kTargetPrivate, at, srcU(x, 0));
                    return;
                }
                note("v_writelane_b32 writes every lane");
                return resultU(x, srcU(x, 0));
            }
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
            case G::v_cvt_pk_u8_f32:
            case G::v_cvt_pkaccum_u8_f32: {
                // byte src1[1:0] of src2 (pk_u8) / of the destination (pkaccum) <- u8(src0), round to nearest even
                ValueId v = b->unary(V::Op::Convert, b->ternary(V::Op::Clamp, b->unary(V::Op::Round, srcF(x, 0), F), f(0), f(255), F), U);
                ValueId sh = bin(V::Op::Shl, bin(V::Op::And, srcU(x, 1), u(3), U), u(3), U);
                ValueId keep = b->unary(V::Op::Not, bin(V::Op::Shl, u(0xff), sh, U), U);
                ValueId acc = op == G::v_cvt_pk_u8_f32 ? srcU(x, 2) : b->loadVar(vVar(x.dst.index));
                return resultU(x, bin(V::Op::Or, bin(V::Op::And, acc, keep, U), bin(V::Op::Shl, v, sh, U), U));
            }
            case G::v_sad_u8:
            case G::v_sad_hi_u8:
            case G::v_msad_u8: {
                ValueId d = sad4(srcU(x, 0), srcU(x, 1), op == G::v_msad_u8);
                if (op == G::v_sad_hi_u8) d = bin(V::Op::Shl, d, u(16), U);
                return resultU(x, bin(Add, d, srcU(x, 2), U));
            }
            case G::v_sad_u16: {
                ValueId a = srcU(x, 0), c = srcU(x, 1);
                ValueId lo = absDiffU(bin(V::Op::And, a, u(0xffff), U), bin(V::Op::And, c, u(0xffff), U));
                ValueId hi = absDiffU(bin(V::Op::Shr, a, u(16), U), bin(V::Op::Shr, c, u(16), U));
                return resultU(x, bin(Add, bin(Add, lo, hi, U), srcU(x, 2), U));
            }
            case G::v_qsad_pk_u16_u8:
            case G::v_mqsad_pk_u16_u8:
            case G::v_mqsad_u32_u8: {
                // Four SADs of src1's four bytes against the 4-byte windows of the 64-bit src0 that start at
                // byte 0..3, each accumulated into a 16-bit (qsad_pk, mqsad_pk) or 32-bit (mqsad_u32) field of
                // src2. The m forms skip the bytes where src1 is 0.
                bool masked = op != G::v_qsad_pk_u16_u8;
                ValueId lo = readS(x.src[0], 0), hi = readS(x.src[0], 1), ref = srcU(x, 1);
                ValueId sad[4];
                for (unsigned k = 0; k < 4; ++k) {
                    ValueId w = k == 0 ? lo : bin(V::Op::Or, bin(V::Op::Shr, lo, u(8 * k), U), bin(V::Op::Shl, hi, u(32 - 8 * k), U), U);
                    sad[k] = sad4(w, ref, masked);
                }
                if (op == G::v_mqsad_u32_u8) {
                    ValueId sum[4];  // (every input is read before any result is written: dst may overlap src2)
                    for (unsigned k = 0; k < 4; ++k) sum[k] = bin(Add, sad[k], readS(x.src[2], k), U);
                    for (unsigned k = 0; k < 4; ++k) writeV(x.dst.index + k, sum[k]);
                    return;
                }
                ValueId acc[2] = {readS(x.src[2], 0), readS(x.src[2], 1)};
                ValueId out[2];
                for (unsigned h = 0; h < 2; ++h) {
                    ValueId a0 = bin(Add, sad[2 * h], bin(V::Op::And, acc[h], u(0xffff), U), U);
                    ValueId a1 = bin(Add, sad[2 * h + 1], bin(V::Op::Shr, acc[h], u(16), U), U);
                    out[h] = bin(V::Op::Or, bin(V::Op::And, a0, u(0xffff), U), bin(V::Op::Shl, bin(V::Op::And, a1, u(0xffff), U), u(16), U), U);
                }
                writeV(x.dst.index, out[0]);
                writeV(x.dst.index + 1, out[1]);
                return;
            }
            case G::v_mullit_f32: {
                // -FLT_MAX when src1 is -FLT_MAX, -inf or NaN, or src2 is <= 0 or NaN; else src0 * src1.
                ValueId a = srcF(x, 0), c = srcF(x, 1), d = srcF(x, 2);
                ValueId cNan = b->unary(V::Op::IsNan, c, V::kBool), dNan = b->unary(V::Op::IsNan, d, V::kBool);
                ValueId bad = bin(V::Op::Or, bin(V::Op::Le, c, f(-3.402823466e38f), V::kBool), cNan, V::kBool);
                bad = bin(V::Op::Or, bad, bin(V::Op::Or, bin(V::Op::Le, d, f(0), V::kBool), dNan, V::kBool), V::kBool);
                return resultF(x, sel(bad, f(-3.402823466e38f), bin(Mul, a, c, F), F));
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
            case G::v_lshl_b64:
            case G::v_lshr_b64:
            case G::v_ashr_i64: {
                // {dst+1, dst} = src0 (a pair) shifted by src1[5:0]
                auto [lo, hi] = shift64(readS(x.src[0], 0), readS(x.src[0], 1), bin(V::Op::And, srcU(x, 1), u(63), U),
                                        op == G::v_lshl_b64 ? 0 : op == G::v_lshr_b64 ? 1 : 2);
                writeV(x.dst.index, lo);
                writeV(x.dst.index + 1, hi);
                return;
            }
            case G::v_mad_u64_u32:
            case G::v_mad_i64_i32: {
                // {dst+1, dst} = src0 * src1 + {src2+1, src2}; carry-out of the 64-bit add to sdst.
                ValueId a = srcU(x, 0), c = srcU(x, 1);
                ValueId plo = bin(Mul, a, c, U), phi = op == G::v_mad_i64_i32 ? mulHiI(a, c) : mulHiU(a, c);
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
                if (waveMode && !all) {
                    // popcount of the mask's bits below this lane (lo: lanes 0-31, hi: lanes 32-63).
                    ValueId lane = bin(V::Op::And, laneIndex, u(63), U);
                    ValueId upper = bin(V::Op::Ge, lane, u(32), V::kBool);
                    ValueId below = bin(V::Op::Sub, bin(V::Op::Shl, u(1), bin(V::Op::And, lane, u(31), U), U), u(1), U);
                    ValueId sel0 = op == G::v_mbcnt_lo_u32_b32 ? sel(upper, u(0xffffffffu), below, U) : sel(upper, below, u(0), U);
                    ValueId cnt = b->unary(V::Op::BitCount, bin(V::Op::And, srcU(x, 0), sel0, U), U);
                    return resultU(x, bin(Add, cnt, srcU(x, 1), U));
                }
                if (!all || laneIndex == V::kNoValue) {
                    note("v_mbcnt with a partial mask or outside compute counts 0");
                    needWave();
                    return resultU(x, srcU(x, 1));
                }
                ValueId part = op == G::v_mbcnt_lo_u32_b32 ? bin(V::Op::Min, laneIndex, u(32), U)
                                                           : bin(V::Op::Sub, bin(V::Op::Max, laneIndex, u(32), U), u(32), U);
                return resultU(x, bin(Add, part, srcU(x, 1), U));
            }
            case G::v_movrels_b32:
            case G::v_movreld_b32:
            case G::v_movrelsd_b32: return movrel(x);
            case G::v_div_scale_f32: return divScale(x);
            case G::v_div_fmas_f32: {
                // fma, times 2^32 in the lanes whose VCC bit (set by v_div_scale) says the operands were scaled.
                ValueId r = b->ternary(V::Op::Fma, srcF(x, 0), srcF(x, 1), srcF(x, 2), F);
                return resultF(x, sel(b->loadVar(vccV()), bin(Mul, r, f(4294967296.0f), F), r, F));
            }
            case G::v_div_fixup_f32: return divFixup(x);
            default: break;
        }
        error("unsupported vector instruction " + std::string(name));
    }

    // Relative VGPR indexing by M0: movrels reads v[src + M0], movreld writes v[dst + M0], movrelsd
    // both. VGPRs are variables, so the index becomes a select chain (reads) or guarded stores
    // (writes) over the registers the program uses at or past the base.
    unsigned highestVgpr = 0;
    uint32_t ldsPrivEnd = 0;  // outside compute: bytes of the per-lane memory standing for LDS (privateLds)
    std::map<unsigned, uint32_t> spillBase;  // VGPRs written by v_writelane -> byte offset of their 64 lanes (private memory)
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
                setSccMask(r);
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
            setSccMask(r);
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
            case G::s_lshr_b64:
            case G::s_ashr_i64: {
                if (maskApprox(x.src[0])) return error(std::string(mnemonic(op)) + " of a lane mask");
                // 64-bit shift of a register pair by [5:0].
                auto [rlo, rhi] = shift64(s0(0), s0(1), bin(V::Op::And, s1(), u(63), U),
                                          op == G::s_lshl_b64 ? 0 : op == G::s_lshr_b64 ? 1 : 2);
                setScc(bin(V::Op::Or, nz(rlo), nz(rhi), V::kBool));
                put(rlo, 0);
                return put(rhi, 1);
            }
            case G::s_bfm_b64: {
                // ((1 << width) - 1) << offset, 64-bit: width s0[5:0], offset s1[5:0]
                auto [olo, ohi] = shift64(u(1), u(0), bin(V::Op::And, s0(), u(63), U), 0);
                ValueId mlo = bin(V::Op::Sub, olo, u(1), U);
                ValueId mhi = bin(V::Op::Sub, ohi, b2u(bin(V::Op::Eq, olo, u(0), V::kBool)), U);
                auto [rlo, rhi] = shift64(mlo, mhi, bin(V::Op::And, s1(), u(63), U), 0);
                put(rlo, 0);
                return put(rhi, 1);
            }
            case G::s_bfe_u64:
            case G::s_bfe_i64: {
                if (maskApprox(x.src[0])) return error(std::string(mnemonic(op)) + " of a lane mask");
                // offset s1[5:0], width s1[22:16] (0: 0; >= 64: to the top)
                ValueId c = s1();
                ValueId off = bin(V::Op::And, c, u(63), U), width = bin(V::Op::And, bin(V::Op::Shr, c, u(16), U), u(0x7f), U);
                auto [lo, hi] = shift64(s0(0), s0(1), off, 1);
                // (a field past bit 63 ends there: it is sign-extended from bit 63)
                ValueId w = b->binary(V::Op::Min, width, bin(V::Op::Sub, u(64), off, U), U);
                ValueId up = bin(V::Op::And, bin(V::Op::Sub, u(64), w, U), u(63), U);  // 64 - width, 0 when width is 64
                auto [l2, h2] = shift64(lo, hi, up, 0);
                auto [rlo, rhi] = shift64(l2, h2, up, op == G::s_bfe_i64 ? 2 : 1);
                ValueId zero = bin(V::Op::Eq, width, u(0), V::kBool);
                rlo = sel(zero, u(0), rlo, U);
                rhi = sel(zero, u(0), rhi, U);
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
                    return setSccMask(r);
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
                    return setSccMask(r);
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
                if (maskApprox(x.src[0])) {
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
            case G::s_brev_b64: {
                if (maskApprox(x.src[0])) return error("s_brev_b64 of a lane mask");
                ValueId lo = s0(0), hi = s0(1);
                put(b->unary(V::Op::BitReverse, hi, U), 0);
                return put(b->unary(V::Op::BitReverse, lo, U), 1);
            }
            case G::s_bcnt0_i32_b64:
                if (maskApprox(x.src[0])) {
                    note("s_bcnt0 of a lane mask counts this lane only");
                    ValueId r = b2u(notB(readMask(x.src[0])));
                    sccNz(r);
                    return put(r);
                }
                {
                    ValueId r = bin(V::Op::Add, b->unary(V::Op::BitCount, b->unary(V::Op::Not, s0(0), U), U),
                                    b->unary(V::Op::BitCount, b->unary(V::Op::Not, s0(1), U), U), U);
                    sccNz(r);
                    return put(r);
                }
            case G::s_ff0_i32_b32: {
                ValueId v = b->unary(V::Op::Not, s0(), U);
                return put(sel(nz(v), b->unary(V::Op::FindLsb, v, U), u(0xffffffffu), U));
            }
            case G::s_ff0_i32_b64:
            case G::s_ff1_i32_b64: {
                if (maskApprox(x.src[0]) && op == G::s_ff1_i32_b64) {
                    // (the first active lane: this one, as v_readfirstlane reads this lane's own value)
                    note("s_ff1 of a lane mask gives lane 0");
                    return put(u(0));
                }
                if (maskApprox(x.src[0])) return error("s_ff0_i32_b64 of a lane mask");
                ValueId lo = s0(0), hi = s0(1);
                if (op == G::s_ff0_i32_b64) lo = b->unary(V::Op::Not, lo, U), hi = b->unary(V::Op::Not, hi, U);
                ValueId inHi = bin(V::Op::Add, b->unary(V::Op::FindLsb, hi, U), u(32), U);
                return put(sel(nz(lo), b->unary(V::Op::FindLsb, lo, U), sel(nz(hi), inHi, u(0xffffffffu), U), U));
            }
            case G::s_flbit_i32_b64:
            case G::s_flbit_i32_i64: {
                if (maskApprox(x.src[0])) return error(std::string(mnemonic(op)) + " of a lane mask");
                // leading zeros from bit 63 (i64: leading bits equal to the sign); none: -1
                ValueId lo = s0(0), hi = s0(1);
                if (op == G::s_flbit_i32_i64) {
                    ValueId neg = bin(V::Op::Ne, bin(V::Op::Shr, hi, u(31), U), u(0), V::kBool);
                    lo = sel(neg, b->unary(V::Op::Not, lo, U), lo, U);
                    hi = sel(neg, b->unary(V::Op::Not, hi, U), hi, U);
                }
                ValueId fromHi = bin(V::Op::Sub, u(31), b->unary(V::Op::FindMsb, hi, U), U);
                ValueId fromLo = bin(V::Op::Sub, u(63), b->unary(V::Op::FindMsb, lo, U), U);
                return put(sel(nz(hi), fromHi, sel(nz(lo), fromLo, u(0xffffffffu), U), U));
            }
            case G::s_bitset0_b64:
            case G::s_bitset1_b64: {
                if (maskApprox(x.dst)) return error(std::string(mnemonic(op)) + " of a lane mask");
                ValueId n = bin(V::Op::And, s0(), u(63), U);
                ValueId inHi = bin(V::Op::Ge, n, u(32), V::kBool);
                ValueId m1 = bin(V::Op::Shl, u(1), bin(V::Op::And, n, u(31), U), U);
                ValueId lo = readS(x.dst, 0), hi = readS(x.dst, 1);
                auto apply = [&](ValueId w) {
                    return op == G::s_bitset1_b64 ? bin(V::Op::Or, w, m1, U) : bin(V::Op::And, w, b->unary(V::Op::Not, m1, U), U);
                };
                put(sel(inHi, lo, apply(lo), U), 0);
                return put(sel(inHi, apply(hi), hi, U), 1);
            }
            case G::s_ff1_i32_b32: {
                ValueId v = s0();
                return put(sel(nz(v), b->unary(V::Op::FindLsb, v, U), u(0xffffffffu), U));
            }
            case G::s_flbit_i32_b32: {
                ValueId v = s0();
                return put(sel(nz(v), bin(V::Op::Sub, u(31), b->unary(V::Op::FindMsb, v, U), U), u(0xffffffffu), U));
            }
            case G::s_flbit_i32: {
                // Leading bits equal to the sign bit; -1 when all 32 are equal.
                ValueId v = s0();
                ValueId flipped = sel(bin(V::Op::Lt, asI(v), b->constI32(0), V::kBool), b->unary(V::Op::Not, v, U), v, U);
                return put(sel(nz(flipped), bin(V::Op::Sub, u(31), b->unary(V::Op::FindMsb, flipped, U), U), u(0xffffffffu), U));
            }
            case G::s_quadmask_b32:
            case G::s_quadmask_b64: {
                // One bit per group of four: set when any of the four is.
                bool wide = op == G::s_quadmask_b64;
                ValueId lo, hi = u(0);
                if (isPairMask(x.src[0])) {
                    if (!maskWords(x.src[0], lo, hi)) return;
                } else {
                    lo = s0(0);
                    if (wide) hi = s0(1);
                }
                ValueId rlo = u(0), rhi = u(0);
                for (unsigned q = 0; q < (wide ? 16u : 8u); ++q) {
                    ValueId w = q < 8 ? lo : hi;
                    ValueId nib = bin(V::Op::And, bin(V::Op::Shr, w, u(4 * (q & 7)), U), u(0xf), U);
                    ValueId bitq = bin(V::Op::Shl, b2u(nz(nib)), u(q), U);
                    rlo = bin(V::Op::Or, rlo, bitq, U);
                }
                if (wide) rhi = u(0);
                sccNz(rlo);
                put(rlo, 0);
                if (wide) put(rhi, 1);
                return;
            }
            case G::s_movrels_b32:
            case G::s_movrels_b64:
            case G::s_movreld_b32:
            case G::s_movreld_b64: {
                // SGPRs are variables: the index in M0 becomes a select chain (reads) or guarded writes.
                bool wide = op == G::s_movrels_b64 || op == G::s_movreld_b64;
                bool read = op == G::s_movrels_b32 || op == G::s_movrels_b64;
                ValueId m0 = b->loadVar(m0V());
                for (unsigned h = 0; h < (wide ? 2u : 1u); ++h) {
                    if (read) {
                        unsigned base = x.src[0].code + h;
                        ValueId value = b->loadVar(sVar(base));
                        for (unsigned k = 1; base + k <= highestSgpr; ++k)
                            value = sel(bin(V::Op::Eq, m0, u(k), V::kBool), b->loadVar(sVar(base + k)), value, U);
                        put(value, h);
                    } else {
                        ValueId value = s0(h);
                        for (unsigned k = 0; x.dst.code + h + k <= highestSgpr; ++k) {
                            b->beginIf(bin(V::Op::Eq, m0, u(k), V::kBool));
                            Operand d2 = x.dst;
                            d2.code = uint16_t(x.dst.code + k);
                            writeS(d2, value, h);
                            b->end();
                        }
                    }
                }
                return;
            }
            case G::s_cbranch_g_fork:
            case G::s_cbranch_i_fork:  // (the branch itself is the block's terminator)
            case G::s_cbranch_join: return;  // reconvergence is the CFG's post-dominator
            case G::s_setvskip: {
                // VSKIP = src0[src1[4:0]]: while set, vector instructions are skipped (liftOne).
                ValueId v = nz(bin(V::Op::And, bin(V::Op::Shr, s0(), shiftAmount(s1()), U), u(1), U));
                b->storeVar(vskipV(), v);
                return;
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
            case G::s_bitcmp0_b64:
            case G::s_bitcmp1_b64: {
                if (maskApprox(x.src[0])) return error(std::string(mnemonic(op)) + " of a lane mask");
                ValueId n = bin(V::Op::And, s1(), u(63), U);
                ValueId w = sel(bin(V::Op::Ge, n, u(32), V::kBool), s0(1), s0(0), U);
                ValueId v = nz(bin(V::Op::And, bin(V::Op::Shr, w, bin(V::Op::And, n, u(31), U), U), u(1), U));
                return setScc(op == G::s_bitcmp1_b64 ? v : notB(v));
            }
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
            case G::s_getreg_b32: {
                // hwreg(id, offset, size) in simm16: id [5:0], offset [10:6], size - 1 [15:11]. Only MODE (1) is kept.
                unsigned id = x.simm16 & 0x3f, off = x.simm16 >> 6 & 31, size = (x.simm16 >> 11) + 1;
                if (id != 1) {
                    note("s_getreg_b32 of a hardware register other than MODE reads 0");
                    return put(u(0));
                }
                return put(bfe(b->loadVar(modeV()), u(off), u(size), false));
            }
            case G::s_setreg_b32:
            case G::s_setreg_imm32_b32: {
                unsigned id = x.simm16 & 0x3f, off = x.simm16 >> 6 & 31, size = (x.simm16 >> 11) + 1;
                if (id != 1) {
                    note("s_setreg_b32 of a hardware register other than MODE is ignored");
                    return;
                }
                bool imm = op == G::s_setreg_imm32_b32;
                ValueId value = imm ? u(x.src[0].value) : readS(x.dst);
                uint32_t fieldMask = size >= 32 ? 0xffffffffu : (1u << size) - 1;
                ValueId mask = u(fieldMask << off);
                ValueId cur = b->loadVar(modeV());
                ValueId nv = bin(V::Op::Or, bin(V::Op::And, cur, b->unary(V::Op::Not, mask, U), U),
                                 bin(V::Op::And, bin(V::Op::Shl, value, u(off), U), mask, U), U);
                b->storeVar(modeV(), nv);
                if (imm) curMode = (curMode & ~(fieldMask << off)) | ((x.src[0].value & fieldMask) << off);
                if (!modeDynamic && (curMode & 0xf)) note("MODE rounding mode other than nearest-even is not honoured");
                return;
            }
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
            // Shaders cannot read a GPU clock in Metal: a counter that advances 1000 ticks per read, so
            // elapsed-time differences are positive and wait loops on it end.
            note("s_memtime returns a synthetic counter (1000 ticks per read), not the GPU clock");
            ValueId t = b->loadVar(var(timeVar, V::kU32));
            writeS(x.dst, t, 0);
            writeS(x.dst, u(0), 1);
            b->storeVar(uint32_t(timeVar), bin(V::Op::Add, t, u(1000), V::kU32));
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

    // Where a MUBUF/MTBUF access lands in its buffer (V# base = the slot's start), and whether it is in range.
    // Sea Islands rules (the address calculation of the ISA manual, buffer addressing):
    //   AINDEX = (idxen ? vindex : 0) + (ADD_TID_ENABLE ? lane id : 0); AOFFSET = inst offset + (offen ? voffset : 0)
    //   plain:    BUFOFFSET = AINDEX * stride + AOFFSET
    //   swizzled: AOFFSET_LSB + elem * AINDEX_LSB + istride * (AINDEX_MSB * stride + AOFFSET_MSB * elem), with
    //             elem = 2 << ELEMENT_SIZE, istride = 8 << INDEX_STRIDE, MSB = quotient and LSB = remainder
    //   address = BUFOFFSET + soffset
    //   range:    stride == 0: address >= num_records is out; stride != 0: AINDEX >= num_records, or (idxen or
    //             ADD_TID) and AOFFSET >= stride, is out. Out of range reads give 0 and writes are dropped.
    //   addr64:   the address is the 64-bit vaddr pair + offset + soffset, not range checked; here relative to the
    //             V# base, so one with bits above 31 set (a pointer outside the buffer) counts as out of range.
    bool noteTid = false;
    struct BufAccess {
        ValueId addr = V::kNoValue;  // byte address (u32)
        ValueId hi = V::kNoValue;    // addr64: the bits above 31 of the address
        ValueId strided = V::kNoValue, idxBad = V::kNoValue, num = V::kNoValue;  // bool, bool, u32
        ValueId lane = V::kNoValue;
    };
    BufAccess bufferAccess(const Instruction& x, const DescriptorInfo* known) {
        auto U = V::kU32;
        const MemoryFields& mf = x.mem;
        BufAccess r;
        ValueId soff = readS(x.src[3]);
        r.lane = laneIndex != V::kNoValue ? bin(V::Op::And, laneIndex, u(63), U) : u(0);
        if (mf.addr64) {
            ValueId lo = b->loadVar(vVar(x.src[1].index)), hi = b->loadVar(vVar(x.src[1].index + 1));
            ValueId nlo = bin(V::Op::Add, lo, bin(V::Op::Add, u(mf.offset), soff, U), U);
            // (two additions can carry twice)
            ValueId t = bin(V::Op::Add, lo, u(mf.offset), U);
            ValueId carry = bin(V::Op::Add, b2u(bin(V::Op::Lt, t, lo, V::kBool)), b2u(bin(V::Op::Lt, nlo, t, V::kBool)), U);
            r.addr = nlo;
            r.hi = bin(V::Op::Add, hi, carry, U);
            return r;
        }
        ValueId w1 = readS(x.src[2], 1), w2 = readS(x.src[2], 2), w3 = readS(x.src[2], 3);
        ValueId stride = bin(V::Op::And, bin(V::Op::Shr, w1, u(16), U), u(0x3fff), U);
        const bool known_ = known && known->layoutKnown;
        const bool mayTid = !known_ || known->addTid, maySwizzle = !known_ || known->swizzled;
        ValueId swz = maySwizzle ? nz(bin(V::Op::Shr, w1, u(31), U)) : V::kNoValue;
        ValueId tidEn = mayTid ? nz(bin(V::Op::And, bin(V::Op::Shr, w3, u(23), U), u(1), U)) : V::kNoValue;
        ValueId lane = r.lane;
        unsigned v = 0;
        ValueId aindex = u(0);
        if (mf.idxen) aindex = b->loadVar(vVar(x.src[1].index + v++));
        if (mayTid) aindex = bin(V::Op::Add, aindex, sel(tidEn, lane, u(0), U), U);
        ValueId aoffset = u(mf.offset);
        if (mf.offen) aoffset = bin(V::Op::Add, aoffset, b->loadVar(vVar(x.src[1].index + v)), U);
        ValueId bufOff = bin(V::Op::Add, bin(V::Op::Mul, aindex, stride, U), aoffset, U);
        if (maySwizzle) {
            ValueId es = bin(V::Op::And, bin(V::Op::Shr, w3, u(19), U), u(3), U);
            ValueId is = bin(V::Op::And, bin(V::Op::Shr, w3, u(21), U), u(3), U);
            ValueId elem = bin(V::Op::Shl, u(2), es, U), istr = bin(V::Op::Shl, u(8), is, U);
            ValueId imsb = bin(V::Op::Shr, aindex, bin(V::Op::Add, is, u(3), U), U);
            ValueId ilsb = bin(V::Op::And, aindex, bin(V::Op::Sub, istr, u(1), U), U);
            ValueId omsb = bin(V::Op::Shr, aoffset, bin(V::Op::Add, es, u(1), U), U);
            ValueId olsb = bin(V::Op::And, aoffset, bin(V::Op::Sub, elem, u(1), U), U);
            ValueId swizzled = bin(V::Op::Add, bin(V::Op::Add, olsb, bin(V::Op::Mul, elem, ilsb, U), U),
                                   bin(V::Op::Mul, istr, bin(V::Op::Add, bin(V::Op::Mul, imsb, stride, U), bin(V::Op::Mul, omsb, elem, U), U), U), U);
            bufOff = sel(bin(V::Op::And, swz, nz(stride), V::kBool), swizzled, bufOff, U);
        }
        r.addr = bin(V::Op::Add, bufOff, soff, U);
        r.strided = nz(stride);
        r.num = w2;
        ValueId offBad = bin(V::Op::Ge, aoffset, stride, V::kBool);
        if (!mf.idxen) offBad = mayTid ? bin(V::Op::And, offBad, tidEn, V::kBool) : b->constBool(false);
        r.idxBad = bin(V::Op::Or, bin(V::Op::Ge, aindex, w2, V::kBool), offBad, V::kBool);
        return r;
    }
    // In range for an access `extra` bytes after the address (a dword of a wide access)?
    ValueId bufferInRange(const BufAccess& a, unsigned extra) {
        if (a.hi != V::kNoValue) return bin(V::Op::Eq, a.hi, u(0), V::kBool);
        ValueId end = bin(V::Op::Add, a.addr, u(extra), V::kU32);
        return sel(a.strided, notB(a.idxBad), bin(V::Op::Lt, end, a.num, V::kBool), V::kBool);
    }
    // body() when `ok`, else 0.
    ValueId guardedValue(ValueId ok, const std::function<ValueId()>& body) {
        uint32_t t = b->variable(V::kU32);
        b->storeVar(t, u(0));
        b->beginIf(ok);
        b->storeVar(t, body());
        b->end();
        return b->loadVar(t);
    }

    // Typed load: `n` components of format dfmt/nfmt at addr -> f32/u32 bit patterns.
    // Component widths and bit offsets of the packed dword formats (6 10_11_11, 7 11_11_10,
    // 8 10_10_10_2, 9 2_10_10_10): AMD names give the fields from the most significant bit, so x is
    // the last one named and sits at bit 0.
    static void packedLayout(unsigned dfmt, unsigned widths[4], unsigned shifts[4]) {
        static const unsigned w6[4] = {11, 11, 10, 0}, w7[4] = {10, 11, 11, 0}, w8[4] = {2, 10, 10, 10}, w9[4] = {10, 10, 10, 2};
        const unsigned* w = dfmt == 6 ? w6 : dfmt == 7 ? w7 : dfmt == 8 ? w8 : w9;
        unsigned at = 0;
        for (unsigned k = 0; k < 4; ++k) {
            widths[k] = w[k];
            shifts[k] = at;
            at += w[k];
        }
    }

    // An unsigned float with a 5-bit exponent (bias 15) and `mant` mantissa bits -> f32 bits.
    ValueId smallFloat(ValueId field, unsigned mant) {
        auto U = V::kU32;
        auto F = V::kF32;
        ValueId m = bin(V::Op::And, field, u((1u << mant) - 1), U);
        ValueId e = bin(V::Op::Shr, field, u(mant), U);
        // normal: exponent rebased to 127, mantissa to 23 bits; e == 31: inf / NaN; e == 0: m * 2^-(14 + mant)
        ValueId normal = bin(V::Op::Or, bin(V::Op::Shl, bin(V::Op::Add, e, u(112), U), u(23), U), bin(V::Op::Shl, m, u(23 - mant), U), U);
        ValueId special = bin(V::Op::Or, u(0x7f800000u), bin(V::Op::Shl, m, u(23 - mant), U), U);
        ValueId denorm = bit(bin(V::Op::Mul, b->unary(V::Op::Convert, m, F), f(std::bit_cast<float>(uint32_t(113 - mant) << 23)), F), U);
        ValueId r = sel(bin(V::Op::Eq, e, u(31), V::kBool), special, normal, U);
        return sel(bin(V::Op::Eq, e, u(0), V::kBool), denorm, r, U);
    }

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
                case 6: {  // snorm_nz (OpenGL convention: (2c + 1) / (2^w - 1), no exact zero)
                    ValueId s = b->unary(V::Op::Convert, asI(bfe(raw, u(0), u(width), true)), F);
                    ValueId v = bin(V::Op::Div, bin(V::Op::Add, bin(V::Op::Mul, s, f(2), F), f(1), F), f(maxU), F);
                    return bit(v, U);
                }
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
            // AMD names list the fields from the most significant bit; x is always in the low bits
            // (2_10_10_10 = A2B10G10R10: x 10 bits at 0, w 2 bits at 30; 10_10_10_2: x 2 bits at 0).
            ValueId w = b->load(slot, addr, U);
            unsigned widths[4], shifts[4];
            packedLayout(dfmt, widths, shifts);
            for (unsigned k = 0; k < 4; ++k)
                c.push_back(convert(b->ternary(V::Op::BitfieldExtract, w, u(shifts[k]), u(widths[k]), U), widths[k]));
        } else {  // 10_11_11 / 11_11_10: unsigned small floats (5-bit exponent, bias 15)
            // Named from the most significant bit (10_11_11 = B10G11R11: x 11 bits at 0, z 10 at 22).
            ValueId w = b->load(slot, addr, U);
            unsigned widths[4], shifts[4];
            packedLayout(dfmt, widths, shifts);
            for (unsigned k = 0; k < 3; ++k) {
                ValueId field = b->ternary(V::Op::BitfieldExtract, w, u(shifts[k]), u(widths[k]), U);
                c.push_back(smallFloat(field, widths[k] - 5));
            }
        }
        // Missing components: (0, 0, 0, 1) as float or integer.
        bool integer = nfmt == 4 || nfmt == 5;
        while (c.size() < n) c.push_back(c.size() == 3 ? u(integer ? 1u : 0x3f800000u) : u(0));
        c.resize(n);
        return c;
    }

    // The inverse of loadFormat for typed stores: components (VGPR bits) encoded per the number
    // format, packed per the data format. Formats with elements under a dword go through
    // storeSubDword (atomics: the buffer is read_write); 11/10-bit floats are not written.
    void storeFormat(uint32_t slot, ValueId addr, unsigned dfmt, unsigned nfmt, unsigned n, const std::vector<ValueId>& v) {
        auto U = V::kU32;
        auto F = V::kF32;
        unsigned bits[16] = {0, 8, 16, 8, 32, 16, 0, 0, 0, 0, 8, 32, 16, 32, 32, 0};
        unsigned count[16] = {0, 1, 1, 2, 1, 2, 3, 3, 4, 4, 4, 2, 4, 3, 4, 0};
        if (dfmt == 0 || dfmt == 6 || dfmt == 7 || dfmt == 15) {
            error("unsupported buffer data format " + std::to_string(dfmt) + " for a store");
            return;
        }
        auto encode = [&](ValueId raw, unsigned width) -> ValueId {
            // -> the field's bits, zero-extended
            uint32_t mask = width == 32 ? 0xffffffffu : (1u << width) - 1;
            // (32-bit: the largest floats below 2^32 / 2^31, which convert without overflow)
            float maxU = width == 32 ? 4294967040.0f : float((1ull << width) - 1);
            float maxS = width == 32 ? 2147483520.0f : float((1ull << (width - 1)) - 1);
            ValueId fv = bit(raw, F);
            ValueId r;
            switch (nfmt) {
                case 0:  // unorm (NaN -> 0)
                    fv = sel(b->unary(V::Op::IsNan, fv, V::kBool), f(0), fv, F);
                    r = b->unary(V::Op::Convert, b->unary(V::Op::Round, bin(V::Op::Mul, b->unary(V::Op::Saturate, fv, F), f(maxU), F), F), U);
                    break;
                case 1: {  // snorm
                    fv = sel(b->unary(V::Op::IsNan, fv, V::kBool), f(0), fv, F);
                    ValueId c = b->ternary(V::Op::Clamp, fv, f(-1), f(1), F);
                    r = bit(b->unary(V::Op::Convert, b->unary(V::Op::Round, bin(V::Op::Mul, c, f(maxS), F), F), V::kI32), U);
                    break;
                }
                case 2:  // uscaled
                    r = b->unary(V::Op::Convert, b->ternary(V::Op::Clamp, sel(b->unary(V::Op::IsNan, fv, V::kBool), f(0), fv, F), f(0), f(maxU), F), U);
                    break;
                case 3:  // sscaled
                    r = bit(b->unary(V::Op::Convert, b->ternary(V::Op::Clamp, sel(b->unary(V::Op::IsNan, fv, V::kBool), f(0), fv, F), f(-maxS - (width == 32 ? 128.0f : 1.0f)), f(maxS), F), V::kI32), U);
                    break;
                case 4:
                case 5: r = raw; break;  // uint / sint: the low bits
                case 7:
                    if (width == 32) return raw;
                    if (width == 16) {
                        r = halfBits(fv);
                        break;
                    }
                    [[fallthrough]];
                default:
                    error("unsupported buffer number format " + std::to_string(nfmt) + " for a store of width " + std::to_string(width));
                    return u(0);
            }
            return width == 32 ? r : bin(V::Op::And, r, u(mask), U);
        };
        auto comp = [&](unsigned k) { return k < v.size() && k < n ? v[k] : u(0); };
        unsigned cnt = std::min(count[dfmt], n);  // (components the instruction does not give are left)
        if (dfmt == 4 || dfmt == 11 || dfmt == 13 || dfmt == 14) {
            for (unsigned k = 0; k < cnt; ++k) b->store(slot, bin(V::Op::Add, addr, u(4 * k), U), encode(comp(k), 32));
            return;
        }
        if (dfmt == 8 || dfmt == 9) {  // 10_10_10_2 / 2_10_10_10, one dword
            unsigned widths[4], shifts[4];
            packedLayout(dfmt, widths, shifts);
            ValueId w = u(0);
            uint32_t mask = 0;
            for (unsigned k = 0; k < cnt; ++k) {
                w = bin(V::Op::Or, w, bin(V::Op::Shl, encode(comp(k), widths[k]), u(shifts[k]), U), U);
                mask |= ((1u << widths[k]) - 1) << shifts[k];
            }
            storeBits(slot, addr, w, mask);
            return;
        }
        // 8- and 16-bit components, packed from the low bits; only the components the instruction
        // gives are written (the others keep their bytes).
        unsigned width = bits[dfmt], total = width * count[dfmt];
        if (total < 32) {
            ValueId packed = u(0);
            for (unsigned k = 0; k < cnt; ++k) packed = bin(V::Op::Or, packed, bin(V::Op::Shl, encode(comp(k), width), u(width * k), U), U);
            storeSubDword(slot, addr, packed, width * cnt);
            return;
        }
        for (unsigned d = 0; d * 32 < total; ++d) {  // one dword, or two for 16_16_16_16
            unsigned per = 32 / width, first = d * per;
            if (first >= cnt) break;
            ValueId w = u(0);
            uint32_t mask = 0;
            for (unsigned k = first; k < std::min(first + per, cnt); ++k) {
                w = bin(V::Op::Or, w, bin(V::Op::Shl, encode(comp(k), width), u(width * (k - first)), U), U);
                mask |= ((1u << width) - 1) << (width * (k - first));
            }
            storeBits(slot, bin(V::Op::Add, addr, u(4 * d), U), w, mask);
        }
    }

    // A dword store of the bits in `mask` only: a plain store when it covers the dword, else the
    // two atomics of storeSubDword (the buffer is read_write then, see liftBuffer).
    void storeBits(uint32_t target, ValueId addr, ValueId value, uint32_t mask) {
        auto U = V::kU32;
        if (mask == 0xffffffffu) {
            b->store(target, addr, value);
            return;
        }
        ValueId keep[] = {u(~mask)};
        b->atomic(V::Op::AtomicAnd, target, addr, keep, U);
        ValueId set[] = {bin(V::Op::And, value, u(mask), U)};
        b->atomic(V::Op::AtomicOr, target, addr, set, U);
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
        auto U = V::kU32;
        if (x.mem.lds && (store || atomic)) return error("lds=1 on a buffer store or atomic");
        // (a V# built by the program: the source holds an address, not a descriptor to describe)
        std::optional<DescriptorInfo> known = in.describe && !rawBuffer.count(idx) ? in.describe(*s) : std::nullopt;
        // Element size per op.
        unsigned n = 0, dfmt = 0, nfmt = 0;
        bool typed = x.encoding == Encoding::MTBUF || name.find("_format_") != std::string_view::npos;
        if (typed) {
            n = unsigned(name.size() - name.rfind('_') - 1);
            if (x.encoding == Encoding::MTBUF) {
                dfmt = x.mem.dfmt;
                nfmt = x.mem.nfmt;
            } else {
                if (known && known->dfmt) {
                    dfmt = known->dfmt;
                    nfmt = known->nfmt;
                } else {
                    // Unknown at build time: 32-bit floats, one per component.
                    unsigned byCount[5] = {0, 4, 11, 13, 14};
                    dfmt = byCount[n];
                    nfmt = 7;
                    note("typed buffer access without a known format: read as 32-bit floats");
                }
            }
        }
        // (byte / short stores, and typed ones of elements under a dword, are atomics on their dword)
        // (and partial ones: fewer components than a packed format has, which keep the others' bits)
        static const unsigned fmtCount[16] = {0, 1, 1, 2, 1, 2, 3, 3, 4, 4, 4, 2, 4, 3, 4, 0};
        bool packedFmt = typed && !(dfmt == 4 || dfmt == 11 || dfmt == 13 || dfmt == 14);
        bool subDword = store && (name.ends_with("_byte") || name.ends_with("_short") ||
                                  (typed && (dfmt == 1 || dfmt == 2 || dfmt == 3)) || (packedFmt && n < fmtCount[dfmt & 15]));
        V::Access access = store && !subDword ? V::Access::Write : atomic || subDword ? V::Access::ReadWrite : V::Access::Read;
        // (dwords 0: the source holds the buffer's address, the V# is built by the program.)
        uint32_t slot = resource(V::ResourceKind::Buffer, *s, rawBuffer.count(idx) ? 0 : 4, access, "buf");
        BufAccess ba = bufferAccess(x, known ? &*known : nullptr);
        ValueId addr = ba.addr;
        unsigned data = x.src[0].index;
        auto guarded = [&](auto&& body) {
            if (writesExec) b->beginIf(b->loadVar(execV()));
            body();
            if (writesExec) b->end();
        };
        if (atomic) return liftBufferAtomic(x, name, slot, ba);
        if (store) {
            // (typed: the first and last byte of the element must be in range; wide: each dword is checked)
            auto body = [&] {
                if (typed) {
                    static const unsigned bitsOf[16] = {0, 8, 16, 8, 32, 16, 0, 0, 0, 0, 8, 32, 16, 32, 32, 0};
                    unsigned bytes = std::max(1u, (bitsOf[dfmt & 15] * fmtCount[dfmt & 15] + 7) / 8);
                    std::vector<ValueId> v;
                    for (unsigned k = 0; k < n; ++k) v.push_back(b->loadVar(vVar(data + k)));
                    b->beginIf(bin(V::Op::And, bufferInRange(ba, 0), bufferInRange(ba, bytes - 1), V::kBool));
                    storeFormat(slot, addr, dfmt, nfmt, n, v);
                    b->end();
                    return;
                }
                if (name.ends_with("_byte") || name.ends_with("_short")) {
                    b->beginIf(bufferInRange(ba, 0));
                    storeSubDword(slot, name.ends_with("_short") ? bin(V::Op::And, addr, u(~1u), U) : addr, b->loadVar(vVar(data)),
                                  name.ends_with("_byte") ? 8 : 16);
                    b->end();
                    return;
                }
                unsigned dw = x.src[0].count;
                ValueId base = bin(V::Op::And, addr, u(~3u), U);
                for (unsigned k = 0; k < dw; ++k) {
                    b->beginIf(bufferInRange(ba, 4 * k));
                    b->store(slot, bin(V::Op::Add, base, u(4 * k), U), b->loadVar(vVar(data + k)));
                    b->end();
                }
            };
            guarded(body);
            return;
        }
        // Loads (out of range: zero).
        std::vector<ValueId> vals;
        if (typed) {
            static const unsigned bitsOf[16] = {0, 8, 16, 8, 32, 16, 0, 0, 0, 0, 8, 32, 16, 32, 32, 0};
            unsigned bytes = std::max(1u, (bitsOf[dfmt & 15] * fmtCount[dfmt & 15] + 7) / 8);
            ValueId ok = bin(V::Op::And, bufferInRange(ba, 0), bufferInRange(ba, bytes - 1), V::kBool);
            std::vector<uint32_t> tmp;
            for (unsigned k = 0; k < n; ++k) {
                tmp.push_back(b->variable(U));
                b->storeVar(tmp.back(), u(0));
            }
            b->beginIf(ok);
            std::vector<ValueId> got = loadFormat(slot, addr, dfmt, nfmt, n);
            for (unsigned k = 0; k < n; ++k) b->storeVar(tmp[k], got[k]);
            b->end();
            for (unsigned k = 0; k < n; ++k) vals.push_back(b->loadVar(tmp[k]));
        } else if (name.ends_with("_ubyte") || name.ends_with("_sbyte") || name.ends_with("_ushort") || name.ends_with("_sshort")) {
            bool half = name.ends_with("short");
            ValueId a = half ? bin(V::Op::And, addr, u(~1u), U) : addr;
            ValueId got = guardedValue(bufferInRange(ba, 0), [&] {
                ValueId w = b->load(slot, bin(V::Op::And, a, u(~3u), U), U);
                ValueId sh = bin(V::Op::Shl, bin(V::Op::And, a, u(3), U), u(3), U);
                return bin(V::Op::Shr, w, sh, U);
            });
            unsigned width = half ? 16 : 8;
            vals.push_back(bfe(got, u(0), u(width), name.ends_with("_sbyte") || name.ends_with("_sshort")));
        } else {
            unsigned dw = name.ends_with("x4") ? 4 : name.ends_with("x3") ? 3 : name.ends_with("x2") ? 2 : 1;
            ValueId base = bin(V::Op::And, addr, u(~3u), U);
            for (unsigned k = 0; k < dw; ++k)
                vals.push_back(guardedValue(bufferInRange(ba, 4 * k), [&] { return b->load(slot, bin(V::Op::Add, base, u(4 * k), U), U); }));
        }
        if (x.mem.lds) {
            // The data goes to LDS instead of the VGPRs: at M0[15:0] + the instruction's offset + lane * 4.
            if (m.sharedBytes == 0) return error("buffer load into LDS in a program without compute LDS");
            if (laneIndex == V::kNoValue) note("buffer load into LDS outside compute: the lane id is taken as 0");
            if (vals.size() != 1) return error("buffer load into LDS of more than one dword");
            ValueId la = bin(V::Op::Add, bin(V::Op::And, b->loadVar(m0V()), u(0xffff), U),
                             bin(V::Op::Add, u(x.mem.offset), bin(V::Op::Shl, ba.lane != V::kNoValue ? ba.lane : u(0), u(2), U), U), U);
            if (writesExec) b->beginIf(b->loadVar(execV()));
            b->store(V::kTargetShared, la, vals[0]);
            if (writesExec) b->end();
            return;
        }
        for (unsigned k = 0; k < vals.size(); ++k) writeV(data + k, vals[k]);
        if (x.mem.tfe) writeV(data + unsigned(vals.size()), u(0));
    }

    // MUBUF atomics. 32-bit: native IR atomics, a compare-exchange loop for inc / dec / fcmpswap / fmin / fmax. 64-bit
    // (_x2): add / sub (carry through the low dword), and / or / xor / swap are atomic per dword; the rest is a read-
    // modify-write, lane by lane in lane order in a one-wave compute workgroup (as DS), else not atomic (noted).
    // Out of range: dropped, 0 returned.
    void liftBufferAtomic(const Instruction& x, std::string_view name, uint32_t slot, const BufAccess& ba) {
        auto U = V::kU32;
        std::string op(name.substr(std::strlen("buffer_atomic_")));
        const bool wide = op.ends_with("_x2");
        if (wide) op.resize(op.size() - 3);
        static const std::map<std::string, DsKind> kinds = {
            {"add", DsKind::Add}, {"sub", DsKind::Sub}, {"smin", DsKind::MinI}, {"umin", DsKind::MinU}, {"smax", DsKind::MaxI},
            {"umax", DsKind::MaxU}, {"and", DsKind::And}, {"or", DsKind::Or}, {"xor", DsKind::Xor}, {"swap", DsKind::Xchg},
            {"cmpswap", DsKind::Cmpst}, {"inc", DsKind::Inc}, {"dec", DsKind::Dec}, {"fcmpswap", DsKind::CmpstF},
            {"fmin", DsKind::MinF}, {"fmax", DsKind::MaxF}};
        auto it = kinds.find(op);
        if (it == kinds.end()) return error("unsupported buffer atomic " + std::string(name));
        const DsKind kind = it->second;
        const unsigned data = x.src[0].index;
        const bool glc = x.mem.glc;
        const bool cmp = kind == DsKind::Cmpst || kind == DsKind::CmpstF;
        auto reg = [&](unsigned k) { return b->loadVar(vVar(data + k)); };
        auto guarded = [&](auto&& body) {
            if (writesExec) b->beginIf(b->loadVar(execV()));
            body();
            if (writesExec) b->end();
        };
        // The old value comes back in vdata (glc). Results are written after every operand was read.
        uint32_t oldLo = b->variable(U), oldHi = b->variable(U);
        b->storeVar(oldLo, u(0));
        b->storeVar(oldHi, u(0));
        auto finish = [&] {
            if (!glc) return;
            b->storeVar(vVar(data), b->loadVar(oldLo));
            if (wide) b->storeVar(vVar(data + 1), b->loadVar(oldHi));
        };
        auto casLoop = [&](ValueId a, const std::function<ValueId(ValueId)>& next) {
            uint32_t got = b->variable(U);
            b->beginLoop();
            ValueId old = b->load(slot, a, U);
            ValueId nv = next(old);
            ValueId cmpNew[] = {old, nv};
            ValueId prev = b->atomic(V::Op::AtomicCompareExchange, slot, a, cmpNew, U);
            b->beginIf(bin(V::Op::Eq, prev, old, V::kBool));
            b->storeVar(got, old);
            b->breakLoop();
            b->end();
            b->end();
            return b->loadVar(got);
        };
        ValueId a0 = bin(V::Op::And, ba.addr, u(~3u), U);
        if (!wide) {
            V::Op native = V::Op::AtomicAdd;
            bool haveNative = true;
            switch (kind) {
                case DsKind::Add: native = V::Op::AtomicAdd; break;
                case DsKind::Sub: native = V::Op::AtomicSub; break;
                case DsKind::MinI:
                case DsKind::MinU: native = V::Op::AtomicMin; break;
                case DsKind::MaxI:
                case DsKind::MaxU: native = V::Op::AtomicMax; break;
                case DsKind::And: native = V::Op::AtomicAnd; break;
                case DsKind::Or: native = V::Op::AtomicOr; break;
                case DsKind::Xor: native = V::Op::AtomicXor; break;
                case DsKind::Xchg: native = V::Op::AtomicExchange; break;
                case DsKind::Cmpst: native = V::Op::AtomicCompareExchange; break;
                default: haveNative = false; break;
            }
            bool sign = kind == DsKind::MinI || kind == DsKind::MaxI;
            guarded([&] {
                // vdata = {value} or {swap value, compare value}
                ValueId d0 = reg(0), d1 = cmp ? reg(1) : u(0);
                b->beginIf(bufferInRange(ba, 0));
                ValueId r;
                if (haveNative) {
                    std::vector<ValueId> vals = kind == DsKind::Cmpst ? std::vector<ValueId>{d1, d0} : std::vector<ValueId>{d0};
                    if (sign)
                        for (ValueId& v : vals) v = asI(v);
                    r = b->atomic(native, slot, a0, vals, sign ? V::kI32 : U);
                    if (sign) r = bit(r, U);
                } else {
                    // (compare-exchange kinds: DS cmpst is MEM == DATA0 ? DATA1 : MEM, the data is {swap, compare})
                    ValueId c0 = cmp ? d1 : d0, c1 = cmp ? d0 : u(0);
                    r = casLoop(a0, [&](ValueId old) { return dsCombine32(kind, old, c0, c1); });
                }
                b->storeVar(oldLo, r);
                b->end();
                finish();
            });
            return;
        }
        auto word = [&](unsigned k) { return bin(V::Op::Add, a0, u(4 * k), U); };
        auto inRange = [&] { return bin(V::Op::And, bufferInRange(ba, 0), bufferInRange(ba, 7), V::kBool); };
        if (!waveMode) needWave();  // (in a one-wave compute workgroup the lanes take turns, exactly)
        bool perDword = !waveMode && (kind == DsKind::Add || kind == DsKind::Sub || kind == DsKind::And || kind == DsKind::Or ||
                        kind == DsKind::Xor || kind == DsKind::Xchg);
        if (perDword) {
            note("64-bit buffer atomic " + std::string(name) + ": atomic per dword, not as a pair");
            guarded([&] {
                ValueId d0 = reg(0), d1 = reg(1);
                b->beginIf(inRange());
                ValueId lo, hi;
                switch (kind) {
                    case DsKind::Add:
                    case DsKind::Sub: {
                        bool add = kind == DsKind::Add;
                        lo = b->atomic(add ? V::Op::AtomicAdd : V::Op::AtomicSub, slot, word(0), std::vector<ValueId>{d0}, U);
                        // carry (add) or borrow (sub) out of the low dword
                        ValueId c = add ? bin(V::Op::Lt, bin(V::Op::Add, lo, d0, U), lo, V::kBool) : bin(V::Op::Lt, lo, d0, V::kBool);
                        hi = b->atomic(add ? V::Op::AtomicAdd : V::Op::AtomicSub, slot, word(1),
                                       std::vector<ValueId>{bin(V::Op::Add, d1, b2u(c), U)}, U);
                        break;
                    }
                    default: {
                        V::Op o = kind == DsKind::And ? V::Op::AtomicAnd : kind == DsKind::Or ? V::Op::AtomicOr
                                  : kind == DsKind::Xor ? V::Op::AtomicXor : V::Op::AtomicExchange;
                        lo = b->atomic(o, slot, word(0), std::vector<ValueId>{d0}, U);
                        hi = b->atomic(o, slot, word(1), std::vector<ValueId>{d1}, U);
                    }
                }
                b->storeVar(oldLo, lo);
                b->storeVar(oldHi, hi);
                b->end();
                finish();
            });
            return;
        }
        auto rmw = [&] {
            b->beginIf(inRange());
            P64 old{b->load(slot, word(0), U), b->load(slot, word(1), U)};
            P64 d0{reg(cmp ? 2 : 0), reg(cmp ? 3 : 1)}, d1{cmp ? reg(0) : u(0), cmp ? reg(1) : u(0)};
            P64 nv = dsCombine64(kind, old, d0, d1);
            b->store(slot, word(0), nv.lo);
            b->store(slot, word(1), nv.hi);
            b->storeVar(oldLo, old.lo);
            b->storeVar(oldHi, old.hi);
            b->end();
        };
        if (!waveMode) {
            note("64-bit buffer atomic " + std::string(name) + " is not atomic (needs a one-wave compute workgroup for lane order)");
            guarded([&] {
                rmw();
                finish();
            });
            return;
        }
        uint32_t round = b->variable(U);
        b->storeVar(round, u(0));
        b->beginLoop();
        ValueId r = b->loadVar(round);
        b->beginIf(bin(V::Op::Ge, r, u(64), V::kBool));
        b->breakLoop();
        b->end();
        ValueId mine = bin(V::Op::Eq, bin(V::Op::And, laneIndex, u(63), U), r, V::kBool);
        b->beginIf(writesExec ? bin(V::Op::And, mine, b->loadVar(execV()), V::kBool) : mine);
        rmw();
        finish();
        b->end();
        b->barrier();
        b->storeVar(round, bin(V::Op::Add, r, u(1), U));
        b->end();
    }

    // ---- Images --------------------------------------------------------------------------------

    // image_load_pck / pck_sgn: a channel value (typed as the texture's texels) as the raw integer the format stores, zero-
    // extended (pck) or sign-extended (pck_sgn) from `bits`.
    ValueId pckOut(ValueId comp, V::Scalar elem, unsigned bits, unsigned num, bool sgn) {
        auto U = V::kU32;
        auto F = V::kF32;
        auto fieldMask = [&](ValueId v) { return bits >= 32 ? v : bin(V::Op::And, v, u((1u << bits) - 1), U); };
        auto asField = [&](ValueId v) { return sgn ? bfe(v, u(0), u(bits), true) : fieldMask(v); };
        switch (num) {
            case 0: {  // unorm
                ValueId scaled = b->unary(V::Op::Round, bin(V::Op::Mul, b->ternary(V::Op::Clamp, comp, f(0), f(1), F), f(float((1ull << bits) - 1)), F), F);
                return asField(b->unary(V::Op::Convert, scaled, U));
            }
            case 1:
            case 6: {  // snorm
                ValueId scaled = b->unary(V::Op::Round, bin(V::Op::Mul, b->ternary(V::Op::Clamp, comp, f(-1), f(1), F), f(float((1ull << (bits - 1)) - 1)), F), F);
                return asField(bit(b->unary(V::Op::Convert, scaled, V::kI32), U));
            }
            case 2: return asField(b->unary(V::Op::Convert, b->unary(V::Op::Round, comp, F), U));
            case 3: return asField(bit(b->unary(V::Op::Convert, b->unary(V::Op::Round, comp, F), V::kI32), U));
            case 7:
                if (bits == 16) return sgn ? bfe(halfBits(comp), u(0), u(16), true) : halfBits(comp);
                return bit(comp, U);
            default: return asField(elem == V::Scalar::F32 ? bit(comp, U) : elem == V::Scalar::I32 ? bit(comp, U) : comp);
        }
    }
    // image_store_pck: the raw channel (VGPR bits) back to the texel type's bits.
    ValueId pckIn(ValueId raw, V::Scalar elem, unsigned bits, unsigned num) {
        auto U = V::kU32;
        auto F = V::kF32;
        auto field = [&] { return bits >= 32 ? raw : bin(V::Op::And, raw, u((1u << bits) - 1), U); };
        auto sfield = [&] { return bits >= 32 ? raw : bfe(raw, u(0), u(bits), true); };
        switch (num) {
            case 0: return bit(bin(V::Op::Div, b->unary(V::Op::Convert, field(), F), f(float((1ull << bits) - 1)), F), U);
            case 1:
            case 6:
                return bit(bin(V::Op::Max, bin(V::Op::Div, b->unary(V::Op::Convert, asI(sfield()), F), f(float((1ull << (bits - 1)) - 1)), F), f(-1), F), U);
            case 2: return bit(b->unary(V::Op::Convert, field(), F), U);
            case 3: return bit(b->unary(V::Op::Convert, asI(sfield()), F), U);
            case 5: return sfield();
            case 7:
                if (bits == 16) return bit(halfToF(field(), 0), U);
                return raw;
            default: (void)elem; return field();
        }
    }

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
        const bool getLod = name == "image_get_lod";
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
            std::string_view mods = getLod ? std::string_view() : name.substr(gather ? 13 : 12);  // "_c_b_cl_o" etc.
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
            const bool fragment = m.stage == V::Stage::Fragment;
            // Address VGPRs: [offset] [bias] [compare] [derivatives] coordinates [lod | clamp].
            ValueId offsetPacked = has("o") ? addrU() : V::kNoValue;
            ValueId bias = has("b") ? addrF() : V::kNoValue;
            ValueId compare = has("c") ? addrF() : V::kNoValue;
            std::vector<ValueId> dxv, dyv;  // derivatives per coordinate (cubes: the two face coordinates)
            if (has("d") || has("cd")) {
                const unsigned dcount = cubeDim ? 2 : cw;
                for (unsigned k = 0; k < dcount; ++k) dxv.push_back(addrF());
                for (unsigned k = 0; k < dcount; ++k) dyv.push_back(addrF());
            }
            // Texel offsets: 6-bit signed per component at bits [5:0], [13:8], [21:16].
            ValueId offs[3] = {V::kNoValue, V::kNoValue, V::kNoValue};
            if (offsetPacked != V::kNoValue)
                for (unsigned k = 0; k < 3; ++k) offs[k] = asI(bfe(offsetPacked, u(8 * k), u(6), true));
            auto sizeOf = [&](ValueId level) {  // the texture's size (u32 vector; scalar for 1D) at a mip level
                unsigned sw = r.dim == V::TexDim::D1 ? 1 : (r.dim == V::TexDim::D2 || r.dim == V::TexDim::Cube) ? 2 : 3;
                V::Inst q{V::Op::ImageSize, sw == 1 ? U : V::vec(U, uint8_t(sw)), V::kNoValue, false, {}, tex, 0, {}, 0};
                if (level != V::kNoValue) q.args.push_back(level), q.image = V::kImageLod;
                return b->emit(q);
            };
            auto sizeF = [&](ValueId level, unsigned k) {  // component k of the size, as f32
                ValueId sz = sizeOf(level);
                unsigned sw = r.dim == V::TexDim::D1 ? 1 : (r.dim == V::TexDim::D2 || r.dim == V::TexDim::Cube) ? 2 : 3;
                return b->unary(V::Op::Convert, sw == 1 ? sz : b->extract(sz, k, U), F);
            };
            std::vector<ValueId> coord;
            ValueId faceIndex = V::kNoValue;  // cubes: the face (0-5), selecting the direction components
            std::function<ValueId(std::array<ValueId, 6>)> pick;
            ValueId cubeS = V::kNoValue, cubeT = V::kNoValue, cubeFace = V::kNoValue;
            if (cubeDim) {
                cubeS = addrF(), cubeT = addrF(), cubeFace = addrF();
                // Texel offsets move the point over the face (the face coordinates span one texture width).
                if (offs[0] != V::kNoValue) {
                    cubeS = bin(V::Op::Add, cubeS, bin(V::Op::Div, b->unary(V::Op::Convert, offs[0], F), sizeF(V::kNoValue, 0), F), F);
                    cubeT = bin(V::Op::Add, cubeT, bin(V::Op::Div, b->unary(V::Op::Convert, offs[1], F), sizeF(V::kNoValue, 1), F), F);
                }
                // (s, t, face[, slice]) in [1, 2] -> direction.
                ValueId uu = bin(V::Op::Mul, bin(V::Op::Sub, cubeS, f(1.5f), F), f(2), F);
                ValueId vv = bin(V::Op::Mul, bin(V::Op::Sub, cubeT, f(1.5f), F), f(2), F);
                ValueId nu = b->unary(V::Op::Neg, uu, F), nv = b->unary(V::Op::Neg, vv, F);
                // The face VGPR holds slice * 8 + face for cube arrays.
                ValueId fs = b->unary(V::Op::Convert, b->unary(V::Op::Round, cubeFace, F), U);
                faceIndex = bin(V::Op::And, fs, u(7), U);
                pick = [&](std::array<ValueId, 6> per) {
                    ValueId r2 = per[5];
                    for (int k = 4; k >= 0; --k) r2 = sel(bin(V::Op::Eq, faceIndex, u(unsigned(k)), V::kBool), per[k], r2, F);
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
                    for (unsigned k = 0; k < cw; ++k) coord[k] = bin(V::Op::Div, coord[k], sizeF(V::kNoValue, k), F);
                }
            }
            ValueId lod = V::kNoValue, clampLod = V::kNoValue;
            if (has("l")) lod = addrF();
            if (has("lz")) lod = f(0);
            if (has("cl")) clampLod = addrF();
            // A 1D texture takes no offset operand in Metal: the offset moves the coordinate by whole texels.
            if (offs[0] != V::kNoValue && !cubeDim && cw == 1 && !gather)
                coord[0] = bin(V::Op::Add, coord[0], bin(V::Op::Div, b->unary(V::Op::Convert, offs[0], F), sizeF(V::kNoValue, 0), F), F);
            // Gradients: for a cube, face-space derivatives (ds, dt) become derivatives of the direction, which on each
            // face is linear in (u, v) = 2 * (s, t) - 3 with the major axis constant.
            std::vector<ValueId> grads;
            if (!dxv.empty()) {
                if (cubeDim) {
                    ValueId dux = bin(V::Op::Mul, dxv[0], f(2), F), dvx = bin(V::Op::Mul, dxv[1], f(2), F);
                    ValueId duy = bin(V::Op::Mul, dyv[0], f(2), F), dvy = bin(V::Op::Mul, dyv[1], f(2), F);
                    auto dir = [&](ValueId du, ValueId dv) {
                        ValueId z = f(0), ndu = b->unary(V::Op::Neg, du, F), ndv = b->unary(V::Op::Neg, dv, F);
                        std::vector<ValueId> d;
                        d.push_back(pick({z, z, du, du, du, ndu}));
                        d.push_back(pick({ndv, ndv, z, z, ndv, ndv}));
                        d.push_back(pick({ndu, du, dv, ndv, z, z}));
                        return b->construct(d, V::vec(F, 3));
                    };
                    grads = {dir(dux, dvx), dir(duy, dvy)};
                } else {
                    grads = {cw == 1 ? dxv[0] : b->construct(dxv, V::vec(F, uint8_t(cw))),
                             cw == 1 ? dyv[0] : b->construct(dyv, V::vec(F, uint8_t(cw)))};
                }
            }
            // The coordinate without its array layer, and the derivative of it across the quad (implicit lod outside
            // a pixel shader: the quad is four consecutive lanes, as the hardware takes its derivatives).
            const unsigned gw = cubeDim ? 3 : cw;
            auto quadGrads = [&](std::vector<ValueId>& gx, std::vector<ValueId>& gy) {
                gx.clear(), gy.clear();
                for (unsigned k = 0; k < gw; ++k) {
                    auto at = [&](unsigned lane) {
                        return b->emit(V::Inst{V::Op::QuadShuffle, F, V::kNoValue, false, {coord[k], u(lane)}, 0, 0, {}, 0});
                    };
                    ValueId c0 = at(0);
                    gx.push_back(bin(V::Op::Sub, at(1), c0, F));
                    gy.push_back(bin(V::Op::Sub, at(2), c0, F));
                }
            };
            // log2 of the texel footprint (the larger of the two gradient lengths in texels).
            auto lodFromGrads = [&](const std::vector<ValueId>& gx, const std::vector<ValueId>& gy) {
                ValueId sx = f(0), sy = f(0);
                for (unsigned k = 0; k < gw; ++k) {
                    ValueId scale = cubeDim ? bin(V::Op::Mul, sizeF(V::kNoValue, 0), f(0.5f), F) : sizeF(V::kNoValue, k);
                    ValueId tx = bin(V::Op::Mul, gx[k], scale, F), ty = bin(V::Op::Mul, gy[k], scale, F);
                    sx = bin(V::Op::Add, sx, bin(V::Op::Mul, tx, tx, F), F);
                    sy = bin(V::Op::Add, sy, bin(V::Op::Mul, ty, ty, F), F);
                }
                ValueId rho2 = bin(V::Op::Max, sx, sy, F);
                return bin(V::Op::Mul, b->unary(V::Op::Log2, bin(V::Op::Max, rho2, f(1e-30f), F), F), f(0.5f), F);
            };
            const bool quadOk = quadLane != V::kNoValue && !fragment;
            if (getLod) {
                // (lod as sampled, clamped; lod computed, unclamped) in data[0], data[1]
                std::vector<ValueId> lods(2, f(0));
                V::Type ct = coord.size() == 1 ? F : V::vec(F, uint8_t(coord.size()));
                ValueId cv = coord.size() == 1 ? coord[0] : b->construct(coord, ct);
                if (fragment) {
                    ValueId q = b->emit(V::Inst{V::Op::ImageQueryLod, V::vec(F, 2), V::kNoValue, false, {cv}, tex, smp, {}, 0});
                    lods = {b->extract(q, 0, F), b->extract(q, 1, F)};
                } else if (quadOk && r.dim != V::TexDim::D1 && !(gw == 1)) {
                    std::vector<ValueId> gx, gy;
                    quadGrads(gx, gy);
                    ValueId raw = lodFromGrads(gx, gy);
                    ValueId levels = b->emit(V::Inst{V::Op::ImageLevels, U, V::kNoValue, false, {}, tex, 0, {}, 0});
                    ValueId top = bin(V::Op::Max, b->unary(V::Op::Convert, bin(V::Op::Sub, levels, u(1), U), F), f(0), F);
                    lods = {bin(V::Op::Min, bin(V::Op::Max, raw, f(0), F), top, F), raw};
                    note("image_get_lod outside a pixel shader takes the lod from the quad's lane differences");
                } else {
                    note("image_get_lod outside a pixel shader returns lod 0");
                }
                unsigned k = 0;
                for (unsigned c = 0; c < 4; ++c)
                    if (dmask >> c & 1) writeV(dataReg + k++, c < 2 ? asU(lods[c], F) : u(0));
                return;
            }
            if (compare != V::kNoValue && !r.depth) {
                m.resources[tex].depth = true;
                note("a texture sampled with compare is declared as a depth texture");
            }
            // Offsets as an operand: 2D, 3D and arrays (1D and cubes were handled on the coordinate).
            ValueId offsetVec = V::kNoValue;
            if (offs[0] != V::kNoValue && !cubeDim && cw > 1) {
                std::vector<ValueId> o;
                for (unsigned k = 0; k < cw; ++k) o.push_back(offs[k]);
                offsetVec = b->construct(o, V::vec(V::kI32, uint8_t(cw)));
            }
            // Implicit lod outside a pixel shader: from the quad's differences when the lanes form quads (a gradient
            // sample; the bias scales the gradients by 2^bias), else lod 0.
            if (!gather && lod == V::kNoValue && grads.empty() && !fragment) {
                if (quadOk && r.dim != V::TexDim::D1) {
                    std::vector<ValueId> gx, gy;
                    quadGrads(gx, gy);
                    if (bias != V::kNoValue) {
                        ValueId scale = b->unary(V::Op::Exp2, bias, F);
                        for (unsigned k = 0; k < gw; ++k) gx[k] = bin(V::Op::Mul, gx[k], scale, F), gy[k] = bin(V::Op::Mul, gy[k], scale, F);
                        bias = V::kNoValue;
                    }
                    grads = {gw == 1 ? gx[0] : b->construct(gx, V::vec(F, uint8_t(gw))), gw == 1 ? gy[0] : b->construct(gy, V::vec(F, uint8_t(gw)))};
                    note("implicit-lod sampling outside a pixel shader takes its derivatives from the quad's lane differences");
                } else {
                    if (bias != V::kNoValue) note("bias outside a pixel shader is dropped");
                    bias = V::kNoValue;
                    note("implicit-lod sampling outside a pixel shader uses lod 0");
                    lod = f(0);
                }
            }
            if (r.dim == V::TexDim::D1 && !gather && (lod != V::kNoValue || bias != V::kNoValue || !grads.empty() || clampLod != V::kNoValue)) {
                // (Metal's 1D textures take no lod operands)
                note("lod, bias, gradients and clamp of a 1D texture sample are ignored");
                lod = bias = clampLod = V::kNoValue;
                grads.clear();
            }
            V::Type ct = coord.size() == 1 ? F : V::vec(F, uint8_t(coord.size()));
            if (gather) {
                const unsigned compIndex = dmask ? unsigned(std::countr_zero(dmask)) : 0;
                const bool explicitLevel = lod != V::kNoValue, hasBias = bias != V::kNoValue || !grads.empty();
                const V::Resource rr0 = m.resources[tex];
                const bool emulate = (explicitLevel || hasBias) && !cubeDim && (rr0.depth || rr0.elem == V::Scalar::F32) && r.dim != V::TexDim::D3;
                if (emulate) {
                    // Gather at a mip level: Metal's gather reads level 0, so the four texels are point-sampled at their
                    // centres with an explicit lod (the sampler still applies its addressing and border rules).
                    ValueId level;
                    if (explicitLevel) {
                        level = lod;
                    } else if (fragment) {
                        V::Type qt = ct;
                        ValueId cvq = coord.size() == 1 ? coord[0] : b->construct(coord, qt);
                        ValueId q = b->emit(V::Inst{V::Op::ImageQueryLod, V::vec(F, 2), V::kNoValue, false, {cvq}, tex, smp, {}, 0});
                        level = b->extract(q, 1, F);
                        if (bias != V::kNoValue) level = bin(V::Op::Add, level, bias, F);
                    } else {
                        level = f(0);
                        note("gather with bias or gradients outside a pixel shader uses lod 0");
                    }
                    ValueId levels = b->emit(V::Inst{V::Op::ImageLevels, U, V::kNoValue, false, {}, tex, 0, {}, 0});
                    ValueId top = b->unary(V::Op::Convert, bin(V::Op::Sub, levels, u(1), U), F);
                    level = bin(V::Op::Min, bin(V::Op::Max, b->unary(V::Op::Floor, bin(V::Op::Add, level, f(0.5f), F), F), f(0), F), top, F);
                    ValueId levelU = b->unary(V::Op::Convert, level, U);
                    ValueId w = bin(V::Op::Max, sizeF(levelU, 0), f(1), F), h = bin(V::Op::Max, sizeF(levelU, 1), f(1), F);
                    ValueId fx = b->unary(V::Op::Floor, bin(V::Op::Sub, bin(V::Op::Mul, coord[0], w, F), f(0.5f), F), F);
                    ValueId fy = b->unary(V::Op::Floor, bin(V::Op::Sub, bin(V::Op::Mul, coord[1], h, F), f(0.5f), F), F);
                    if (offs[0] != V::kNoValue) {
                        fx = bin(V::Op::Add, fx, b->unary(V::Op::Convert, offs[0], F), F);
                        fy = bin(V::Op::Add, fy, b->unary(V::Op::Convert, offs[1], F), F);
                    }
                    const V::Type rt = rr0.depth ? F : V::Type(rr0.elem, 4);
                    for (unsigned k = 0; k < 4; ++k) {
                        static const float ox[4] = {0, 1, 1, 0}, oy[4] = {1, 1, 0, 0};  // (i0,j1) (i1,j1) (i1,j0) (i0,j0)
                        std::vector<ValueId> c2;
                        c2.push_back(bin(V::Op::Div, bin(V::Op::Add, fx, f(ox[k] + 0.5f), F), w, F));
                        c2.push_back(bin(V::Op::Div, bin(V::Op::Add, fy, f(oy[k] + 0.5f), F), h, F));
                        if (arrayed) c2.push_back(coord[2]);
                        ValueId cv2 = b->construct(c2, V::vec(F, uint8_t(c2.size())));
                        uint8_t o2 = V::kImageLod;
                        std::vector<ValueId> ex2{level};
                        if (compare != V::kNoValue) o2 |= V::kImageCompare, ex2.push_back(compare);
                        ValueId s2 = b->imageSample(tex, smp, cv2, o2, ex2, rt);
                        ValueId v2 = rr0.depth ? s2 : b->extract(s2, compIndex, F);
                        writeV(dataReg + k, asU(v2, F));
                    }
                    if (x.mem.tfe || x.mem.lwe) writeV(dataReg + 4, u(0));
                    if (clampLod != V::kNoValue) note("sampling lod clamp (_cl) is ignored by gather");
                    note("gather with lod or bias is four point samples at the texel centres");
                    return;
                }
                if (explicitLevel || hasBias) note("gather ignores lod/bias");
                if (clampLod != V::kNoValue) note("sampling lod clamp (_cl) is ignored by gather");
                lod = bias = clampLod = V::kNoValue;
                grads.clear();
            }
            // Optional operands in ImageOperand bit order: lod, bias, grad (2), offset, compare, minlod.
            uint8_t ops = 0;
            std::vector<ValueId> extra;
            if (lod != V::kNoValue) ops |= V::kImageLod, extra.push_back(lod);
            if (bias != V::kNoValue) ops |= V::kImageBias, extra.push_back(bias);
            if (!grads.empty()) ops |= V::kImageGrad, extra.insert(extra.end(), grads.begin(), grads.end());
            if (offsetVec != V::kNoValue) ops |= V::kImageOffset, extra.push_back(offsetVec);
            if (compare != V::kNoValue) ops |= V::kImageCompare, extra.push_back(compare);
            if (clampLod != V::kNoValue && !gather && !(ops & V::kImageLod) && r.dim != V::TexDim::D1)
                ops |= V::kImageMinLod, extra.push_back(clampLod);
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
            // (a cube is six layers of faces: Metal's array size of a cube array counts cubes)
            if (cubeDim) comps[2] = r.dim == V::TexDim::CubeArray ? bin(V::Op::Mul, comps[2], u(6), U) : u(6);
            unsigned k = 0;
            for (unsigned c = 0; c < 4; ++c)
                if (dmask >> c & 1) writeV(dataReg + k++, comps[c]);
            return;
        }
        std::vector<ValueId> coord;
        // (loads, stores and atomics see a cube as a 2D array of faces: x, y, layer = slice * 6 + face)
        unsigned icw = cubeDim ? 2 : cw;
        for (unsigned k = 0; k < icw; ++k) coord.push_back(addrU());
        if (cubeDim) {
            ValueId layer = addrU();
            if (r.dim == V::TexDim::CubeArray) {
                coord.push_back(bin(V::Op::Rem, layer, u(6), U));
                coord.push_back(bin(V::Op::Div, layer, u(6), U));
            } else {
                coord.push_back(layer);
            }
        } else if (arrayed) {
            coord.push_back(addrU());
        }
        ValueId mip = name.find("_mip") != std::string_view::npos ? addrU() : V::kNoValue;
        // _pck forms move the raw channel values of the format, not the converted ones (needs the T#'s format)
        bool pck = name.find("_pck") != std::string_view::npos;
        unsigned fmtBits = 0, fmtNum = 0xff;
        if (pck) {
            for (Binding& bd : out.bindings)
                if (bd.slot == tex) bd.needsFormat = true;
            if (in.describe)
                if (auto d = in.describe(*ts)) fmtBits = d->texBits, fmtNum = d->texNumFormat;
            if (!fmtBits || fmtNum == 0xff) {
                note(std::string(name) + " without a known texture format moves the converted texel values");
                pck = false;
            }
        }
        V::Type ct = coord.size() == 1 ? U : V::vec(U, uint8_t(coord.size()));
        ValueId cv = coord.size() == 1 ? coord[0] : b->construct(coord, ct);
        V::Type texel(r.elem, 4);
        if (atomic) {
            // 32-bit integer atomics: native ops where the IR has them, a compare-exchange loop for the others
            // (inc / dec / rsub and the float compare-swap / min / max, which work on the texel's bits).
            std::string op(name.substr(std::strlen("image_atomic_")));
            V::Op aop = V::Op::ImageAtomicAdd;
            bool native = true;
            DsKind kind = DsKind::Add;
            if (op == "add") aop = V::Op::ImageAtomicAdd, kind = DsKind::Add;
            else if (op == "sub") aop = V::Op::ImageAtomicSub, kind = DsKind::Sub;
            else if (op == "smin" || op == "umin") aop = V::Op::ImageAtomicMin, kind = op == "smin" ? DsKind::MinI : DsKind::MinU;
            else if (op == "smax" || op == "umax") aop = V::Op::ImageAtomicMax, kind = op == "smax" ? DsKind::MaxI : DsKind::MaxU;
            else if (op == "and") aop = V::Op::ImageAtomicAnd, kind = DsKind::And;
            else if (op == "or") aop = V::Op::ImageAtomicOr, kind = DsKind::Or;
            else if (op == "xor") aop = V::Op::ImageAtomicXor, kind = DsKind::Xor;
            else if (op == "swap") aop = V::Op::ImageAtomicExchange, kind = DsKind::Xchg;
            else if (op == "cmpswap") aop = V::Op::ImageAtomicCompareExchange, kind = DsKind::Cmpst;
            else {
                native = false;
                if (op == "rsub") kind = DsKind::Rsub;
                else if (op == "inc") kind = DsKind::Inc;
                else if (op == "dec") kind = DsKind::Dec;
                else if (op == "fcmpswap") kind = DsKind::CmpstF;
                else if (op == "fmin") kind = DsKind::MinF;
                else if (op == "fmax") kind = DsKind::MaxF;
                else return error("unsupported image atomic " + std::string(name));
            }
            V::Type t(r.elem, 1);
            // (umin on an i32 texture would compare the other way: the texture's type follows the signed ops, see
            // resource(); those run as a compare-exchange loop on the bits)
            if ((op == "umin" || op == "umax") && r.elem == V::Scalar::I32) native = false;
            const bool signedElem = r.elem == V::Scalar::I32;
            auto val = [&](unsigned k) {
                ValueId v = b->loadVar(vVar(dataReg + k));
                return signedElem ? asI(v) : v;
            };
            if (writesExec) b->beginIf(b->loadVar(execV()));
            ValueId old;
            if (native) {
                V::Inst q{aop, t, V::kNoValue, false, {cv}, tex, 0, {}, 0};
                // cmpswap: data[0] the new value, data[1] the comparand
                if (aop == V::Op::ImageAtomicCompareExchange) q.args.push_back(val(1)), q.args.push_back(val(0));
                else q.args.push_back(val(0));
                old = b->emit(q);
            } else {
                // loop { old = texel; if (compare_exchange(old, f(old)) == old) break; }   (the data are read first)
                const bool cmp = kind == DsKind::CmpstF;
                ValueId d0 = b->loadVar(vVar(dataReg)), d1 = cmp ? b->loadVar(vVar(dataReg + 1)) : u(0);
                uint32_t got = b->variable(t);
                b->beginLoop();
                ValueId cur = b->imageFetch(tex, cv, texel);
                ValueId oldv = b->extract(cur, 0, texel.element());
                ValueId oldBits = signedElem ? bit(oldv, U) : oldv;
                // (compare-swap: the comparand is data[1], the new value data[0], as the DS ops take them)
                ValueId next = cmp ? dsCombine32(kind, oldBits, d1, d0) : dsCombine32(kind, oldBits, d0, u(0));
                V::Inst q{V::Op::ImageAtomicCompareExchange, t, V::kNoValue, false, {cv, oldv, signedElem ? asI(next) : next}, tex, 0, {}, 0};
                ValueId prev = b->emit(q);
                b->beginIf(bin(V::Op::Eq, prev, oldv, V::kBool));
                b->storeVar(got, oldv);
                b->breakLoop();
                b->end();
                b->end();
                old = b->loadVar(got);
            }
            if (x.mem.glc) b->storeVar(vVar(dataReg), signedElem ? bit(old, U) : old);
            if (writesExec) b->end();
            return;
        }
        if (store) {
            std::vector<ValueId> comps;
            unsigned k = 0;
            if (writesExec) b->beginIf(b->loadVar(execV()));
            // Components outside dmask keep what the texel holds.
            ValueId current = dmask != 0xf ? b->imageFetch(tex, cv, texel) : V::kNoValue;
            for (unsigned c = 0; c < 4; ++c) {
                if (dmask >> c & 1) {
                    ValueId raw = b->loadVar(vVar(dataReg + k++));
                    if (pck) raw = pckIn(raw, r.elem, fmtBits, fmtNum);
                    comps.push_back(r.elem == V::Scalar::U32 ? raw : bit(raw, texel.element()));
                } else {
                    comps.push_back(b->extract(current, c, texel.element()));
                }
            }
            ValueId value = b->construct(comps, texel);
            if (mip != V::kNoValue) {
                if (dmask != 0xf) note("a partial image_store_mip keeps the other channels of mip 0");
                b->emit(V::Inst{V::Op::ImageStore, V::kVoid, V::kNoValue, false, {cv, value, mip}, tex, 0, {}, V::kImageLod});
            } else {
                b->imageStore(tex, cv, value);
            }
            if (writesExec) b->end();
            return;
        }
        ValueId res;
        if (storage) {
            if (mip != V::kNoValue) note("image_load_mip on a texture the program writes reads mip 0");
            res = b->imageFetch(tex, cv, texel);
        } else {
            V::Inst q{V::Op::ImageFetch, texel, V::kNoValue, false, {cv}, tex, 0, {}, 0};
            if (mip != V::kNoValue) q.args.push_back(mip), q.image = V::kImageLod;
            res = b->emit(q);
        }
        if (pck) {
            unsigned k = 0;
            for (unsigned c = 0; c < 4; ++c) {
                if (!(dmask >> c & 1)) continue;
                writeV(dataReg + k++, pckOut(b->extract(res, c, texel.element()), r.elem, fmtBits, fmtNum, name.find("_pck_sgn") != std::string_view::npos));
            }
            if (x.mem.tfe || x.mem.lwe) writeV(dataReg + k, u(0));
            return;
        }
        writeTexel(res, texel);
    }

    // ---- LDS -----------------------------------------------------------------------------------

    void liftDs(size_t idx) {
        const Instruction& x = prog[idx].in;
        std::string_view name = mnemonic(x.op);
        auto U = V::kU32;
        if (name == "ds_swizzle_b32") return swizzle(x);
        if (x.mem.gds) {
            // Append/consume buffers: a counter in GDS at M0[15:0] + offset. Each lane is its own
            // wave here (v_mbcnt lifts to 0), so each lane takes one entry: append returns the
            // counter before, consume the counter after.
            if (name == "ds_append" || name == "ds_consume") {
                DescriptorSource gds;
                gds.kind = DescriptorSource::Gds;
                const uint32_t slot = resource(V::ResourceKind::Buffer, gds, 0, V::Access::ReadWrite, "gds");
                const bool append = name == "ds_append";
                ValueId at = bin(V::Op::And, b->loadVar(m0V()), u(0xfffc), U);
                at = bin(V::Op::Add, at, u(x.mem.offset & 0xfffc), U);
                auto body = [&] {
                    std::vector<ValueId> one{u(1)};
                    ValueId old = b->atomic(append ? V::Op::AtomicAdd : V::Op::AtomicSub, slot, at, one, U);
                    writeV(x.dst.index, append ? old : bin(V::Op::Sub, old, u(1), U));
                };
                if (writesExec) b->beginIf(b->loadVar(execV()));
                body();
                if (writesExec) b->end();
                return;
            }
            return error("GDS access is not supported (" + std::string(name) + ")");
        }
        // Outside compute, LDS is per-lane memory here (privateLds): Sony's compiler spills
        // registers of pixel shaders to LDS at lane index * 4 + offset, and the lane index
        // (v_mbcnt) lifts to 0. Addresses are clamped into the array.
        const bool priv = ldsPrivEnd != 0;
        if (m.sharedBytes == 0 && !priv) return error("LDS access in a program without LDS (" + std::string(name) + ")");
        uint32_t lds = priv ? V::kTargetPrivate : V::kTargetShared;
        ValueId base = x.src[0].kind != OperandKind::None ? b->loadVar(vVar(x.src[0].index)) : u(0);
        auto at = [&](uint32_t off) {
            ValueId a = bin(V::Op::Add, base, u(off), U);
            return priv ? bin(V::Op::Min, a, u(ldsPrivEnd - 4), U) : a;
        };
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
            if (name.find("src2") != std::string_view::npos) return dsRmw(x, name, priv, lds, at);
            guarded([&] {
                if (two) {
                    for (unsigned k = 0; k < esz / 4; ++k) b->store(lds, at(off0 * mul + 4 * k), reg(x.src[1], k));
                    for (unsigned k = 0; k < esz / 4; ++k) b->store(lds, at(off1 * mul + 4 * k), reg(x.src[2], k));
                } else if ((name == "ds_write_b8" || name == "ds_write_b16") && priv) {
                    // (per lane: a plain read-modify-write)
                    ValueId a = at(x.mem.offset), word = bin(V::Op::And, a, u(~3u), U);
                    ValueId sh = bin(V::Op::Shl, bin(V::Op::And, a, u(3), U), u(3), U);
                    uint32_t low = name == "ds_write_b8" ? 0xffu : 0xffffu;
                    ValueId keep = bin(V::Op::And, b->load(lds, word, U), b->unary(V::Op::Not, bin(V::Op::Shl, u(low), sh, U), U), U);
                    ValueId field = bin(V::Op::Shl, bin(V::Op::And, reg(x.src[1], 0), u(low), U), sh, U);
                    b->store(lds, word, bin(V::Op::Or, keep, field, U));
                } else if (name == "ds_write_b8" || name == "ds_write_b16") {
                    storeSubDword(lds, at(x.mem.offset), reg(x.src[1], 0), name == "ds_write_b8" ? 8 : 16);
                } else {
                    for (unsigned k = 0; k < x.src[1].count; ++k) b->store(lds, at(x.mem.offset + 4 * k), reg(x.src[1], k));
                }
            });
            return;
        }
        return dsRmw(x, name, priv, lds, at);
    }

    // ---- DS read-modify-write ops (atomics on LDS) ----------------------------------------------------------------

    enum class DsKind { Add, Sub, Rsub, Inc, Dec, MinI, MaxI, MinU, MaxU, And, Or, Xor, Mskor, Write, Cmpst, CmpstF, MinF,
                        MaxF, Xchg, Wrap, Xchg2, Condxchg };
    struct DsOpInfo {
        DsKind kind = DsKind::Add;
        bool wide = false, rtn = false, src2 = false, st64 = false, isF = false;
    };
    static bool parseDsOp(std::string_view name, DsOpInfo& d) {
        if (!name.starts_with("ds_")) return false;
        std::string n(name.substr(3));
        size_t us = n.rfind('_');
        if (us == std::string::npos) return false;
        std::string ty = n.substr(us + 1);
        n.resize(us);
        d.wide = ty.size() == 3 && ty.substr(1) == "64";
        d.isF = ty[0] == 'f';
        bool isI = ty[0] == 'i';
        auto strip = [&](const char* tag) {
            size_t at = n.find(tag);
            if (at == std::string::npos) return false;
            n.erase(at, std::strlen(tag));
            return true;
        };
        d.rtn = strip("_rtn");
        d.src2 = strip("_src2");
        static const std::map<std::string, DsKind> ops = {
            {"add", DsKind::Add}, {"sub", DsKind::Sub}, {"rsub", DsKind::Rsub}, {"inc", DsKind::Inc}, {"dec", DsKind::Dec},
            {"and", DsKind::And}, {"or", DsKind::Or}, {"xor", DsKind::Xor}, {"mskor", DsKind::Mskor}, {"write", DsKind::Write},
            {"wrxchg", DsKind::Xchg}, {"wrxchg2", DsKind::Xchg2}, {"wrxchg2st64", DsKind::Xchg2}, {"wrap", DsKind::Wrap},
            {"condxchg32", DsKind::Condxchg}};
        if (n == "min") d.kind = d.isF ? DsKind::MinF : isI ? DsKind::MinI : DsKind::MinU;
        else if (n == "max") d.kind = d.isF ? DsKind::MaxF : isI ? DsKind::MaxI : DsKind::MaxU;
        else if (n == "cmpst") d.kind = d.isF ? DsKind::CmpstF : DsKind::Cmpst;
        else if (auto it = ops.find(n); it != ops.end()) d.kind = it->second;
        else return false;
        d.st64 = n == "wrxchg2st64";
        return true;
    }

    struct P64 {
        ValueId lo, hi;
    };
    P64 add64(P64 a, P64 c) {
        auto U = V::kU32;
        ValueId lo = bin(V::Op::Add, a.lo, c.lo, U);
        ValueId carry = b2u(bin(V::Op::Lt, lo, a.lo, V::kBool));
        return {lo, bin(V::Op::Add, bin(V::Op::Add, a.hi, c.hi, U), carry, U)};
    }
    P64 sub64(P64 a, P64 c) {
        auto U = V::kU32;
        ValueId borrow = b2u(bin(V::Op::Lt, a.lo, c.lo, V::kBool));
        return {bin(V::Op::Sub, a.lo, c.lo, U), bin(V::Op::Sub, bin(V::Op::Sub, a.hi, c.hi, U), borrow, U)};
    }
    ValueId ult64(P64 a, P64 c) {  // a < c
        ValueId hiLt = bin(V::Op::Lt, a.hi, c.hi, V::kBool), hiEq = bin(V::Op::Eq, a.hi, c.hi, V::kBool);
        return bin(V::Op::Or, hiLt, bin(V::Op::And, hiEq, bin(V::Op::Lt, a.lo, c.lo, V::kBool), V::kBool), V::kBool);
    }
    ValueId slt64(P64 a, P64 c) {
        ValueId hiLt = bin(V::Op::Lt, asI(a.hi), asI(c.hi), V::kBool), hiEq = bin(V::Op::Eq, a.hi, c.hi, V::kBool);
        return bin(V::Op::Or, hiLt, bin(V::Op::And, hiEq, bin(V::Op::Lt, a.lo, c.lo, V::kBool), V::kBool), V::kBool);
    }
    ValueId eq64(P64 a, P64 c) {
        return bin(V::Op::And, bin(V::Op::Eq, a.lo, c.lo, V::kBool), bin(V::Op::Eq, a.hi, c.hi, V::kBool), V::kBool);
    }
    // f64 comparisons on the bit patterns (no f64 arithmetic needed): a NaN compares false with everything, +0 == -0,
    // otherwise the sign-magnitude order (denormals are compared as they are, as with denormals allowed in MODE).
    ValueId isNan64(P64 a) {
        auto U = V::kU32;
        ValueId mag = bin(V::Op::And, a.hi, u(0x7fffffffu), U);
        return bin(V::Op::Or, bin(V::Op::Gt, mag, u(0x7ff00000u), V::kBool),
                   bin(V::Op::And, bin(V::Op::Eq, mag, u(0x7ff00000u), V::kBool), nz(a.lo), V::kBool), V::kBool);
    }
    ValueId isZero64(P64 a) {
        return bin(V::Op::And, bin(V::Op::Eq, bin(V::Op::And, a.hi, u(0x7fffffffu), V::kU32), u(0), V::kBool),
                   bin(V::Op::Eq, a.lo, u(0), V::kBool), V::kBool);
    }
    P64 orderKey64(P64 a) {  // monotonic unsigned key of the double
        ValueId neg = nz(bin(V::Op::Shr, a.hi, u(31), V::kU32));
        P64 inv = not64(a), pos{a.lo, bin(V::Op::Or, a.hi, u(0x80000000u), V::kU32)};
        return sel64(neg, inv, pos);
    }
    ValueId flt64(P64 a, P64 c) {  // a < c as doubles
        ValueId ok = notB(bin(V::Op::Or, bin(V::Op::Or, isNan64(a), isNan64(c), V::kBool),
                              bin(V::Op::And, isZero64(a), isZero64(c), V::kBool), V::kBool));
        return bin(V::Op::And, ok, ult64(orderKey64(a), orderKey64(c)), V::kBool);
    }
    ValueId feq64(P64 a, P64 c) {  // a == c as doubles
        ValueId bothZero = bin(V::Op::And, isZero64(a), isZero64(c), V::kBool);
        ValueId same = bin(V::Op::And, eq64(a, c), notB(isNan64(a)), V::kBool);
        return bin(V::Op::Or, same, bothZero, V::kBool);
    }
    P64 sel64(ValueId c, P64 a, P64 d) { return {sel(c, a.lo, d.lo, V::kU32), sel(c, a.hi, d.hi, V::kU32)}; }
    P64 map64(V::Op o, P64 a, P64 c) { return {bin(o, a.lo, c.lo, V::kU32), bin(o, a.hi, c.hi, V::kU32)}; }
    P64 not64(P64 a) { return {b->unary(V::Op::Not, a.lo, V::kU32), b->unary(V::Op::Not, a.hi, V::kU32)}; }

    // The new value of a 32-bit LDS location: old, DATA0, DATA1 (as the SI/CI ISA defines each op).
    ValueId dsCombine32(DsKind k, ValueId old, ValueId d0, ValueId d1) {
        auto U = V::kU32;
        auto F = V::kF32;
        switch (k) {
            case DsKind::Add: return bin(V::Op::Add, old, d0, U);
            case DsKind::Sub: return bin(V::Op::Sub, old, d0, U);
            case DsKind::Rsub: return bin(V::Op::Sub, d0, old, U);
            case DsKind::Inc: return sel(bin(V::Op::Ge, old, d0, V::kBool), u(0), bin(V::Op::Add, old, u(1), U), U);
            case DsKind::Dec:
                return sel(bin(V::Op::Or, bin(V::Op::Eq, old, u(0), V::kBool), bin(V::Op::Gt, old, d0, V::kBool), V::kBool), d0,
                           bin(V::Op::Sub, old, u(1), U), U);
            case DsKind::MinI: return bit(bin(V::Op::Min, asI(old), asI(d0), V::kI32), U);
            case DsKind::MaxI: return bit(bin(V::Op::Max, asI(old), asI(d0), V::kI32), U);
            case DsKind::MinU: return bin(V::Op::Min, old, d0, U);
            case DsKind::MaxU: return bin(V::Op::Max, old, d0, U);
            case DsKind::And: return bin(V::Op::And, old, d0, U);
            case DsKind::Or: return bin(V::Op::Or, old, d0, U);
            case DsKind::Xor: return bin(V::Op::Xor, old, d0, U);
            case DsKind::Mskor: return bin(V::Op::Or, bin(V::Op::And, old, b->unary(V::Op::Not, d0, U), U), d1, U);
            case DsKind::Write:
            case DsKind::Xchg:
            case DsKind::Xchg2: return d0;
            case DsKind::Cmpst: return sel(bin(V::Op::Eq, old, d0, V::kBool), d1, old, U);
            case DsKind::CmpstF: return sel(bin(V::Op::Eq, asF(old), asF(d0), V::kBool), d1, old, U);
            case DsKind::MinF: return sel(bin(V::Op::Lt, asF(d0), asF(old), V::kBool), d0, old, U);
            case DsKind::MaxF: return sel(bin(V::Op::Gt, asF(d0), asF(old), V::kBool), d0, old, U);
            case DsKind::Wrap: return sel(bin(V::Op::Ge, old, d0, V::kBool), bin(V::Op::Sub, old, d0, U), bin(V::Op::Add, old, d1, U), U);
            case DsKind::Condxchg: break;
        }
        (void)F;
        return old;
    }
    P64 dsCombine64(DsKind k, P64 old, P64 d0, P64 d1) {
        auto U = V::kU32;
        P64 one{u(1), u(0)}, zero{u(0), u(0)};
        switch (k) {
            case DsKind::Add: return add64(old, d0);
            case DsKind::Sub: return sub64(old, d0);
            case DsKind::Rsub: return sub64(d0, old);
            case DsKind::Inc: return sel64(notB(ult64(old, d0)), zero, add64(old, one));
            case DsKind::Dec: return sel64(bin(V::Op::Or, eq64(old, zero), ult64(d0, old), V::kBool), d0, sub64(old, one));
            case DsKind::MinI: return sel64(slt64(d0, old), d0, old);
            case DsKind::MaxI: return sel64(slt64(old, d0), d0, old);
            case DsKind::MinU: return sel64(ult64(d0, old), d0, old);
            case DsKind::MaxU: return sel64(ult64(old, d0), d0, old);
            case DsKind::And: return map64(V::Op::And, old, d0);
            case DsKind::Or: return map64(V::Op::Or, old, d0);
            case DsKind::Xor: return map64(V::Op::Xor, old, d0);
            case DsKind::Mskor: return map64(V::Op::Or, map64(V::Op::And, old, not64(d0)), d1);
            case DsKind::Write:
            case DsKind::Xchg:
            case DsKind::Xchg2: return d0;
            case DsKind::Cmpst: return sel64(eq64(old, d0), d1, old);
            case DsKind::CmpstF: return sel64(feq64(old, d0), d1, old);
            case DsKind::MinF: return sel64(flt64(d0, old), d0, old);
            case DsKind::MaxF: return sel64(flt64(old, d0), d0, old);
            default: break;
        }
        (void)U;
        return old;
    }

    void dsRmw(const Instruction& x, std::string_view name, bool priv, uint32_t lds, const std::function<ValueId(uint32_t)>& at) {
        auto U = V::kU32;
        DsOpInfo d;
        if (!parseDsOp(name, d)) return error("unsupported LDS operation " + std::string(name));
        if (name == "ds_condxchg32_rtn_b64")
            note("ds_condxchg32_rtn_b64: each dword is written (without its top bit) when its top bit is set; the old dwords return");
        const unsigned words = d.wide ? 2 : 1;
        auto reg = [&](const Operand& o, unsigned k) { return b->loadVar(vVar(o.index + k)); };
        auto pairOf = [&](const Operand& o) { return P64{reg(o, 0), reg(o, 1)}; };
        auto guarded = [&](auto&& body) {
            if (writesExec) b->beginIf(b->loadVar(execV()));
            body();
            if (writesExec) b->end();
        };
        const bool hasRet = x.dst.kind != OperandKind::None;
        // (the results land after every operand has been read: the destination may overlap a data register)
        std::vector<std::pair<unsigned, ValueId>> pending;
        auto ret = [&](unsigned k, ValueId v) {
            if (hasRet) pending.emplace_back(k, v);
        };
        auto flush = [&] {
            for (auto& [k, v] : pending) b->storeVar(vVar(x.dst.index + k), v);
            pending.clear();
        };
        // The addresses: one (plus the 16-bit offset), two scaled 8-bit offsets for wrxchg2, A and B for src2 forms.
        std::vector<ValueId> addrs;
        if (d.kind == DsKind::Xchg2) {
            uint32_t mul = words * 4 * (d.st64 ? 64 : 1);
            addrs = {at((x.mem.offset & 0xff) * mul), at((x.mem.offset >> 8) * mul)};
        } else if (d.src2) {
            ValueId base = x.src[0].kind != OperandKind::None ? b->loadVar(vVar(x.src[0].index)) : u(0);
            addrs.push_back(priv ? bin(V::Op::Min, base, u(ldsPrivEnd - 4), U) : base);
            // B = A + 4 * (offset1[7] ? {A[31], A[31:17]} : {offset1[6], offset1[6:0], offset0}), a signed 16-bit count
            uint32_t off1 = x.mem.offset >> 8;
            ValueId disp;
            if (off1 & 0x80) {
                ValueId top = bin(V::Op::Shr, base, u(31), U);
                ValueId lowBits = bin(V::Op::And, bin(V::Op::Shr, base, u(17), U), u(0x7fff), U);
                disp = bin(V::Op::Or, bin(V::Op::Shl, top, u(15), U), lowBits, U);
            } else {
                disp = u(((off1 & 0x40) << 9) | ((off1 & 0x7f) << 8) | (x.mem.offset & 0xff));
            }
            disp = bfe(disp, u(0), u(16), true);  // sign-extended
            ValueId bAt = bin(V::Op::Add, base, bin(V::Op::Shl, disp, u(2), U), U);
            addrs.push_back(priv ? bin(V::Op::Min, bAt, u(ldsPrivEnd - 4), U) : bAt);
        } else {
            addrs.push_back(at(x.mem.offset));
        }
        auto word = [&](ValueId a, unsigned k) { return k ? bin(V::Op::Add, a, u(4 * k), U) : a; };
        // Plain read-modify-write of one location (nothing else touches it: private memory, or serialised lanes).
        auto plain0 = [&](ValueId a0, ValueId a1) {
            if (d.kind == DsKind::Xchg2) {
                for (unsigned loc = 0; loc < 2; ++loc) {
                    const Operand& dop = loc ? x.src[2] : x.src[1];
                    ValueId a = loc ? a1 : a0;
                    for (unsigned k = 0; k < words; ++k) {
                        ret(loc * words + k, b->load(lds, word(a, k), U));
                        b->store(lds, word(a, k), reg(dop, k));
                    }
                }
                return;
            }
            if (d.kind == DsKind::Condxchg) {
                for (unsigned k = 0; k < 2; ++k) {
                    ValueId old = b->load(lds, word(a0, k), U), nv = reg(x.src[1], k);
                    ret(k, old);
                    ValueId set = nz(bin(V::Op::And, nv, u(0x80000000u), U));
                    b->store(lds, word(a0, k), sel(set, bin(V::Op::And, nv, u(0x7fffffffu), U), old, U));
                }
                return;
            }
            if (d.wide) {
                P64 old{b->load(lds, a0, U), b->load(lds, word(a0, 1), U)};
                P64 d0{u(0), u(0)}, d1{u(0), u(0)};
                if (d.src2) d0 = P64{b->load(lds, a1, U), b->load(lds, word(a1, 1), U)};
                else if (x.src[1].kind != OperandKind::None) d0 = pairOf(x.src[1]);
                if (!d.src2 && x.src[2].kind != OperandKind::None) d1 = pairOf(x.src[2]);
                P64 nv = dsCombine64(d.kind, old, d0, d1);
                ret(0, old.lo), ret(1, old.hi);
                b->store(lds, a0, nv.lo);
                b->store(lds, word(a0, 1), nv.hi);
                return;
            }
            ValueId old = b->load(lds, a0, U);
            ValueId d0 = d.src2 ? b->load(lds, a1, U) : x.src[1].kind != OperandKind::None ? reg(x.src[1], 0) : u(0);
            ValueId d1 = !d.src2 && x.src[2].kind != OperandKind::None ? reg(x.src[2], 0) : u(0);
            ret(0, old);
            b->store(lds, a0, dsCombine32(d.kind, old, d0, d1));
        };
        auto plain = [&](ValueId a0, ValueId a1) {
            plain0(a0, a1);
            flush();
        };
        if (priv) {
            guarded([&] { plain(addrs[0], addrs.size() > 1 ? addrs[1] : addrs[0]); });
            return;
        }
        if (d.wide) {
            // 64-bit values have no atomics in threadgroup memory: the lanes of the wave take turns, one per round,
            // with a barrier between rounds (the workgroup is one wave, in lockstep).
            needWave();
            if (!waveMode) return error("64-bit LDS read-modify-write (" + std::string(name) + ") needs a one-wave compute workgroup");
            uint32_t round = b->variable(U);
            b->storeVar(round, u(0));
            b->beginLoop();
            ValueId r = b->loadVar(round);
            b->beginIf(bin(V::Op::Ge, r, u(64), V::kBool));
            b->breakLoop();
            b->end();
            ValueId mine = bin(V::Op::Eq, bin(V::Op::And, laneIndex, u(63), U), r, V::kBool);
            b->beginIf(writesExec ? bin(V::Op::And, mine, b->loadVar(execV()), V::kBool) : mine);
            plain(addrs[0], addrs.size() > 1 ? addrs[1] : addrs[0]);
            b->end();
            b->barrier();
            b->storeVar(round, bin(V::Op::Add, r, u(1), U));
            b->end();
            return;
        }
        // 32-bit: native atomics where the IR has them, a compare-exchange loop otherwise.
        V::Op native = V::Op::Add;
        bool haveNative = true;
        switch (d.kind) {
            case DsKind::Add: native = V::Op::AtomicAdd; break;
            case DsKind::Sub: native = V::Op::AtomicSub; break;
            case DsKind::MinI:
            case DsKind::MinU: native = V::Op::AtomicMin; break;
            case DsKind::MaxI:
            case DsKind::MaxU: native = V::Op::AtomicMax; break;
            case DsKind::And: native = V::Op::AtomicAnd; break;
            case DsKind::Or: native = V::Op::AtomicOr; break;
            case DsKind::Xor: native = V::Op::AtomicXor; break;
            case DsKind::Xchg: native = V::Op::AtomicExchange; break;
            case DsKind::Cmpst: native = V::Op::AtomicCompareExchange; break;
            default: haveNative = false; break;
        }
        auto cas = [&](ValueId a, const std::function<ValueId(ValueId)>& next, unsigned retIndex) {
            // loop { old = *a; if (cmpxchg(a, old, next(old)) == old) break; }
            uint32_t got = b->variable(U);
            b->beginLoop();
            ValueId old = b->load(lds, a, U);
            ValueId nv = next(old);
            ValueId cmpNew[] = {old, nv};
            ValueId prev = b->atomic(V::Op::AtomicCompareExchange, lds, a, cmpNew, U);
            b->beginIf(bin(V::Op::Eq, prev, old, V::kBool));
            b->storeVar(got, old);
            b->breakLoop();
            b->end();
            b->end();
            ret(retIndex, b->loadVar(got));
        };
        guarded([&] {
            struct Flush {
                std::function<void()> f;
                ~Flush() { f(); }
            } flushAtEnd{flush};
            if (d.kind == DsKind::Xchg2) {
                ValueId v0[] = {reg(x.src[1], 0)}, v1[] = {reg(x.src[2], 0)};
                ret(0, b->atomic(V::Op::AtomicExchange, lds, addrs[0], v0, U));
                ret(1, b->atomic(V::Op::AtomicExchange, lds, addrs[1], v1, U));
                return;
            }
            if (d.kind == DsKind::Condxchg) {
                for (unsigned k = 0; k < 2; ++k) {
                    ValueId nv = reg(x.src[1], k);
                    cas(word(addrs[0], k), [&](ValueId old) {
                        return sel(nz(bin(V::Op::And, nv, u(0x80000000u), U)), bin(V::Op::And, nv, u(0x7fffffffu), U), old, U);
                    }, k);
                }
                return;
            }
            if (d.src2) {
                // MEM[A] = f(MEM[A], MEM[B]) (B is only read)
                ValueId bv = b->load(lds, addrs[1], U);
                cas(addrs[0], [&](ValueId old) { return dsCombine32(d.kind, old, bv, u(0)); }, 0);
                return;
            }
            ValueId d0 = x.src[1].kind != OperandKind::None ? reg(x.src[1], 0) : u(0);
            ValueId d1 = x.src[2].kind != OperandKind::None ? reg(x.src[2], 0) : u(0);
            if (d.kind == DsKind::Write) {
                b->store(lds, addrs[0], d0);
                return;
            }
            if (haveNative) {
                bool sign = d.kind == DsKind::MinI || d.kind == DsKind::MaxI;
                V::Type t = sign ? V::kI32 : U;
                std::vector<ValueId> vals = native == V::Op::AtomicCompareExchange ? std::vector<ValueId>{d0, d1} : std::vector<ValueId>{d0};
                if (sign) vals[0] = asI(vals[0]);
                ValueId r = b->atomic(native, lds, addrs[0], vals, t);
                ret(0, sign ? bit(r, U) : r);
                return;
            }
            cas(addrs[0], [&](ValueId old) { return dsCombine32(d.kind, old, d0, d1); }, 0);
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
        if (lane == V::kNoValue && waveMode) {
            // The lane read: quad permutation (bit 15 set) or bit masks on the lane id within each half of the wave
            // (and / or / xor of the low five bits; lane bit 5 is kept). The value is the source lane's register,
            // whether or not that lane is active.
            ValueId id = bin(V::Op::And, laneIndex, u(63), U);
            ValueId src;
            if (off & 0x8000) {
                ValueId shift = bin(V::Op::Shl, bin(V::Op::And, id, u(3), U), u(1), U);
                src = bin(V::Op::Or, bin(V::Op::And, id, u(~3u), U), bin(V::Op::And, bin(V::Op::Shr, u(off & 0xff), shift, U), u(3), U), U);
            } else {
                uint32_t andMask = off & 0x1f, orMask = (off >> 5) & 0x1f, xorMask = (off >> 10) & 0x1f;
                ValueId l5 = bin(V::Op::Xor, bin(V::Op::Or, bin(V::Op::And, id, u(andMask), U), u(orMask), U), u(xorMask), U);
                src = bin(V::Op::Or, bin(V::Op::And, id, u(0x20), U), l5, U);
            }
            return writeV(x.dst.index, b->emit(V::Inst{V::Op::SubgroupShuffle, U, V::kNoValue, false, {v, src}, 0, 0, {}, 0}));
        }
        if (lane == V::kNoValue) {
            note("a ds_swizzle_b32 pattern beyond the quad (or outside pixel/compute) returns this lane's own value");
            needWave();
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
        if (attr < in.psInputCntl.size()) {
            uint32_t cntl = in.psInputCntl[attr];
            io.location = cntl & 0x1F;
            if (cntl & (1u << 10)) interp = V::Interp::Flat;
        }
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
        // SPI_PS_INPUT_CNTL bit 5: the attribute is a constant (DEFAULT_VAL, bits 8-9), not a parameter.
        if (attr < in.psInputCntl.size() && (in.psInputCntl[attr] & 0x20)) {
            if (x.op == G::v_interp_p1_f32) return;
            const unsigned dv = (in.psInputCntl[attr] >> 8) & 3;
            const float xyz = dv >= 2 ? 1.0f : 0.0f, w = (dv & 1) ? 1.0f : 0.0f;
            return writeV(x.dst.index, bit(f(chan == 3 ? w : xyz), V::kU32));
        }
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
        bool vector = x.encoding >= Encoding::VOP2 && x.encoding != Encoding::SMRD;
        if (usesVskip && vector) {
            // VSKIP set: the wave skips vector instructions.
            b->beginIf(notB(b->loadVar(vskipV())));
            liftOneBody(idx);
            b->end();
            return;
        }
        liftOneBody(idx);
    }

    void liftOneBody(size_t idx) {
        const Instruction& x = prog[idx].in;
        curOp = x.op;
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
            // (wave mode: the whole wave branches together, on whether any lane has the bit)
            case G::s_cbranch_vccz: return notB(waveMode ? anyLane(b->loadVar(vccV())) : b->loadVar(vccV()));
            case G::s_cbranch_vccnz: return waveMode ? anyLane(b->loadVar(vccV())) : b->loadVar(vccV());
            case G::s_cbranch_execz: return notB(waveMode ? anyLane(b->loadVar(execV())) : b->loadVar(execV()));
            case G::s_cbranch_execnz: return waveMode ? anyLane(b->loadVar(execV())) : b->loadVar(execV());
            case G::s_cbranch_g_fork:
            case G::s_cbranch_i_fork: {
                if (waveMode) {
                    error("s_cbranch_g_fork / s_cbranch_i_fork in wave mode");
                    return b->constBool(false);
                }
                // Lanes whose mask bit (and EXEC bit) is set take the branch; the others fall through. Both
                // groups rejoin at the post-dominator, where the program's s_cbranch_join (a no-op here)
                // sits: the hardware's branch stack serialises the two paths, per-lane control flow does not
                // need it.
                const Instruction& fx = prog[bb.last].in;
                const Operand& m = fx.op == G::s_cbranch_g_fork ? fx.src[0] : fx.dst;
                return bin(V::Op::And, readMask(m), b->loadVar(execV()), V::kBool);
            }
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
        if (usesTime) b->storeVar(var(timeVar, V::kU32), u(0));
        if (usesVskip) b->storeVar(vskipV(), b->constBool(false));
        if (usesMode) b->storeVar(modeV(), u(in.modeRegister));
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
                    for (unsigned v = 1; v <= in.vgprCompCnt && v < 4; ++v) {
                        uint32_t rate = v < 3 ? in.instanceStepRate[v - 1] : 1;
                        b->storeVar(vVar(v), rate > 1 ? bin(V::Op::Div, iid, u(rate), U) : iid);
                    }
                    if (in.vgprCompCnt >= 1 && (!in.instanceStepRate[0] || (in.vgprCompCnt >= 2 && !in.instanceStepRate[1])))
                        note("instance step-rate VGPRs (v1, v2) hold the instance id");
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

    // LDS in a vertex or pixel shader: per-lane memory as large as the furthest DS offset reaches
    // (the lane part of the address lifts to 0, see liftDs).
    // Only the spill idiom is accepted: every DS address register holds the lane index (v_mbcnt),
    // optionally shifted or offset by a constant, as last written in program order. Anything
    // else could address another lane's slot or past the array, and is an error rather than a
    // silent clamp.
    void privateLds() {
        uint32_t end = 0;
        std::vector<bool> lane(256, false);  // VGPR last written with a lane-index expression
        auto isConst = [](const Operand& o) { return o.kind == OperandKind::Const || o.kind == OperandKind::Literal; };
        auto laneReg = [&](const Operand& o) { return o.kind == OperandKind::Vgpr && o.index < 256 && lane[o.index]; };
        for (const Decoded& d : prog) {
            const Instruction& x = d.in;
            std::string_view name = mnemonic(x.op);
            if (x.encoding == Encoding::DS && !x.mem.gds && name != "ds_swizzle_b32" && x.src[0].kind != OperandKind::None &&
                !laneReg(x.src[0])) {
                error("LDS outside compute with an address other than the lane index (" + std::string(name) + ")");
                return;
            }
            if (x.dst.kind == OperandKind::Vgpr) {
                // (v_mbcnt of a full mask: the lane index, which lifts to 0 outside compute)
                bool fullMbcnt = (name == "v_mbcnt_lo_u32_b32" || name == "v_mbcnt_hi_u32_b32") && isConst(x.src[0]) &&
                                 x.src[0].value == 0xffffffffu && (isConst(x.src[1]) ? x.src[1].value == 0 : laneReg(x.src[1]));
                bool derived = fullMbcnt ||
                               (name == "v_mov_b32" && laneReg(x.src[0])) ||
                               (name == "v_lshlrev_b32" && isConst(x.src[0]) && laneReg(x.src[1])) ||
                               (name == "v_lshl_b32" && laneReg(x.src[0]) && isConst(x.src[1])) ||
                               ((name == "v_add_i32" || name == "v_or_b32") &&
                                ((laneReg(x.src[0]) && isConst(x.src[1])) || (isConst(x.src[0]) && laneReg(x.src[1]))));
                for (unsigned k = 0; k < std::max<unsigned>(1, x.dst.count); ++k)
                    if (x.dst.index + k < 256) lane[x.dst.index + k] = derived && k == 0;
            }
            if (x.encoding != Encoding::DS || x.mem.gds) continue;
            if (name == "ds_swizzle_b32") continue;
            uint32_t reach = x.mem.offset + 16;
            if (name.find("read2") != std::string_view::npos || name.find("write2") != std::string_view::npos) {
                uint32_t esz = name.find("_b64") != std::string_view::npos ? 8 : 4;
                uint32_t mul = name.find("st64") != std::string_view::npos ? esz * 64 : esz;
                reach = std::max(x.mem.offset & 0xff, x.mem.offset >> 8) * mul + 16;
            }
            end = std::max(end, reach);
        }
        if (end == 0) return;
        // (zero-initialised per invocation: a large array would be a cost and register pressure)
        if (end > 4096) {
            error("LDS outside compute reaches " + std::to_string(end) + " bytes (per-lane memory up to 4096)");
            return;
        }
        m.privateBytes = ldsPrivEnd = (end + 15) / 16 * 16;
        note("LDS outside compute lifted as per-lane memory (register spills; lanes cannot exchange data through it)");
    }

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
        if (in.stage != ShaderStage::Compute) privateLds();
        if (!waveMode) {
            // Registers used as scalar spill slots (v_writelane): 64 dwords each of private memory.
            uint32_t at = (m.privateBytes + 15) & ~15u;
            for (const Decoded& d : prog)
                if (d.in.op == G::v_writelane_b32 && d.in.dst.kind == OperandKind::Vgpr && !spillBase.count(d.in.dst.index)) {
                    if (at + 256 > 4096) {
                        error("too many VGPRs used as scalar spill slots (v_writelane)");
                        break;
                    }
                    spillBase[d.in.dst.index] = at;
                    at += 256;
                }
            if (!spillBase.empty()) m.privateBytes = at;
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
        curMode = in.modeRegister;
        for (const Decoded& d : prog) {
            const Instruction& x = d.in;
            auto seeS = [&](const Operand& o) {
                if (o.kind == OperandKind::Sgpr && o.count) highestSgpr = std::max(highestSgpr, unsigned(o.code + o.count - 1));
            };
            seeS(x.dst);
            seeS(x.sdst);
            for (const Operand& o : x.src) seeS(o);
            if (x.op == G::s_setvskip) usesVskip = true;
            if (x.op == G::s_getreg_b32 || x.op == G::s_setreg_b32 || x.op == G::s_setreg_imm32_b32) usesMode = true;
            if (x.op == G::s_setreg_b32 && (x.simm16 & 0x3f) == 1) modeDynamic = true;
            if (x.op == G::s_memtime) usesTime = true;
        }
        highestSgpr = std::min(std::max(highestSgpr, in.userSgprs + 8u), 103u);
        if (curMode & 0xf) note("MODE rounding mode other than nearest-even is not honoured");
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
                if (auto it = rsrcSource.find(i); it != rsrcSource.end()) {
                    storageUses.insert(it->second);
                    if (n.starts_with("image_atomic"))
                        atomicUses[it->second] |= n == "image_atomic_smin" || n == "image_atomic_smax";
                }
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

namespace {
LiftResult liftWith(const LiftInput& input, bool wave, bool& wants) {
    LiftResult out;
    Lifter l(input, out, wave);
    l.run();
    out.errors.assign(l.errorSet.begin(), l.errorSet.end());
    out.notes.assign(l.noteSet.begin(), l.noteSet.end());
    wants = l.wantsWave;
    return out;
}
}  // namespace

// A first pass in the per-lane model; when it met something that needs a real wave (cross-lane reads, lane masks as
// data) and the workgroup is a single wave (compute, up to 64 invocations), the program is lifted again in wave
// mode. The wave-mode result is used when it lifts without errors.
LiftResult lift(const LiftInput& input) {
    bool wants = false;
    LiftResult out = liftWith(input, false, wants);
    uint64_t threads = uint64_t(std::max(1u, input.workgroupSize[0])) * std::max(1u, input.workgroupSize[1]) * std::max(1u, input.workgroupSize[2]);
    if (!wants || input.stage != ShaderStage::Compute || threads > 64) return out;
    bool ignored = false;
    LiftResult wave = liftWith(input, true, ignored);
    if (!wave.errors.empty()) {
        out.notes.push_back("wave mode failed: " + wave.errors.front());
        return out;
    }
    return wave;
}

}  // namespace vpgcn
