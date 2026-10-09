// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// x86-64 -> C. Every guest function becomes `static void fn_<addr>(VpCpu* cpu, uint32_t entry)`:
// its basic blocks are labels, branches are gotos, direct calls are C calls, and anything whose
// target is only known at run time (indirect call/jmp, a `ret` to an unexpected address) goes
// through vp_dispatch. The host compiler does the optimisation across the whole function.
#include "translate.h"

#include <Zydis/Zydis.h>

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <stdexcept>

namespace vpaot {
namespace {

// printf into a std::string of any length (a fixed buffer would cut long generated lines).
std::string vfmt(const char* f, va_list ap) {
    va_list again;
    va_copy(again, ap);
    char buf[512];
    const int n = vsnprintf(buf, sizeof buf, f, ap);
    if (n < 0) { va_end(again); return std::string(); }
    if ((size_t)n < sizeof buf) { va_end(again); return std::string(buf, (size_t)n); }
    std::string big((size_t)n + 1, '\0');
    vsnprintf(&big[0], big.size(), f, again);
    va_end(again);
    big.resize((size_t)n);
    return big;
}
std::string fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    std::string r = vfmt(f, ap);
    va_end(ap);
    return r;
}

std::string hex(uint64_t v) { return fmt("0x%" PRIx64, v); }

struct Decoder {
    ZydisDecoder dec;
    Decoder() { ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64); }
    bool decode(const Image& img, uint64_t addr, ZydisDecodedInstruction& insn,
                ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT]) const {
        if (!img.is_code(addr)) return false;
        const uint64_t avail = std::min<uint64_t>(15, img.end() - addr);
        return ZYAN_SUCCESS(ZydisDecoderDecodeFull(&dec, img.at(addr), avail, &insn, ops));
    }
};

bool is_jcc(ZydisMnemonic m) {
    switch (m) {
    case ZYDIS_MNEMONIC_JO: case ZYDIS_MNEMONIC_JNO: case ZYDIS_MNEMONIC_JB: case ZYDIS_MNEMONIC_JNB:
    case ZYDIS_MNEMONIC_JZ: case ZYDIS_MNEMONIC_JNZ: case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_JNBE:
    case ZYDIS_MNEMONIC_JS: case ZYDIS_MNEMONIC_JNS: case ZYDIS_MNEMONIC_JP: case ZYDIS_MNEMONIC_JNP:
    case ZYDIS_MNEMONIC_JL: case ZYDIS_MNEMONIC_JNL: case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_JNLE:
    case ZYDIS_MNEMONIC_JCXZ: case ZYDIS_MNEMONIC_JECXZ: case ZYDIS_MNEMONIC_JRCXZ:
        return true;
    default:
        return false;
    }
}

// Condition code 0..15 for Jcc/SETcc/CMOVcc, -1 otherwise.
int cc_of(ZydisMnemonic m) {
    switch (m) {
    case ZYDIS_MNEMONIC_JO: case ZYDIS_MNEMONIC_SETO: case ZYDIS_MNEMONIC_CMOVO: return 0;
    case ZYDIS_MNEMONIC_JNO: case ZYDIS_MNEMONIC_SETNO: case ZYDIS_MNEMONIC_CMOVNO: return 1;
    case ZYDIS_MNEMONIC_JB: case ZYDIS_MNEMONIC_SETB: case ZYDIS_MNEMONIC_CMOVB: return 2;
    case ZYDIS_MNEMONIC_JNB: case ZYDIS_MNEMONIC_SETNB: case ZYDIS_MNEMONIC_CMOVNB: return 3;
    case ZYDIS_MNEMONIC_JZ: case ZYDIS_MNEMONIC_SETZ: case ZYDIS_MNEMONIC_CMOVZ: return 4;
    case ZYDIS_MNEMONIC_JNZ: case ZYDIS_MNEMONIC_SETNZ: case ZYDIS_MNEMONIC_CMOVNZ: return 5;
    case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_SETBE: case ZYDIS_MNEMONIC_CMOVBE: return 6;
    case ZYDIS_MNEMONIC_JNBE: case ZYDIS_MNEMONIC_SETNBE: case ZYDIS_MNEMONIC_CMOVNBE: return 7;
    case ZYDIS_MNEMONIC_JS: case ZYDIS_MNEMONIC_SETS: case ZYDIS_MNEMONIC_CMOVS: return 8;
    case ZYDIS_MNEMONIC_JNS: case ZYDIS_MNEMONIC_SETNS: case ZYDIS_MNEMONIC_CMOVNS: return 9;
    case ZYDIS_MNEMONIC_JP: case ZYDIS_MNEMONIC_SETP: case ZYDIS_MNEMONIC_CMOVP: return 10;
    case ZYDIS_MNEMONIC_JNP: case ZYDIS_MNEMONIC_SETNP: case ZYDIS_MNEMONIC_CMOVNP: return 11;
    case ZYDIS_MNEMONIC_JL: case ZYDIS_MNEMONIC_SETL: case ZYDIS_MNEMONIC_CMOVL: return 12;
    case ZYDIS_MNEMONIC_JNL: case ZYDIS_MNEMONIC_SETNL: case ZYDIS_MNEMONIC_CMOVNL: return 13;
    case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_SETLE: case ZYDIS_MNEMONIC_CMOVLE: return 14;
    case ZYDIS_MNEMONIC_JNLE: case ZYDIS_MNEMONIC_SETNLE: case ZYDIS_MNEMONIC_CMOVNLE: return 15;
    default: return -1;
    }
}

bool ends_block(const ZydisDecodedInstruction& i) {
    switch (i.mnemonic) {
    case ZYDIS_MNEMONIC_RET: case ZYDIS_MNEMONIC_JMP: case ZYDIS_MNEMONIC_UD2: case ZYDIS_MNEMONIC_HLT:
    case ZYDIS_MNEMONIC_INT3: case ZYDIS_MNEMONIC_IRETQ:
        return true;
    default:
        return is_jcc(i.mnemonic) || i.mnemonic == ZYDIS_MNEMONIC_LOOP || i.mnemonic == ZYDIS_MNEMONIC_LOOPE ||
               i.mnemonic == ZYDIS_MNEMONIC_LOOPNE;
    }
}

// The absolute target of a relative branch/call, or 0 when the operand is not immediate.
uint64_t branch_target(const ZydisDecodedInstruction& i, const ZydisDecodedOperand* ops, uint64_t addr) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return 0;
    ZyanU64 t = 0;
    if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&i, &ops[0], addr, &t))) return 0;
    return t;
}

// ---- discovery -----------------------------------------------------------------------------

// Reads the targets of a jump table behind an indirect `jmp` at `jmp_addr`, from the instructions
// before it in the same block (`recent`, oldest first). Recognised shapes:
//   lea T(%rip), %rA ; movslq (%rA,%rB,4), %rC ; add %rA, %rC ; jmp *%rC      (PIE: 4-byte offsets)
//   jmp *T(,%rB,8)   /  mov T(,%rB,8), %rC ; jmp *%rC                        (absolute 8-byte entries)
// A `cmp $N, reg` before the jump bounds the table; otherwise entries are read while they decode.
std::vector<uint64_t> read_jump_table(const Image& img, const Decoder& dec, const std::vector<uint64_t>& recent,
                                      uint64_t jmp_addr, const ZydisDecodedInstruction& jmp,
                                      const ZydisDecodedOperand* jmp_ops,
                                      const std::map<ZydisRegister, uint64_t>& lea_by_reg) {
    std::vector<uint64_t> out;
    uint64_t table = 0;
    bool relative = false;
    uint64_t count = 0;
    ZydisRegister table_base = ZYDIS_REGISTER_NONE; // the register the table's address was read through
    auto consider_mem = [&](const ZydisDecodedOperand& op) {
        if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) return;
        if (op.mem.base == ZYDIS_REGISTER_NONE && op.mem.index != ZYDIS_REGISTER_NONE && op.mem.scale == 8 &&
            op.mem.disp.size && img.mapped((uint64_t)op.mem.disp.value, 8)) {
            table = (uint64_t)op.mem.disp.value;
            relative = false;
        }
    };
    consider_mem(jmp_ops[0]);
    for (size_t i = recent.size(); i-- > 0 && !table;) {
        ZydisDecodedInstruction insn;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        if (!dec.decode(img, recent[i], insn, ops)) break;
        if (insn.mnemonic == ZYDIS_MNEMONIC_LEA && ops[1].mem.base == ZYDIS_REGISTER_RIP && ops[1].mem.disp.size) {
            const uint64_t t = recent[i] + insn.length + (uint64_t)ops[1].mem.disp.value;
            if (img.mapped(t, 4) && !img.is_code(t)) { table = t; relative = true; }
            else if (img.mapped(t, 4)) { table = t; relative = true; } // tables inside .text happen too
        } else if (insn.mnemonic == ZYDIS_MNEMONIC_MOV && insn.operand_count_visible == 2) {
            consider_mem(ops[1]);
        } else if (insn.mnemonic == ZYDIS_MNEMONIC_MOVSXD && ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY && ops[1].mem.scale == 4 &&
                   ops[1].mem.base != ZYDIS_REGISTER_NONE && ops[1].mem.base != ZYDIS_REGISTER_RIP) {
            table_base = ops[1].mem.base; // movslq (T,i,4) with T in a register loaded earlier
        }
    }
    if (!table && table_base != ZYDIS_REGISTER_NONE) {
        // The `lea T(%rip)` was hoisted out of the loop: the last one seen into that register.
        auto it = lea_by_reg.find(table_base);
        if (it != lea_by_reg.end()) { table = it->second; relative = true; }
    }
    if (!table) return out;
    for (size_t i = recent.size(); i-- > 0 && !count;) {
        ZydisDecodedInstruction insn;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        if (!dec.decode(img, recent[i], insn, ops)) break;
        if (insn.mnemonic == ZYDIS_MNEMONIC_CMP && insn.operand_count_visible == 2 && ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            count = ops[1].imm.value.u + 1;
        }
    }
    const uint64_t limit = count ? std::min<uint64_t>(count, 65536) : 4096;
    for (uint64_t i = 0; i < limit; ++i) {
        uint64_t target;
        if (relative) {
            if (!img.mapped(table + i * 4, 4)) break;
            int32_t off;
            std::memcpy(&off, img.at(table + i * 4), 4);
            target = table + (uint64_t)(int64_t)off;
        } else {
            if (!img.mapped(table + i * 8, 8)) break;
            target = img.rd64(table + i * 8);
        }
        ZydisDecodedInstruction insn;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        // Plausible: code, within 1 MiB of the jump, and decodable.
        if (!img.is_code(target) || (target > jmp_addr ? target - jmp_addr : jmp_addr - target) > (1u << 20) ||
            !dec.decode(img, target, insn, ops)) {
            if (count) continue; // a bounded table may hold a few non-code entries (default cases)
            break;
        }
        out.push_back(target);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

Function explore(const Image& img, const Decoder& dec, uint64_t entry, std::vector<uint64_t>& callees, const Options& opt,
                 const std::set<uint64_t>& boundaries) {
    Function f;
    std::set<uint64_t> tails; // boundaries reached by falling through
    f.entry = entry;
    f.blocks.insert(entry);
    std::vector<uint64_t> work{entry};
    std::set<uint64_t> seen;
    // The instructions before a block that falls through into it: jump-table shapes span the
    // conditional branch that bounds the index.
    std::map<uint64_t, std::vector<uint64_t>> recent_at;
    std::map<ZydisRegister, uint64_t> lea_by_reg; // register -> address of the last rip-relative lea into it
    while (!work.empty()) {
        uint64_t a = work.back();
        work.pop_back();
        if (!seen.insert(a).second) continue;
        ZydisDecodedInstruction insn;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        std::vector<uint64_t> recent = recent_at.count(a) ? recent_at[a] : std::vector<uint64_t>{};
        for (;;) {
            if (!dec.decode(img, a, insn, ops)) break;
            const uint64_t next = a + insn.length;
            f.end = std::max(f.end, next);
            if (insn.mnemonic == ZYDIS_MNEMONIC_LEA && ops[1].mem.base == ZYDIS_REGISTER_RIP && ops[1].mem.disp.size &&
                ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                lea_by_reg[ops[0].reg.value] = next + (uint64_t)ops[1].mem.disp.value;
            }
            if (insn.mnemonic == ZYDIS_MNEMONIC_JMP && ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                auto targets = read_jump_table(img, dec, recent, a, insn, ops, lea_by_reg);
                if (!targets.empty()) {
                    for (uint64_t t : targets) { f.blocks.insert(t); work.push_back(t); }
                    f.jump_tables[a] = std::move(targets);
                }
            }
            recent.push_back(a);
            if (recent.size() > 12) recent.erase(recent.begin());
            if (insn.mnemonic == ZYDIS_MNEMONIC_CALL) {
                if (uint64_t t = branch_target(insn, ops, a); t && img.is_code(t)) callees.push_back(t);
                // A context switch (fibers, coroutines, longjmp to another stack) can come back to
                // this return address with none of the host frames that were active here.
                if (opt.resume_points && img.is_code(next)) f.resume_points.insert(next);
            } else if (insn.mnemonic == ZYDIS_MNEMONIC_LEA && ops[1].mem.base == ZYDIS_REGISTER_RIP && ops[1].mem.disp.size) {
                // A function pointer taken in position-independent code.
                const uint64_t t = next + (uint64_t)ops[1].mem.disp.value;
                if (img.is_code(t) && t != a) callees.push_back(t);
            } else if (insn.mnemonic == ZYDIS_MNEMONIC_MOV && ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE && ops[0].size >= 32) {
                // A function pointer as an absolute immediate (non-PIE code).
                const uint64_t t = ops[1].imm.is_signed ? (uint64_t)ops[1].imm.value.s : ops[1].imm.value.u;
                if (img.is_code(t) && t != a) callees.push_back(t);
            } else if (insn.mnemonic == ZYDIS_MNEMONIC_JMP || is_jcc(insn.mnemonic) ||
                       insn.mnemonic == ZYDIS_MNEMONIC_LOOP || insn.mnemonic == ZYDIS_MNEMONIC_LOOPE ||
                       insn.mnemonic == ZYDIS_MNEMONIC_LOOPNE) {
                if (uint64_t t = branch_target(insn, ops, a); t && img.is_code(t) && !opt.natives.count(t)) {
                    // A `jmp` out of the function's own neighbourhood (a tail call) still becomes a
                    // block here: the translation is per reachable code, not per symbol.
                    f.blocks.insert(t);
                    work.push_back(t);
                }
            }
            if (ends_block(insn)) {
                if (is_jcc(insn.mnemonic) || insn.mnemonic == ZYDIS_MNEMONIC_LOOP ||
                    insn.mnemonic == ZYDIS_MNEMONIC_LOOPE || insn.mnemonic == ZYDIS_MNEMONIC_LOOPNE) {
                    f.blocks.insert(next);
                    recent_at[next] = recent;
                    work.push_back(next);
                }
                break;
            }
            a = next;
            if (f.blocks.count(a)) { recent_at[a] = recent; work.push_back(a); break; }
            if (seen.count(a)) break;
            // Falling into the start of another function (cold landing pads one after another,
            // each ending in a call to _Unwind_Resume that the decoder does not know never
            // returns): each pad would otherwise copy every pad after it, quadratically.
            if (a != entry && boundaries.count(a)) { tails.insert(a); break; }
        }
    }
    // A boundary that a real jump also reached is an ordinary block (already explored).
    for (uint64_t t : tails) {
        if (f.blocks.count(t)) continue;
        f.blocks.insert(t);
        f.tail_blocks.insert(t);
    }
    return f;
}

} // namespace

std::map<uint64_t, Function> discover(const Image& img, const std::vector<uint64_t>& roots, Stats& stats,
                                      const Options& opt) {
    Decoder dec;
    std::map<uint64_t, Function> out;
    std::set<uint64_t> boundaries(img.eh_frame_starts.begin(), img.eh_frame_starts.end());
    boundaries.insert(img.landing_pads.begin(), img.landing_pads.end());
    if (!opt.boundaries) boundaries.clear();
    std::vector<uint64_t> work(roots);
    for (uint64_t a : img.code_pointers) work.push_back(a);
    for (uint64_t a : img.eh_frame_starts) work.push_back(a);
    if (opt.scan_data) {
        // Every aligned qword in non-executable memory that points at decodable code: function
        // pointer tables of a non-relocatable image (a relocatable one lists them as relocations).
        for (const auto& r : img.loaded) {
            for (uint64_t a = (r.start + 7) & ~UINT64_C(7); a + 8 <= r.end; a += 8) {
                if (img.is_code(a)) continue;
                const uint64_t v = img.rd64(a);
                ZydisDecodedInstruction insn;
                ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
                if (v && img.is_code(v) && dec.decode(img, v, insn, ops) && !opt.natives.count(v)) work.push_back(v);
            }
        }
    }
    auto drain = [&]() {
        while (!work.empty() && out.size() < opt.max_functions) {
            const uint64_t e = work.back();
            work.pop_back();
            if (out.count(e) || !img.is_code(e) || opt.natives.count(e)) continue;
            std::vector<uint64_t> callees;
            Function f = explore(img, dec, e, callees, opt, boundaries);
            out.emplace(e, std::move(f));
            for (uint64_t c : callees) if (!out.count(c)) work.push_back(c);
        }
    };
    drain();
    // Landing pads: entries into the middle of the function that contains them.
    for (uint64_t pad : img.landing_pads) {
        auto it = out.upper_bound(pad);
        if (out.count(pad)) continue;
        Function* owner = nullptr;
        if (it != out.begin()) {
            --it;
            if (pad > it->second.entry && pad < it->second.end) owner = &it->second;
        }
        if (!owner) {
            // Past the end of what the normal flow reached (code after the last ret, or in a
            // cold section): a function of its own. Entering it with the frame the unwinder
            // restored is the same as entering the middle of the original function.
            work.push_back(pad);
            drain(); // the pad and everything it calls
            continue;
        }
        Function& f = *owner;
        // A pad the function only falls into is a tail block (a dispatch to the pad's address,
        // which would resolve back to this very entry): its code must be in the function.
        if (!f.blocks.count(pad) || f.tail_blocks.count(pad)) {
            // Not reached by the normal flow: explore from it so that its code exists.
            std::vector<uint64_t> callees;
            Function extra = explore(img, dec, pad, callees, opt, boundaries);
            for (uint64_t c : callees) if (!out.count(c)) work.push_back(c);
            f.tail_blocks.erase(pad);
            for (uint64_t t : extra.tail_blocks)
                if (!f.blocks.count(t) || f.tail_blocks.count(t)) f.tail_blocks.insert(t);
            for (uint64_t b : extra.blocks)
                if (!extra.tail_blocks.count(b)) f.tail_blocks.erase(b);
            f.blocks.insert(extra.blocks.begin(), extra.blocks.end());
            f.resume_points.insert(extra.resume_points.begin(), extra.resume_points.end());
            f.end = std::max(f.end, extra.end);
            for (auto& [k, v] : extra.jump_tables) f.jump_tables[k] = v;
        }
        f.extra_entries.insert(pad);
    }
    drain(); // what the landing pads' code calls
    stats.functions = out.size();
    stats.landing_pads = img.landing_pads.size();
    return out;
}

// ---- emission --------------------------------------------------------------------------------

namespace {

struct Emitter {
    const Image& img;
    const Options& opt;
    Stats& stats;
    FILE* out;
    Decoder dec;
    const std::map<uint64_t, Function>* functions = nullptr;
    const Function* current = nullptr;
    std::vector<uint64_t> block_recent; // addresses of the last instructions of the current block
    int vex_mask = -1;     // VEX blendv: the explicit mask register (legacy forms use xmm0)
    bool vex_src2 = false; // VEX: the second source was copied to `vsrc2` before the destination changed
    bool hi = false;       // VEX.256 split into lanes: registers name the upper halves, memory is +16
    bool lock_rmw = false; // inside a LOCK compare-and-swap loop: the memory operand is vp_old / vp_new
    // Per instruction:
    uint64_t rip = 0, next = 0;
    const ZydisDecodedInstruction* insn = nullptr;
    const ZydisDecodedOperand* ops = nullptr;
    std::string body; // statements of the current instruction

    Emitter(const Image& i, const Options& o, Stats& s, FILE* f) : img(i), opt(o), stats(s), out(f) {}

    void line(const std::string& s) { body += "    " + s + "\n"; }

    // -- registers ------------------------------------------------------------------------------
    static bool is_high8(ZydisRegister r) { return r >= ZYDIS_REGISTER_AH && r <= ZYDIS_REGISTER_BH; }
    static int gpr_index(ZydisRegister r) {
        if (r >= ZYDIS_REGISTER_AL && r <= ZYDIS_REGISTER_BL) return r - ZYDIS_REGISTER_AL;
        if (is_high8(r)) return r - ZYDIS_REGISTER_AH;
        if (r >= ZYDIS_REGISTER_SPL && r <= ZYDIS_REGISTER_R15B) return 4 + (r - ZYDIS_REGISTER_SPL);
        if (r >= ZYDIS_REGISTER_AX && r <= ZYDIS_REGISTER_R15W) return r - ZYDIS_REGISTER_AX;
        if (r >= ZYDIS_REGISTER_EAX && r <= ZYDIS_REGISTER_R15D) return r - ZYDIS_REGISTER_EAX;
        if (r >= ZYDIS_REGISTER_RAX && r <= ZYDIS_REGISTER_R15) return r - ZYDIS_REGISTER_RAX;
        return -1;
    }
    static int xmm_index(ZydisRegister r) {
        if (r >= ZYDIS_REGISTER_XMM0 && r <= ZYDIS_REGISTER_XMM15) return r - ZYDIS_REGISTER_XMM0;
        if (r >= ZYDIS_REGISTER_YMM0 && r <= ZYDIS_REGISTER_YMM15) return r - ZYDIS_REGISTER_YMM0;
        return -1;
    }
    // The 128-bit half `h` (0 low, 1 high) of vector register `x`.
    static std::string X(int x, bool h) { return fmt(h ? "cpu->ymmh[%d]" : "cpu->xmm[%d]", x); }
    static const char* ctype(int bits) {
        switch (bits) { case 8: return "uint8_t"; case 16: return "uint16_t"; case 32: return "uint32_t"; default: return "uint64_t"; }
    }

    // An address inside the image: absolute, or relative to the run-time base in --pic output.
    std::string A(uint64_t addr) const {
        if (!opt.pic) return hex(addr);
        return fmt("(VP_MOD.base + 0x%" PRIx64 "ull)", addr - img.base);
    }
    // Does the instruction field at `offset` bytes into the current instruction hold a relocated
    // 8-byte value (an absolute pointer the loader rebases)?
    bool relocated_field(unsigned offset) const {
        return opt.pic && offset && img.is_reloc_site(rip + offset);
    }

    std::string reg_rd(ZydisRegister r, int bits) {
        if (r == ZYDIS_REGISTER_RIP) return A(next);
        if (int x = xmm_index(r); x >= 0) return X(x, hi);
        const int i = gpr_index(r);
        if (i < 0) throw std::runtime_error(fmt("register %s", ZydisRegisterGetString(r)));
        if (bits == 8 && is_high8(r)) return fmt("VP_R8H(%d)", i);
        return fmt("VP_R%d(%d)", bits, i);
    }
    std::string reg_wr(ZydisRegister r, int bits, const std::string& v) {
        if (int x = xmm_index(r); x >= 0) return X(x, hi) + " = " + v + ";";
        const int i = gpr_index(r);
        if (i < 0) throw std::runtime_error(fmt("register %s", ZydisRegisterGetString(r)));
        if (bits == 8 && is_high8(r)) return fmt("VP_W8H(%d, (uint8_t)(%s));", i, v.c_str());
        return fmt("VP_W%d(%d, (%s)(%s));", bits, i, ctype(bits), v.c_str());
    }

    // -- memory -----------------------------------------------------------------------------------
    std::string ea(const ZydisDecodedOperand& op) {
        const auto& m = op.mem;
        std::string e;
        if (m.base == ZYDIS_REGISTER_RIP) {
            e = A(next);
        } else if (m.base != ZYDIS_REGISTER_NONE) {
            e = reg_rd(m.base, 64);
        }
        if (m.index != ZYDIS_REGISTER_NONE && m.scale) {
            const std::string ix = reg_rd(m.index, 64) + (m.scale > 1 ? fmt(" * %u", m.scale) : "");
            e = e.empty() ? ix : e + " + " + ix;
        }
        if (m.disp.size && m.disp.value) {
            if (m.base != ZYDIS_REGISTER_RIP && relocated_field(insn->raw.disp.offset)) {
                const std::string d = A((uint64_t)m.disp.value);
                e = e.empty() ? d : e + " + " + d;
            } else {
                e = e.empty() ? fmt("(uint64_t)%" PRId64, m.disp.value)
                              : e + fmt(" + (uint64_t)%" PRId64 "ll", (long long)m.disp.value);
            }
        }
        if (m.segment == ZYDIS_REGISTER_FS) e = "cpu->fs_base" + (e.empty() ? "" : " + " + e);
        else if (m.segment == ZYDIS_REGISTER_GS) e = "cpu->gs_base" + (e.empty() ? "" : " + " + e);
        if (e.empty()) e = "0";
        if (insn->address_width == 32) e = "(uint64_t)(uint32_t)(" + e + ")";
        if (hi) e += " + 16";
        return "(" + e + ")";
    }

    // Reads operand `op` as an unsigned integer of `bits`; `addr` is a precomputed EA variable.
    std::string rd(const ZydisDecodedOperand& op, int bits, const std::string& addr = "") {
        switch (op.type) {
        case ZYDIS_OPERAND_TYPE_REGISTER: return reg_rd(op.reg.value, bits);
        case ZYDIS_OPERAND_TYPE_MEMORY:
            if (lock_rmw && addr == "ea") return "(uint64_t)vp_old";
            return fmt("vp_ld%d(%s)", bits, addr.empty() ? ea(op).c_str() : addr.c_str());
        case ZYDIS_OPERAND_TYPE_IMMEDIATE:
            if (bits == 64 && relocated_field(insn->raw.imm[0].offset)) return A(op.imm.value.u); // movabs of an image address
            if (op.imm.is_signed) return fmt("((%s)(int64_t)%" PRId64 "ll)", ctype(bits), (long long)op.imm.value.s);
            return fmt("((%s)%" PRIu64 "ull)", ctype(bits), (unsigned long long)op.imm.value.u);
        default: throw std::runtime_error("operand type");
        }
    }
    std::string wr(const ZydisDecodedOperand& op, int bits, const std::string& v, const std::string& addr = "") {
        switch (op.type) {
        case ZYDIS_OPERAND_TYPE_REGISTER: return reg_wr(op.reg.value, bits, v);
        case ZYDIS_OPERAND_TYPE_MEMORY:
            if (lock_rmw && addr == "ea") return fmt("vp_new = (%s)(%s);", ctype(bits), v.c_str());
            return fmt("vp_st%d(%s, (%s)(%s));", bits, addr.empty() ? ea(op).c_str() : addr.c_str(), ctype(bits), v.c_str());
        default: throw std::runtime_error("write to immediate");
        }
    }

    // For read-modify-write operands the address is computed once.
    std::string bind_addr(const ZydisDecodedOperand& op) {
        if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) return "";
        if (lock_rmw) return "ea"; // bound before the loop
        line("const uint64_t ea = " + ea(op) + ";");
        return "ea";
    }

    std::string fn_name(uint64_t a) const { return opt.symbol_prefix + fmt("%" PRIx64, a); }
    std::string label(uint64_t a) const { return fmt("L_%" PRIx64, a); }

    std::string format_insn(const ZydisDecodedInstruction& i, const ZydisDecodedOperand* o, uint64_t at) {
        char text[256];
        ZydisFormatter fm;
        ZydisFormatterInit(&fm, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&fm, &i, o, i.operand_count_visible, text, sizeof text, at, nullptr);
        return text;
    }

    void unsupported(const char* what) {
        stats.unsupported++;
        stats.unsupported_by_mnemonic[what]++;
        if (stats.unsupported_sites.size() < 400 && insn) stats.unsupported_sites.emplace_back(rip, format_insn(*insn, ops, rip));
        line(fmt("vp_unsupported(cpu, %s, \"%s\"); return;", A(rip).c_str(), what));
    }

    // -- integer ALU ------------------------------------------------------------------------------
    void alu2(const char* flags, const char* cop, bool store) {
        const int bits = ops[0].size;
        if (store && !lock_rmw && (insn->attributes & ZYDIS_ATTRIB_HAS_LOCK) && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            // lock add/sub (and/or/xor go through the compare-and-swap loop): one atomic
            // read-modify-write; the flags come from the values seen.
            const bool sub = !strcmp(cop, "-");
            line("const uint64_t ea = " + ea(ops[0]) + ";");
            line(fmt("const uint64_t y = %s;", rd(ops[1], bits).c_str()));
            line(fmt("const uint64_t x = vp_fetch_add%d(ea, (uint%d_t)%s);", bits, bits, sub ? "(0 - y)" : "y"));
            line(fmt("const uint64_t r = (x %s y) & VP_MASK(%d);", cop, bits));
            if (flags[0] == 'l') line(fmt("vp_flags_logic(VP_FC, %d, r);", bits));
            else line(fmt("vp_flags_%s(VP_FC, %d, x, y, r);", flags, bits));
            return;
        }
        const std::string a = bind_addr(ops[0]);
        line(fmt("const uint64_t x = %s, y = %s;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str()));
        line(fmt("const uint64_t r = (x %s y) & VP_MASK(%d);", cop, bits));
        if (flags[0] == 'l') line(fmt("vp_flags_logic(VP_FC, %d, r);", bits));
        else line(fmt("vp_flags_%s(VP_FC, %d, x, y, r);", flags, bits));
        if (store) line(wr(ops[0], bits, "r", a));
    }

    void shift(const char* kind) {
        const int bits = ops[0].size;
        const std::string a = bind_addr(ops[0]);
        const std::string count = insn->operand_count_visible > 1 ? rd(ops[1], 8) : std::string("1");
        line(fmt("const uint64_t x = %s;", rd(ops[0], bits, a).c_str()));
        line(fmt("const unsigned n = (unsigned)(%s) & %u;", count.c_str(), bits == 64 ? 63u : 31u));
        line("if (n) {");
        std::string r;
        if (!strcmp(kind, "shl")) r = fmt("(x << n) & VP_MASK(%d)", bits);
        else if (!strcmp(kind, "shr")) r = "x >> n";
        else r = fmt("vp_sar(%d, x, n)", bits);
        line(fmt("    const uint64_t r = %s;", r.c_str()));
        line(fmt("    vp_flags_%s(VP_FC, %d, x, n, r);", kind, bits));
        line("    " + wr(ops[0], bits, "r", a));
        line("}");
        if (bits == 32) line("else { " + wr(ops[0], 32, "x", a) + " }"); // count 0 still zero-extends
    }

    void rotate(bool left) {
        const int bits = ops[0].size;
        const std::string a = bind_addr(ops[0]);
        const std::string count = insn->operand_count_visible > 1 ? rd(ops[1], 8) : std::string("1");
        line(fmt("const uint64_t x = %s;", rd(ops[0], bits, a).c_str()));
        line(fmt("const unsigned n = (unsigned)(%s) & %u;", count.c_str(), bits == 64 ? 63u : 31u));
        line("if (n) {");
        line(fmt("    const uint64_t r = vp_%s(%d, x, n);", left ? "rol" : "ror", bits));
        if (left) line("    VP_FC->cf = (uint8_t)(r & 1);"), line(fmt("    VP_FC->of = (uint8_t)(VP_FC->cf ^ VP_SIGN(%d, r));", bits));
        else line(fmt("    VP_FC->cf = VP_SIGN(%d, r);", bits)), line(fmt("    VP_FC->of = (uint8_t)(VP_SIGN(%d, r) ^ ((r >> (%d - 2)) & 1));", bits, bits));
        line("    " + wr(ops[0], bits, "r", a));
        line("}");
        if (bits == 32) line("else { " + wr(ops[0], 32, "x", a) + " }");
    }

    // -- SSE helpers ------------------------------------------------------------------------------
    // Reads a packed operand (register or 16-byte memory) as a VpXmm value.
    std::string xmm_rd(const ZydisDecodedOperand& op) {
        if (vex_src2 && &op == &ops[1]) return "vsrc2";
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) return reg_rd(op.reg.value, 128);
        return "vp_ld128(" + ea(op) + ")";
    }
    // Scalar float/double read of an XMM or memory operand.
    std::string f32_rd(const ZydisDecodedOperand& op) {
        if (vex_src2 && &op == &ops[1]) return "vsrc2.f32[0]";
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) return reg_rd(op.reg.value, 128) + ".f32[0]";
        return "vp_bits_f32(vp_ld32(" + ea(op) + "))";
    }
    std::string f64_rd(const ZydisDecodedOperand& op) {
        if (vex_src2 && &op == &ops[1]) return "vsrc2.f64[0]";
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) return reg_rd(op.reg.value, 128) + ".f64[0]";
        return "vp_bits_f64(vp_ld64(" + ea(op) + "))";
    }
    std::string xmm_dst() { return reg_rd(ops[0].reg.value, 128); }

    void sse_scalar(const char* cop, bool dbl) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) { unsupported("sse-dst"); return; }
        const std::string src = dbl ? f64_rd(ops[1]) : f32_rd(ops[1]);
        const char* fld = dbl ? "f64[0]" : "f32[0]";
        line(fmt("%s.%s = %s.%s %s %s;", xmm_dst().c_str(), fld, xmm_dst().c_str(), fld, cop, src.c_str()));
    }
    void sse_packed(const char* cop, bool dbl) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) { unsupported("sse-dst"); return; }
        line("const VpXmm s = " + xmm_rd(ops[1]) + ";");
        const int n = dbl ? 2 : 4;
        for (int i = 0; i < n; ++i) {
            line(fmt("%s.%s[%d] = %s.%s[%d] %s s.%s[%d];", xmm_dst().c_str(), dbl ? "f64" : "f32", i,
                     xmm_dst().c_str(), dbl ? "f64" : "f32", i, cop, dbl ? "f64" : "f32", i));
        }
    }
    void sse_bitwise(const char* cop, bool invert_first) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) { unsupported("sse-dst"); return; }
        line("const VpXmm s = " + xmm_rd(ops[1]) + ";");
        for (int i = 0; i < 2; ++i) {
            line(fmt("%s.u64[%d] = %s%s.u64[%d]%s %s s.u64[%d];", xmm_dst().c_str(), i, invert_first ? "(~" : "",
                     xmm_dst().c_str(), i, invert_first ? ")" : "", cop, i));
        }
    }
    void sse_minmax(bool is_max, bool dbl) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) { unsupported("sse-dst"); return; }
        const char* fld = dbl ? "f64[0]" : "f32[0]";
        line(fmt("%s.%s = vp_%s%s(%s.%s, %s);", xmm_dst().c_str(), fld, is_max ? "max" : "min", dbl ? "sd" : "ss",
                 xmm_dst().c_str(), fld, (dbl ? f64_rd(ops[1]) : f32_rd(ops[1])).c_str()));
    }

    // movs/stos of `w` bits, with or without rep.
    // The string instructions' registers: rsi/rdi/rcx, or esi/edi/ecx with an address-size
    // prefix (zero-extended when written back); the source may have an fs/gs override.
    std::string sreg(int r) const { return insn->address_width == 32 ? fmt("(uint64_t)VP_R32(%d)", r) : fmt("VP_R64(%d)", r); }
    std::string sset(int r, const std::string& v) const { return insn->address_width == 32 ? fmt("VP_W32(%d, (uint32_t)(%s));", r, v.c_str()) : fmt("VP_W64(%d, %s);", r, v.c_str()); }
    std::string ssrc() const {
        const std::string si = sreg(VP_RSI_INDEX);
        if (insn->attributes & ZYDIS_ATTRIB_HAS_SEGMENT_FS) return "(cpu->fs_base + " + si + ")";
        if (insn->attributes & ZYDIS_ATTRIB_HAS_SEGMENT_GS) return "(cpu->gs_base + " + si + ")";
        return si;
    }
    static constexpr int VP_RCX_INDEX = 1, VP_RSI_INDEX = 6, VP_RDI_INDEX = 7;

    void string_op(int w, bool movs) {
        const bool rep = insn->attributes & ZYDIS_ATTRIB_HAS_REP;
        line(fmt("const int64_t step = VP_FC->df ? -%d : %d;", w / 8, w / 8));
        if (rep) line("while (" + sreg(VP_RCX_INDEX) + ") {");
        if (movs) line(fmt("    vp_st%d(%s, vp_ld%d(%s)); ", w, sreg(VP_RDI_INDEX).c_str(), w, ssrc().c_str()) + sset(VP_RSI_INDEX, sreg(VP_RSI_INDEX) + " + step") + " " +
                       sset(VP_RDI_INDEX, sreg(VP_RDI_INDEX) + " + step"));
        else line(fmt("    vp_st%d(%s, VP_R%d(VP_RAX)); ", w, sreg(VP_RDI_INDEX).c_str(), w) + sset(VP_RDI_INDEX, sreg(VP_RDI_INDEX) + " + step"));
        if (rep) line("    " + sset(VP_RCX_INDEX, sreg(VP_RCX_INDEX) + " - 1") + " }");
    }

    // lods / scas / cmps (kind 0, 1, 2), with rep / repe / repne. The repeat test uses the
    // comparison's own zero result, never the (lazily computed) flag.
    void string_scan(int w, int kind) {
        const bool rep = insn->attributes & (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE | ZYDIS_ATTRIB_HAS_REPNE);
        const bool repe = insn->attributes & ZYDIS_ATTRIB_HAS_REPE, repne = insn->attributes & ZYDIS_ATTRIB_HAS_REPNE;
        line(fmt("const int64_t step = VP_FC->df ? -%d : %d;", w / 8, w / 8));
        if (rep) line("while (" + sreg(VP_RCX_INDEX) + ") {");
        if (kind == 0) {
            line(fmt("    VP_W%d(VP_RAX, vp_ld%d(%s)); ", w, w, ssrc().c_str()) + sset(VP_RSI_INDEX, sreg(VP_RSI_INDEX) + " + step"));
        } else {
            if (kind == 1) line(fmt("    const uint64_t x = VP_R%d(VP_RAX), y = vp_ld%d(%s);", w, w, sreg(VP_RDI_INDEX).c_str()));
            else line(fmt("    const uint64_t x = vp_ld%d(%s), y = vp_ld%d(%s);", w, ssrc().c_str(), w, sreg(VP_RDI_INDEX).c_str()));
            line(fmt("    const uint64_t r = (x - y) & VP_MASK(%d);", w));
            line(fmt("    vp_flags_sub(VP_FC, %d, x, y, r);", w));
            if (kind == 2) line("    " + sset(VP_RSI_INDEX, sreg(VP_RSI_INDEX) + " + step"));
            line("    " + sset(VP_RDI_INDEX, sreg(VP_RDI_INDEX) + " + step"));
        }
        if (rep) {
            line("    " + sset(VP_RCX_INDEX, sreg(VP_RCX_INDEX) + " - 1"));
            if (kind && repe) line("    if (r != 0) break;");
            if (kind && repne) line("    if (r == 0) break;");
            line("}");
        }
    }

    // VEX-encoded 128-bit instruction -> the legacy mnemonic with the same semantics, or NONE.
    static ZydisMnemonic legacy_of(ZydisMnemonic m) {
        switch (m) {
#define V(x) case ZYDIS_MNEMONIC_V##x: return ZYDIS_MNEMONIC_##x;
        V(MOVAPS) V(MOVUPS) V(MOVAPD) V(MOVUPD) V(MOVDQA) V(MOVDQU) V(MOVSS) V(MOVSD) V(MOVD) V(MOVQ)
        V(MOVLPS) V(MOVLPD) V(MOVHPS) V(MOVHPD) V(MOVLHPS) V(MOVHLPS)
        V(ADDSS) V(SUBSS) V(MULSS) V(DIVSS) V(ADDSD) V(SUBSD) V(MULSD) V(DIVSD)
        V(ADDPS) V(SUBPS) V(MULPS) V(DIVPS) V(ADDPD) V(SUBPD) V(MULPD) V(DIVPD)
        V(MINSS) V(MAXSS) V(MINSD) V(MAXSD) V(SQRTSS) V(SQRTSD) V(SQRTPS)
        V(XORPS) V(XORPD) V(PXOR) V(ANDPS) V(ANDPD) V(PAND) V(ORPS) V(ORPD) V(POR) V(ANDNPS) V(ANDNPD) V(PANDN)
        V(UCOMISS) V(COMISS) V(UCOMISD) V(COMISD)
        V(CVTSI2SS) V(CVTSI2SD) V(CVTTSS2SI) V(CVTTSD2SI) V(CVTSS2SI) V(CVTSD2SI) V(CVTSS2SD) V(CVTSD2SS)
        V(CVTDQ2PS) V(CVTTPS2DQ) V(SHUFPS) V(PSHUFD) V(UNPCKLPS) V(UNPCKHPS) V(UNPCKLPD) V(UNPCKHPD)
        V(PUNPCKLQDQ) V(PUNPCKHQDQ) V(PADDD) V(PSUBD) V(PADDQ) V(PSUBQ) V(PCMPEQD) V(PCMPEQB)
        V(PMOVMSKB) V(MOVMSKPS) V(CMPSS) V(CMPSD) V(PSLLDQ) V(PSRLDQ)
        V(BLENDVPS) V(BLENDVPD) V(PBLENDVB) V(BLENDPS) V(BLENDPD) V(PINSRB) V(PINSRW) V(PINSRD) V(PINSRQ)
        V(PEXTRB) V(PEXTRW) V(PEXTRD) V(PEXTRQ) V(PSHUFB) V(PMULLD) V(PMULUDQ) V(ROUNDSS) V(ROUNDSD) V(ROUNDPS)
        V(PMINSD) V(PMAXSD) V(PMINUD) V(PMAXUD) V(PMINSW) V(PMAXSW) V(PMINUB) V(PMAXUB) V(PCMPGTD) V(PCMPGTB) V(PCMPGTW)
        V(PCMPEQW) V(PCMPEQQ) V(PSLLD) V(PSRLD) V(PSRAD) V(PSLLQ) V(PSRLQ) V(PSLLW) V(PSRLW) V(PSRAW)
        V(PTEST) V(PALIGNR) V(PUNPCKLBW) V(PUNPCKHBW) V(PUNPCKLWD) V(PUNPCKHWD) V(PUNPCKLDQ) V(PUNPCKHDQ)
        V(PADDB) V(PADDW) V(PSUBB) V(PSUBW) V(PMULLW) V(PMULHW) V(PMULHUW) V(PMADDWD) V(PACKSSDW) V(PACKUSWB) V(PACKSSWB) V(PACKUSDW)
        V(PMOVZXBW) V(PMOVZXBD) V(PMOVZXWD) V(PMOVZXDQ) V(PMOVSXBW) V(PMOVSXBD) V(PMOVSXWD) V(PMOVSXDQ)
        V(MINPS) V(MAXPS) V(MINPD) V(MAXPD) V(RSQRTSS) V(RCPSS) V(RSQRTPS) V(RCPPS) V(SQRTPD) V(HADDPS) V(CVTPS2PD) V(CVTPD2PS)
        V(CVTPS2DQ) V(CVTTPD2DQ) V(CVTDQ2PD) V(SHUFPD) V(PSHUFLW) V(PSHUFHW) V(MOVSHDUP) V(MOVSLDUP) V(MOVDDUP) V(INSERTPS) V(EXTRACTPS) V(DPPS)
        V(LDMXCSR) V(STMXCSR) V(MOVNTPS) V(MOVNTPD) V(MOVNTDQ) V(LDDQU) V(HADDPD) V(HSUBPS) V(HSUBPD) V(ADDSUBPS) V(ADDSUBPD) V(ROUNDPD)
        V(PMULHRSW) V(PMADDUBSW) V(PHADDW) V(PHSUBW) V(PHSUBD) V(PHADDSW) V(PHSUBSW) V(PSIGNB) V(PSIGNW) V(PMULDQ) V(PCMPGTQ)
        V(PMINSB) V(PMAXSB) V(PMINUW) V(PMAXUW) V(PBLENDW) V(PHMINPOSUW) V(MPSADBW) V(PCMPESTRI) V(PCMPESTRM) V(PCMPISTRM)
        V(AESENC) V(AESENCLAST) V(AESDEC) V(AESDECLAST) V(AESIMC) V(AESKEYGENASSIST) V(PCLMULQDQ)
        V(CMPPS) V(CMPPD) V(PABSD) V(PABSW) V(PABSB) V(PSIGND) V(PHADDD) V(PAVGB) V(PAVGW) V(PSADBW) V(PCMPISTRI)
#undef V
        default: return ZYDIS_MNEMONIC_INVALID;
        }
    }

    // -- one instruction --------------------------------------------------------------------------
    // Returns false when the instruction ended the block with a transfer (nothing falls through).
    bool emit_insn() {
        if (insn->attributes & ZYDIS_ATTRIB_HAS_VEX) {
            if (emit_avx()) return true;
            if (legacy_of(insn->mnemonic) != ZYDIS_MNEMONIC_INVALID) return emit_vex();
        }
        return emit_legacy(insn->mnemonic, insn->operand_count_visible);
    }

    void count_insn() {
        stats.instructions++;
        stats.by_mnemonic[ZydisMnemonicGetString(insn->mnemonic)]++;
        if (opt.emit_rip_updates) line("cpu->rip = " + A(rip) + ";");
        if (opt.trace) line("vp_trace(cpu, " + A(rip) + ");");
    }
    bool is_vreg(const ZydisDecodedOperand& o) const { return o.type == ZYDIS_OPERAND_TYPE_REGISTER && xmm_index(o.reg.value) >= 0; }
    bool is_ymm(const ZydisDecodedOperand& o) const {
        return o.type == ZYDIS_OPERAND_TYPE_REGISTER && ZydisRegisterGetClass(o.reg.value) == ZYDIS_REGCLASS_YMM;
    }
    // Half `h` of a vector operand (register or memory) as a VpXmm expression.
    std::string half(const ZydisDecodedOperand& o, bool h) {
        if (o.type == ZYDIS_OPERAND_TYPE_REGISTER) return X(xmm_index(o.reg.value), h);
        return "vp_ld128(" + ea(o) + (h ? " + 16" : "") + ")";
    }
    std::string zero_upper(const ZydisDecodedOperand& o) { return X(xmm_index(o.reg.value), true) + " = (VpXmm){{0}};"; }
    // Writes the 256-bit value held in locals `lo`, `hi_` to a ymm register or memory operand.
    void put256(const ZydisDecodedOperand& o) {
        if (o.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            line(X(xmm_index(o.reg.value), false) + " = lo; " + X(xmm_index(o.reg.value), true) + " = hi_;");
        } else {
            line("{ const uint64_t a_ = " + ea(o) + "; vp_st128(a_, lo); vp_st128(a_ + 16, hi_); }");
        }
    }

    // AVX instructions with no SSE counterpart, or whose 256-bit form is not two independent
    // 128-bit lanes. Returns false for everything else (handled by emit_vex / emit_legacy).
    bool emit_avx() {
        using M = ZydisMnemonic;
        const M m = insn->mnemonic;
        const bool y = insn->avx.vector_length == 256;
        const int n = insn->operand_count_visible;
        auto imm = [&](int i) { return (unsigned)ops[i].imm.value.u; };
        switch (m) {
        case ZYDIS_MNEMONIC_VZEROUPPER:
            count_insn();
            line("memset(cpu->ymmh, 0, sizeof cpu->ymmh);");
            return true;
        case ZYDIS_MNEMONIC_VZEROALL:
            count_insn();
            line("memset(cpu->xmm, 0, sizeof cpu->xmm); memset(cpu->ymmh, 0, sizeof cpu->ymmh);");
            return true;
        case ZYDIS_MNEMONIC_VBROADCASTSS: case ZYDIS_MNEMONIC_VBROADCASTSD: case ZYDIS_MNEMONIC_VBROADCASTF128: {
            count_insn();
            if (m == ZYDIS_MNEMONIC_VBROADCASTF128) line("const VpXmm lo = " + half(ops[1], false) + ", hi_ = lo;");
            else {
                const bool sd = m == ZYDIS_MNEMONIC_VBROADCASTSD;
                const std::string v = ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY ? fmt("vp_ld%d(%s)", sd ? 64 : 32, ea(ops[1]).c_str())
                                                                                : half(ops[1], false) + (sd ? ".u64[0]" : ".u32[0]");
                line(fmt("const uint%d_t v = %s; VpXmm lo; for (int i = 0; i < %d; ++i) lo.u%d[i] = v;", sd ? 64 : 32, v.c_str(), sd ? 2 : 4, sd ? 64 : 32));
                line("const VpXmm hi_ = lo;");
            }
            if (y) put256(ops[0]);
            else line(X(xmm_index(ops[0].reg.value), false) + " = lo; " + zero_upper(ops[0]));
            return true;
        }
        case ZYDIS_MNEMONIC_VINSERTF128: case ZYDIS_MNEMONIC_VINSERTI128:
            count_insn();
            line("VpXmm lo = " + half(ops[1], false) + ", hi_ = " + half(ops[1], true) + "; const VpXmm s = " + half(ops[2], false) + ";");
            line(imm(3) & 1 ? "hi_ = s;" : "lo = s;");
            put256(ops[0]);
            return true;
        case ZYDIS_MNEMONIC_VEXTRACTF128: case ZYDIS_MNEMONIC_VEXTRACTI128:
            count_insn();
            line("const VpXmm v = " + half(ops[1], imm(2) & 1) + ";");
            if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) line(X(xmm_index(ops[0].reg.value), false) + " = v; " + zero_upper(ops[0]));
            else line("vp_st128(" + ea(ops[0]) + ", v);");
            return true;
        case ZYDIS_MNEMONIC_VPERM2F128: case ZYDIS_MNEMONIC_VPERM2I128: {
            count_insn();
            const unsigned k = imm(3);
            line("const VpXmm src[4] = {" + half(ops[1], false) + ", " + half(ops[1], true) + ", " + half(ops[2], false) + ", " + half(ops[2], true) + "};");
            line(fmt("const VpXmm lo = %s, hi_ = %s;", (k & 8) ? "(VpXmm){{0}}" : fmt("src[%u]", k & 3).c_str(),
                     (k & 0x80) ? "(VpXmm){{0}}" : fmt("src[%u]", (k >> 4) & 3).c_str()));
            put256(ops[0]);
            return true;
        }
        case ZYDIS_MNEMONIC_VPERMILPS: case ZYDIS_MNEMONIC_VPERMILPD: {
            count_insn();
            const bool ps = m == ZYDIS_MNEMONIC_VPERMILPS;
            const bool by_imm = ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE;
            for (int h = 0; h < (y ? 2 : 1); ++h) {
                line(fmt("const VpXmm s%d = %s;", h, half(ops[1], h).c_str()));
                if (!by_imm) line(fmt("const VpXmm c%d = %s;", h, half(ops[2], h).c_str()));
            }
            for (int h = 0; h < (y ? 2 : 1); ++h) {
                line(fmt("VpXmm r%d;", h));
                for (int i = 0; i < (ps ? 4 : 2); ++i) {
                    std::string sel;
                    if (by_imm) sel = ps ? fmt("%u", (imm(2) >> (2 * i)) & 3) : fmt("%u", (imm(2) >> (2 * h + i)) & 1);
                    else sel = ps ? fmt("(c%d.u32[%d] & 3)", h, i) : fmt("((c%d.u64[%d] >> 1) & 1)", h, i);
                    line(fmt("r%d.%s[%d] = s%d.%s[%s];", h, ps ? "u32" : "u64", i, h, ps ? "u32" : "u64", sel.c_str()));
                }
            }
            line(X(xmm_index(ops[0].reg.value), false) + " = r0; " + (y ? X(xmm_index(ops[0].reg.value), true) + " = r1;" : zero_upper(ops[0])));
            return true;
        }
        case ZYDIS_MNEMONIC_VMASKMOVPS: case ZYDIS_MNEMONIC_VMASKMOVPD: {
            count_insn();
            const bool ps = m == ZYDIS_MNEMONIC_VMASKMOVPS;
            const int w = ps ? 32 : 64, per = ps ? 4 : 2;
            if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) { // store: masked-off elements are not touched
                line("const uint64_t a_ = " + ea(ops[0]) + ";");
                for (int h = 0; h < (y ? 2 : 1); ++h) {
                    line(fmt("{ const VpXmm k = %s, v = %s; for (int i = 0; i < %d; ++i) if (k.u%d[i] >> %d) vp_st%d(a_ + %d + i * %d, v.u%d[i]); }",
                             half(ops[1], h).c_str(), half(ops[2], h).c_str(), per, w, w - 1, w, h * 16, w / 8, w));
                }
            } else { // load: masked-off elements are zero and not read
                line("const uint64_t a_ = " + ea(ops[2]) + ";");
                for (int h = 0; h < (y ? 2 : 1); ++h) {
                    line(fmt("const VpXmm k%d = %s; VpXmm r%d = {{0}}; for (int i = 0; i < %d; ++i) if (k%d.u%d[i] >> %d) r%d.u%d[i] = vp_ld%d(a_ + %d + i * %d);",
                             h, half(ops[1], h).c_str(), h, per, h, w, w - 1, h, w, w, h * 16, w / 8));
                }
                line(X(xmm_index(ops[0].reg.value), false) + " = r0; " + (y ? X(xmm_index(ops[0].reg.value), true) + " = r1;" : zero_upper(ops[0])));
            }
            return true;
        }
        case ZYDIS_MNEMONIC_VTESTPS: case ZYDIS_MNEMONIC_VTESTPD: case ZYDIS_MNEMONIC_VPTEST: {
            if (m == ZYDIS_MNEMONIC_VPTEST && !y) return false; // the SSE4.1 form
            count_insn();
            const char* mask = m == ZYDIS_MNEMONIC_VTESTPS ? "0x8000000080000000ull" : m == ZYDIS_MNEMONIC_VTESTPD ? "0x8000000000000000ull" : "~0ull";
            line("uint64_t z = 0, c = 0;");
            for (int h = 0; h < (y ? 2 : 1); ++h) {
                line(fmt("{ const VpXmm a = %s, s = %s; for (int i = 0; i < 2; ++i) { z |= a.u64[i] & s.u64[i] & %s; c |= ~a.u64[i] & s.u64[i] & %s; } }",
                         half(ops[0], h).c_str(), half(ops[1], h).c_str(), mask, mask));
            }
            line("VP_FC->zf = z == 0; VP_FC->cf = c == 0; VP_FC->of = VP_FC->sf = VP_FC->pf = VP_FC->af = 0;");
            return true;
        }
        case ZYDIS_MNEMONIC_VCVTPH2PS:
            count_insn();
            if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) line(y ? "const VpXmm h = vp_ld128(" + ea(ops[1]) + ");" : "VpXmm h = {{0}}; h.u64[0] = vp_ld64(" + ea(ops[1]) + ");");
            else line("const VpXmm h = " + half(ops[1], false) + ";");
            line("VpXmm lo, hi_ = {{0}}; for (int i = 0; i < 4; ++i) lo.f32[i] = vp_f16_to_f32(h.u16[i]);");
            if (y) { line("for (int i = 0; i < 4; ++i) hi_.f32[i] = vp_f16_to_f32(h.u16[4 + i]);"); put256(ops[0]); }
            else line(X(xmm_index(ops[0].reg.value), false) + " = lo; " + zero_upper(ops[0]));
            return true;
        case ZYDIS_MNEMONIC_VCVTPS2PH: {
            count_insn();
            const unsigned k = imm(2);
            const std::string rc = (k & 4) ? "((cpu->mxcsr >> 13) & 3)" : fmt("%uu", k & 3);
            line("const VpXmm s0 = " + half(ops[1], false) + (y ? ", s1 = " + half(ops[1], true) : std::string()) + "; VpXmm r = {{0}};");
            line(fmt("for (int i = 0; i < 4; ++i) r.u16[i] = vp_f32_to_f16(s0.f32[i], %s, (cpu->mxcsr >> 6) & 1);", rc.c_str()));
            if (y) line(fmt("for (int i = 0; i < 4; ++i) r.u16[4 + i] = vp_f32_to_f16(s1.f32[i], %s, (cpu->mxcsr >> 6) & 1);", rc.c_str()));
            if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) line(y ? "vp_st128(" + ea(ops[0]) + ", r);" : "vp_st64(" + ea(ops[0]) + ", r.u64[0]);");
            else line(X(xmm_index(ops[0].reg.value), false) + " = r; " + zero_upper(ops[0]));
            return true;
        }
        default: break;
        }
        if (!y) return false;
        // 256-bit forms whose halves depend on each other.
        switch (m) {
        case ZYDIS_MNEMONIC_VMOVMSKPS: case ZYDIS_MNEMONIC_VMOVMSKPD: {
            count_insn();
            const bool ps = m == ZYDIS_MNEMONIC_VMOVMSKPS;
            line("const VpXmm a = " + half(ops[1], false) + ", b = " + half(ops[1], true) + "; uint32_t mk = 0;");
            if (ps) line("for (int i = 0; i < 4; ++i) mk |= (a.u32[i] >> 31) << i | (b.u32[i] >> 31) << (i + 4);");
            else line("for (int i = 0; i < 2; ++i) mk |= (uint32_t)(a.u64[i] >> 63) << i | (uint32_t)(b.u64[i] >> 63) << (i + 2);");
            line(wr(ops[0], ops[0].size, "mk"));
            return true;
        }
        case ZYDIS_MNEMONIC_VCVTPS2PD: case ZYDIS_MNEMONIC_VCVTDQ2PD: { // ymm <- xmm/m128
            count_insn();
            const bool ps = m == ZYDIS_MNEMONIC_VCVTPS2PD;
            line("const VpXmm s = " + half(ops[1], false) + "; VpXmm lo, hi_;");
            line(fmt("for (int i = 0; i < 2; ++i) { lo.f64[i] = (double)s.%s[i]; hi_.f64[i] = (double)s.%s[2 + i]; }", ps ? "f32" : "i32", ps ? "f32" : "i32"));
            put256(ops[0]);
            return true;
        }
        case ZYDIS_MNEMONIC_VCVTPD2PS: case ZYDIS_MNEMONIC_VCVTTPD2DQ: case ZYDIS_MNEMONIC_VCVTPD2DQ: { // xmm <- ymm/m256
            count_insn();
            line("const VpXmm a = " + half(ops[1], false) + ", b = " + half(ops[1], true) + "; VpXmm r;");
            const char* f = m == ZYDIS_MNEMONIC_VCVTPD2PS ? "r.f32[%s] = (float)%s;" : m == ZYDIS_MNEMONIC_VCVTTPD2DQ ? "r.i32[%s] = vp_cvtt_f64_i32(%s);" : "r.i32[%s] = vp_cvt_f64_i32(%s);";
            for (int i = 0; i < 4; ++i) line(fmt(f, std::to_string(i).c_str(), fmt("%s.f64[%d]", i < 2 ? "a" : "b", i & 1).c_str()));
            line(X(xmm_index(ops[0].reg.value), false) + " = r; " + zero_upper(ops[0]));
            return true;
        }
        default: break;
        }
        (void)n;
        return false;
    }

    // Legacy operations whose VEX.256 form is the 128-bit operation on each half independently
    // (AVX1 is float-only at 256 bits; Jaguar has no AVX2). The immediate of a few of them has
    // separate bits per half: `imm_shift` is how far the upper half's bits are.
    static bool lane_split(ZydisMnemonic m, int* imm_shift) {
        *imm_shift = 0;
        switch (m) {
        case ZYDIS_MNEMONIC_SHUFPD: case ZYDIS_MNEMONIC_BLENDPD: *imm_shift = 2; return true;
        case ZYDIS_MNEMONIC_BLENDPS: *imm_shift = 4; return true;
        case ZYDIS_MNEMONIC_MOVAPS: case ZYDIS_MNEMONIC_MOVUPS: case ZYDIS_MNEMONIC_MOVAPD: case ZYDIS_MNEMONIC_MOVUPD:
        case ZYDIS_MNEMONIC_MOVDQA: case ZYDIS_MNEMONIC_MOVDQU: case ZYDIS_MNEMONIC_LDDQU:
        case ZYDIS_MNEMONIC_MOVNTPS: case ZYDIS_MNEMONIC_MOVNTPD: case ZYDIS_MNEMONIC_MOVNTDQ:
        case ZYDIS_MNEMONIC_ADDPS: case ZYDIS_MNEMONIC_SUBPS: case ZYDIS_MNEMONIC_MULPS: case ZYDIS_MNEMONIC_DIVPS:
        case ZYDIS_MNEMONIC_ADDPD: case ZYDIS_MNEMONIC_SUBPD: case ZYDIS_MNEMONIC_MULPD: case ZYDIS_MNEMONIC_DIVPD:
        case ZYDIS_MNEMONIC_MINPS: case ZYDIS_MNEMONIC_MAXPS: case ZYDIS_MNEMONIC_MINPD: case ZYDIS_MNEMONIC_MAXPD:
        case ZYDIS_MNEMONIC_SQRTPS: case ZYDIS_MNEMONIC_SQRTPD: case ZYDIS_MNEMONIC_RSQRTPS: case ZYDIS_MNEMONIC_RCPPS:
        case ZYDIS_MNEMONIC_ANDPS: case ZYDIS_MNEMONIC_ANDPD: case ZYDIS_MNEMONIC_ANDNPS: case ZYDIS_MNEMONIC_ANDNPD:
        case ZYDIS_MNEMONIC_ORPS: case ZYDIS_MNEMONIC_ORPD: case ZYDIS_MNEMONIC_XORPS: case ZYDIS_MNEMONIC_XORPD:
        case ZYDIS_MNEMONIC_SHUFPS: case ZYDIS_MNEMONIC_UNPCKLPS: case ZYDIS_MNEMONIC_UNPCKHPS: case ZYDIS_MNEMONIC_UNPCKLPD: case ZYDIS_MNEMONIC_UNPCKHPD:
        case ZYDIS_MNEMONIC_BLENDVPS: case ZYDIS_MNEMONIC_BLENDVPD: case ZYDIS_MNEMONIC_CMPPS: case ZYDIS_MNEMONIC_CMPPD:
        case ZYDIS_MNEMONIC_CVTDQ2PS: case ZYDIS_MNEMONIC_CVTTPS2DQ: case ZYDIS_MNEMONIC_CVTPS2DQ:
        case ZYDIS_MNEMONIC_HADDPS: case ZYDIS_MNEMONIC_MOVSHDUP: case ZYDIS_MNEMONIC_MOVSLDUP: case ZYDIS_MNEMONIC_MOVDDUP:
        case ZYDIS_MNEMONIC_DPPS: case ZYDIS_MNEMONIC_ROUNDPS: case ZYDIS_MNEMONIC_ROUNDPD:
        case ZYDIS_MNEMONIC_HADDPD: case ZYDIS_MNEMONIC_HSUBPS: case ZYDIS_MNEMONIC_HSUBPD:
        case ZYDIS_MNEMONIC_ADDSUBPS: case ZYDIS_MNEMONIC_ADDSUBPD:
            return true;
        default: return false;
        }
    }

    // VEX: the 128-bit forms are the legacy operation with the result in a separate destination
    // (dst = src1, then dst op= src2; scalar ops take their upper lanes from src1, which the copy
    // provides) and the destination's bits 128..255 zeroed. The 256-bit forms of lane-wise
    // operations run the same thing on each 128-bit half (`hi`: upper halves, memory + 16).
    bool emit_vex() {
        const ZydisMnemonic legacy = legacy_of(insn->mnemonic);
        if (insn->avx.vector_length == 256) {
            int imm_shift = 0;
            if (!lane_split(legacy, &imm_shift)) {
                count_insn();
                unsupported((std::string("ymm-") + ZydisMnemonicGetString(insn->mnemonic)).c_str());
                return true;
            }
            const ZydisDecodedOperand* const original = ops;
            ZydisDecodedOperand upper[ZYDIS_MAX_OPERAND_COUNT];
            std::memcpy(upper, ops, sizeof upper);
            for (int i = 0; i < insn->operand_count_visible; ++i) {
                if (upper[i].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) upper[i].imm.value.u >>= imm_shift;
            }
            line("{");
            bool r = emit_vex_lane();
            line("}");
            line("{");
            hi = true;
            ops = upper;
            r = emit_vex_lane();
            ops = original;
            hi = false;
            line("}");
            stats.instructions--; // counted once
            stats.by_mnemonic[ZydisMnemonicGetString(insn->mnemonic)]--;
            return r;
        }
        const bool r = emit_vex_lane();
        if (is_vreg(ops[0]) && (ops[0].actions & ZYDIS_OPERAND_ACTION_MASK_WRITE)) line(zero_upper(ops[0]));
        return r;
    }

    bool emit_vex_lane() {
        const ZydisMnemonic legacy = legacy_of(insn->mnemonic);
        const int n = insn->operand_count_visible;
        ZydisDecodedOperand remapped[ZYDIS_MAX_OPERAND_COUNT];
        std::memcpy(remapped, ops, sizeof remapped);
        const ZydisDecodedOperand* saved = ops;
        int count = n;
        // Shift-by-immediate forms (vpslld $imm, src, dst) are 3-operand VEX whose legacy form is
        // 2-operand (dst, imm); the other 3-operand-with-immediate forms (vpshufd) are 3-operand legacy.
        const bool shift_imm = legacy == ZYDIS_MNEMONIC_PSLLDQ || legacy == ZYDIS_MNEMONIC_PSRLDQ || legacy == ZYDIS_MNEMONIC_PSLLD ||
                               legacy == ZYDIS_MNEMONIC_PSRLD || legacy == ZYDIS_MNEMONIC_PSRAD || legacy == ZYDIS_MNEMONIC_PSLLQ ||
                               legacy == ZYDIS_MNEMONIC_PSRLQ || legacy == ZYDIS_MNEMONIC_PSLLW || legacy == ZYDIS_MNEMONIC_PSRLW ||
                               legacy == ZYDIS_MNEMONIC_PSRAW;
        const bool three = n >= 3 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER &&
                           xmm_index(ops[0].reg.value) >= 0 && xmm_index(ops[1].reg.value) >= 0 &&
                           (shift_imm || !(ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE && n == 3));
        if (three) {
            // dst = src1 (when they differ), then the legacy 2-operand form dst op= src2 [, imm].
            // Every source is read before the destination changes: the destination may also be
            // a source (vblendvpd xmm0, xmm15, xmm14, xmm0), so the sources are copied first.
            line("const VpXmm vsrc1 = " + X(xmm_index(ops[1].reg.value), hi) + ";");
            remapped[1] = ops[2];
            if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER && xmm_index(ops[2].reg.value) >= 0) {
                line("const VpXmm vsrc2 = " + X(xmm_index(ops[2].reg.value), hi) + ";");
                vex_src2 = true;
            }
            if (n >= 4) remapped[2] = ops[3];
            count = n - 1;
            vex_mask = (n >= 4 && ops[3].type == ZYDIS_OPERAND_TYPE_REGISTER && xmm_index(ops[3].reg.value) >= 0) ? xmm_index(ops[3].reg.value) : -1;
            if (vex_mask >= 0) line("const VpXmm vmask = " + X(vex_mask, hi) + ";");
            line(X(xmm_index(ops[0].reg.value), hi) + " = vsrc1;");
        }
        ops = remapped;
        const bool r = emit_legacy(legacy, count);
        ops = saved;
        vex_mask = -1;
        vex_src2 = false;
        return r;
    }

    // -- x87 ------------------------------------------------------------------------------------
    // Every x87 instruction is a call into runtime/vp_x87.c (exact 80-bit arithmetic); the stack
    // registers are named by their ST(i) index.
    static int st_index(ZydisRegister r) { return (r >= ZYDIS_REGISTER_ST0 && r <= ZYDIS_REGISTER_ST7) ? r - ZYDIS_REGISTER_ST0 : -1; }
    // The memory kind of an operand: 0 f32, 1 f64, 2 f80, 3 i16, 4 i32, 5 i64.
    static int fkind(const ZydisDecodedOperand& o, bool integer) {
        if (integer) return o.size == 16 ? 3 : o.size == 32 ? 4 : 5;
        return o.size == 32 ? 0 : o.size == 64 ? 1 : 2;
    }
    bool emit_x87(ZydisMnemonic m) {
        using M = ZydisMnemonic;
        const int nv = insn->operand_count_visible;
        const bool mem = nv >= 1 && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY;
        auto st = [&](int k) { return nv > k && ops[k].type == ZYDIS_OPERAND_TYPE_REGISTER ? st_index(ops[k].reg.value) : -1; };
        int op = -1;
        bool pop = false, integer = false;
        switch (m) {
        case ZYDIS_MNEMONIC_FADD: case ZYDIS_MNEMONIC_FADDP: case ZYDIS_MNEMONIC_FIADD: op = 0; break;
        case ZYDIS_MNEMONIC_FSUB: case ZYDIS_MNEMONIC_FSUBP: case ZYDIS_MNEMONIC_FISUB: op = 1; break;
        case ZYDIS_MNEMONIC_FSUBR: case ZYDIS_MNEMONIC_FSUBRP: case ZYDIS_MNEMONIC_FISUBR: op = 2; break;
        case ZYDIS_MNEMONIC_FMUL: case ZYDIS_MNEMONIC_FMULP: case ZYDIS_MNEMONIC_FIMUL: op = 3; break;
        case ZYDIS_MNEMONIC_FDIV: case ZYDIS_MNEMONIC_FDIVP: case ZYDIS_MNEMONIC_FIDIV: op = 4; break;
        case ZYDIS_MNEMONIC_FDIVR: case ZYDIS_MNEMONIC_FDIVRP: case ZYDIS_MNEMONIC_FIDIVR: op = 5; break;
        default: break;
        }
        if (op >= 0) {
            pop = m == ZYDIS_MNEMONIC_FADDP || m == ZYDIS_MNEMONIC_FSUBP || m == ZYDIS_MNEMONIC_FSUBRP || m == ZYDIS_MNEMONIC_FMULP ||
                  m == ZYDIS_MNEMONIC_FDIVP || m == ZYDIS_MNEMONIC_FDIVRP;
            integer = m == ZYDIS_MNEMONIC_FIADD || m == ZYDIS_MNEMONIC_FISUB || m == ZYDIS_MNEMONIC_FISUBR || m == ZYDIS_MNEMONIC_FIMUL ||
                      m == ZYDIS_MNEMONIC_FIDIV || m == ZYDIS_MNEMONIC_FIDIVR;
            if (mem) line(fmt("vp_x87_arith_mem(cpu, %d, %s, %d);", op, ea(ops[0]).c_str(), fkind(ops[0], integer)));
            else if (nv >= 2) line(fmt("vp_x87_arith_reg(cpu, %d, %d, %d, %d);", op, st(0), st(1), pop));
            else line(fmt("vp_x87_arith_reg(cpu, %d, 1, 0, %d);", op, pop)); // faddp: st1 op= st0, pop
            return true;
        }
        switch (m) {
        case ZYDIS_MNEMONIC_FLD: case ZYDIS_MNEMONIC_FILD:
            if (mem) line(fmt("vp_x87_fld_mem(cpu, %s, %d);", ea(ops[0]).c_str(), fkind(ops[0], m == ZYDIS_MNEMONIC_FILD)));
            else line(fmt("vp_x87_fld_reg(cpu, %d);", st(0)));
            return true;
        case ZYDIS_MNEMONIC_FBLD: line("vp_x87_fbld(cpu, " + ea(ops[0]) + ");"); return true;
        case ZYDIS_MNEMONIC_FBSTP: line("vp_x87_fbstp(cpu, " + ea(ops[0]) + ");"); return true;
        case ZYDIS_MNEMONIC_FST: case ZYDIS_MNEMONIC_FSTP: case ZYDIS_MNEMONIC_FSTPNCE:
            pop = m != ZYDIS_MNEMONIC_FST;
            if (mem) line(fmt("vp_x87_fst_mem(cpu, %s, %d, %d, 0);", ea(ops[0]).c_str(), fkind(ops[0], false), pop));
            else line(fmt("vp_x87_fst_reg(cpu, %d, %d);", st(0), pop));
            return true;
        case ZYDIS_MNEMONIC_FIST: case ZYDIS_MNEMONIC_FISTP: case ZYDIS_MNEMONIC_FISTTP:
            line(fmt("vp_x87_fst_mem(cpu, %s, %d, %d, %d);", ea(ops[0]).c_str(), fkind(ops[0], true), m != ZYDIS_MNEMONIC_FIST,
                     m == ZYDIS_MNEMONIC_FISTTP));
            return true;
        case ZYDIS_MNEMONIC_FLD1: line("vp_x87_fld_const(cpu, 0);"); return true;
        case ZYDIS_MNEMONIC_FLDL2T: line("vp_x87_fld_const(cpu, 1);"); return true;
        case ZYDIS_MNEMONIC_FLDL2E: line("vp_x87_fld_const(cpu, 2);"); return true;
        case ZYDIS_MNEMONIC_FLDPI: line("vp_x87_fld_const(cpu, 3);"); return true;
        case ZYDIS_MNEMONIC_FLDLG2: line("vp_x87_fld_const(cpu, 4);"); return true;
        case ZYDIS_MNEMONIC_FLDLN2: line("vp_x87_fld_const(cpu, 5);"); return true;
        case ZYDIS_MNEMONIC_FLDZ: line("vp_x87_fld_const(cpu, 6);"); return true;
        case ZYDIS_MNEMONIC_FXCH: line(fmt("vp_x87_fxch(cpu, %d);", nv >= 1 ? st(0) : 1)); return true;
        case ZYDIS_MNEMONIC_FCOM: case ZYDIS_MNEMONIC_FCOMP: case ZYDIS_MNEMONIC_FUCOM: case ZYDIS_MNEMONIC_FUCOMP:
        case ZYDIS_MNEMONIC_FICOM: case ZYDIS_MNEMONIC_FICOMP: {
            pop = m == ZYDIS_MNEMONIC_FCOMP || m == ZYDIS_MNEMONIC_FUCOMP || m == ZYDIS_MNEMONIC_FICOMP;
            const bool quiet = m == ZYDIS_MNEMONIC_FUCOM || m == ZYDIS_MNEMONIC_FUCOMP;
            if (mem) line(fmt("vp_x87_fcom_mem(cpu, %s, %d, %d);", ea(ops[0]).c_str(),
                              fkind(ops[0], m == ZYDIS_MNEMONIC_FICOM || m == ZYDIS_MNEMONIC_FICOMP), pop));
            else line(fmt("vp_x87_fcom_reg(cpu, %d, %d, %d);", nv >= 1 ? st(0) : 1, pop, quiet));
            return true;
        }
        case ZYDIS_MNEMONIC_FCOMPP: line("vp_x87_fcom_reg(cpu, 1, 2, 0);"); return true;
        case ZYDIS_MNEMONIC_FUCOMPP: line("vp_x87_fcom_reg(cpu, 1, 2, 1);"); return true;
        case ZYDIS_MNEMONIC_FCOMI: case ZYDIS_MNEMONIC_FCOMIP: case ZYDIS_MNEMONIC_FUCOMI: case ZYDIS_MNEMONIC_FUCOMIP:
            line(fmt("vp_x87_fcomi(cpu, VP_FC, %d, %d, %d);", nv >= 2 ? st(1) : 1, m == ZYDIS_MNEMONIC_FCOMIP || m == ZYDIS_MNEMONIC_FUCOMIP,
                     m == ZYDIS_MNEMONIC_FUCOMI || m == ZYDIS_MNEMONIC_FUCOMIP));
            return true;
        case ZYDIS_MNEMONIC_FCMOVB: case ZYDIS_MNEMONIC_FCMOVE: case ZYDIS_MNEMONIC_FCMOVBE: case ZYDIS_MNEMONIC_FCMOVU:
        case ZYDIS_MNEMONIC_FCMOVNB: case ZYDIS_MNEMONIC_FCMOVNE: case ZYDIS_MNEMONIC_FCMOVNBE: case ZYDIS_MNEMONIC_FCMOVNU: {
            const int cc = m == ZYDIS_MNEMONIC_FCMOVB ? 2 : m == ZYDIS_MNEMONIC_FCMOVE ? 4 : m == ZYDIS_MNEMONIC_FCMOVBE ? 6 :
                           m == ZYDIS_MNEMONIC_FCMOVU ? 10 : m == ZYDIS_MNEMONIC_FCMOVNB ? 3 : m == ZYDIS_MNEMONIC_FCMOVNE ? 5 :
                           m == ZYDIS_MNEMONIC_FCMOVNBE ? 7 : 11;
            line(fmt("vp_x87_fcmov(cpu, %d, vp_cc(VP_FC, %d));", st(1), cc));
            return true;
        }
        case ZYDIS_MNEMONIC_FCHS: line("vp_x87_unary(cpu, 0);"); return true;
        case ZYDIS_MNEMONIC_FABS: line("vp_x87_unary(cpu, 1);"); return true;
        case ZYDIS_MNEMONIC_FSQRT: line("vp_x87_unary(cpu, 2);"); return true;
        case ZYDIS_MNEMONIC_FRNDINT: line("vp_x87_unary(cpu, 3);"); return true;
        case ZYDIS_MNEMONIC_FTST: line("vp_x87_ftst(cpu);"); return true;
        case ZYDIS_MNEMONIC_FXAM: line("vp_x87_fxam(cpu);"); return true;
        case ZYDIS_MNEMONIC_FPREM: line("vp_x87_fprem(cpu, 0);"); return true;
        case ZYDIS_MNEMONIC_FPREM1: line("vp_x87_fprem(cpu, 1);"); return true;
        case ZYDIS_MNEMONIC_FSCALE: line("vp_x87_fscale(cpu);"); return true;
        case ZYDIS_MNEMONIC_FXTRACT: line("vp_x87_fxtract(cpu);"); return true;
        case ZYDIS_MNEMONIC_FSIN: line("vp_x87_transcendental(cpu, 0);"); return true;
        case ZYDIS_MNEMONIC_FCOS: line("vp_x87_transcendental(cpu, 1);"); return true;
        case ZYDIS_MNEMONIC_FSINCOS: line("vp_x87_transcendental(cpu, 2);"); return true;
        case ZYDIS_MNEMONIC_FPTAN: line("vp_x87_transcendental(cpu, 3);"); return true;
        case ZYDIS_MNEMONIC_FPATAN: line("vp_x87_transcendental(cpu, 4);"); return true;
        case ZYDIS_MNEMONIC_F2XM1: line("vp_x87_transcendental(cpu, 5);"); return true;
        case ZYDIS_MNEMONIC_FYL2X: line("vp_x87_transcendental(cpu, 6);"); return true;
        case ZYDIS_MNEMONIC_FYL2XP1: line("vp_x87_transcendental(cpu, 7);"); return true;
        case ZYDIS_MNEMONIC_FINCSTP: line("vp_x87_fincstp(cpu);"); return true;
        case ZYDIS_MNEMONIC_FDECSTP: line("vp_x87_fdecstp(cpu);"); return true;
        case ZYDIS_MNEMONIC_FFREE: line(fmt("vp_x87_ffree(cpu, %d);", st(0))); return true;
        case ZYDIS_MNEMONIC_FFREEP: line(fmt("vp_x87_ffree(cpu, %d); vp_x87_fstp_discard(cpu);", st(0))); return true;
        case ZYDIS_MNEMONIC_FNINIT: line("vp_x87_fninit(cpu);"); return true;
        case ZYDIS_MNEMONIC_FNCLEX: line("vp_x87_fnclex(cpu);"); return true;
        case ZYDIS_MNEMONIC_FWAIT: case ZYDIS_MNEMONIC_FNOP: case ZYDIS_MNEMONIC_FENI8087_NOP: case ZYDIS_MNEMONIC_FDISI8087_NOP:
        case ZYDIS_MNEMONIC_FSETPM287_NOP:
            return true;
        case ZYDIS_MNEMONIC_EMMS: case ZYDIS_MNEMONIC_FEMMS: line("cpu->ftag = 0;"); return true;
        case ZYDIS_MNEMONIC_FNSTSW:
            if (mem) line("vp_st16(" + ea(ops[0]) + ", vp_x87_fnstsw(cpu));");
            else line("VP_W16(VP_RAX, vp_x87_fnstsw(cpu));");
            return true;
        case ZYDIS_MNEMONIC_FNSTCW: line("vp_st16(" + ea(ops[0]) + ", vp_x87_fnstcw(cpu));"); return true;
        case ZYDIS_MNEMONIC_FLDCW: line("vp_x87_fldcw(cpu, vp_ld16(" + ea(ops[0]) + "));"); return true;
        case ZYDIS_MNEMONIC_FNSTENV: line("vp_x87_fnstenv(cpu, " + ea(ops[0]) + ");"); return true;
        case ZYDIS_MNEMONIC_FLDENV: line("vp_x87_fldenv(cpu, " + ea(ops[0]) + ");"); return true;
        case ZYDIS_MNEMONIC_FNSAVE: line("vp_x87_fnsave(cpu, " + ea(ops[0]) + ");"); return true;
        case ZYDIS_MNEMONIC_FRSTOR: line("vp_x87_frstor(cpu, " + ea(ops[0]) + ");"); return true;
        case ZYDIS_MNEMONIC_FXSAVE: case ZYDIS_MNEMONIC_FXSAVE64: line("vp_x87_fxsave(cpu, " + ea(ops[0]) + ");"); return true;
        case ZYDIS_MNEMONIC_FXRSTOR: case ZYDIS_MNEMONIC_FXRSTOR64: line("vp_x87_fxrstor(cpu, " + ea(ops[0]) + ");"); return true;
        default: return false;
        }
        (void)integer;
    }

    static bool writes_flags_conditionally(const ZydisDecodedInstruction& i, const ZydisDecodedOperand* o) {
        switch (i.mnemonic) {
        case ZYDIS_MNEMONIC_SHL: case ZYDIS_MNEMONIC_SHR: case ZYDIS_MNEMONIC_SAR:
        case ZYDIS_MNEMONIC_ROL: case ZYDIS_MNEMONIC_ROR: case ZYDIS_MNEMONIC_RCL: case ZYDIS_MNEMONIC_RCR:
        case ZYDIS_MNEMONIC_SHLD: case ZYDIS_MNEMONIC_SHRD: {
            // The count is the last visible operand; an immediate one that is not 0 once masked always writes.
            const ZydisDecodedOperand& c = o[i.operand_count_visible - 1];
            if (i.operand_count_visible >= 2 && c.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                const unsigned mask = o[0].size == 64 ? 63u : 31u;
                return (c.imm.value.u & mask) == 0;
            }
            return i.operand_count_visible >= 2; // by cl (the 1-operand forms shift by 1)
        }
        case ZYDIS_MNEMONIC_CMPSB: case ZYDIS_MNEMONIC_CMPSW: case ZYDIS_MNEMONIC_CMPSD: case ZYDIS_MNEMONIC_CMPSQ:
        case ZYDIS_MNEMONIC_SCASB: case ZYDIS_MNEMONIC_SCASW: case ZYDIS_MNEMONIC_SCASD: case ZYDIS_MNEMONIC_SCASQ:
            return i.meta.category == ZYDIS_CATEGORY_STRINGOP &&
                   (i.attributes & (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE | ZYDIS_ATTRIB_HAS_REPNE)) != 0;
        default: return false;
        }
    }

    // bt* m, reg: the word holding the bit, ea + (signed offset >> log2(bits)) * bytes.
    std::string bt_ea() {
        const int bits = ops[0].size;
        const int sh = bits == 16 ? 4 : bits == 32 ? 5 : 6;
        return fmt("(%s + (uint64_t)(vp_sext(%d, %s) >> %d) * %d)", ea(ops[0]).c_str(), ops[1].size, rd(ops[1], ops[1].size).c_str(), sh, bits / 8);
    }

    // A LOCK-prefixed read-modify-write with no single atomic counterpart: the normal operation
    // inside a compare-and-swap loop on the memory operand (vp_old in, vp_new out).
    static bool lock_loop(ZydisMnemonic m) {
        switch (m) {
        case ZYDIS_MNEMONIC_INC: case ZYDIS_MNEMONIC_DEC: case ZYDIS_MNEMONIC_NEG: case ZYDIS_MNEMONIC_NOT:
        case ZYDIS_MNEMONIC_BTS: case ZYDIS_MNEMONIC_BTR: case ZYDIS_MNEMONIC_BTC:
        case ZYDIS_MNEMONIC_ADC: case ZYDIS_MNEMONIC_SBB: case ZYDIS_MNEMONIC_AND: case ZYDIS_MNEMONIC_OR: case ZYDIS_MNEMONIC_XOR:
            return true;
        default: return false;
        }
    }

    bool emit_legacy(ZydisMnemonic mnemonic, int operand_count) {
        if (!lock_rmw && (insn->attributes & ZYDIS_ATTRIB_HAS_LOCK) && operand_count >= 1 &&
            ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY && lock_loop(mnemonic)) {
            const int bits = ops[0].size;
            const bool bt_reg = (mnemonic == ZYDIS_MNEMONIC_BTS || mnemonic == ZYDIS_MNEMONIC_BTR || mnemonic == ZYDIS_MNEMONIC_BTC) &&
                                operand_count >= 2 && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER;
            line("const uint64_t ea = " + (bt_reg ? bt_ea() : ea(ops[0])) + ";");
            line(fmt("uint%d_t vp_old = vp_atomic_ld%d(ea), vp_new = vp_old;", bits, bits));
            // adc/sbb: the carry in is the one before the instruction, not one a failed try wrote.
            line("const uint8_t vp_cin = VP_FC->cf; (void)vp_cin;");
            line("for (;;) {");
            struct Guard { bool& f; ~Guard() { f = false; } } guard{lock_rmw};
            lock_rmw = true;
            const bool r = emit_legacy(mnemonic, operand_count);
            line(fmt("if (vp_cas%d(ea, &vp_old, vp_new)) break;", bits));
            line("}");
            return r;
        }
        return emit_legacy_op(mnemonic, operand_count);
    }

    bool emit_legacy_op(ZydisMnemonic mnemonic, int operand_count) {
        using M = ZydisMnemonic;
        const M m = mnemonic;
        const int nops = operand_count;
        const int bits = nops ? ops[0].size : 0;
        const char* name = ZydisMnemonicGetString(insn->mnemonic);
        stats.instructions++;
        stats.by_mnemonic[name]++;
        if (opt.emit_rip_updates) line("cpu->rip = " + A(rip) + ";");
        if (opt.trace) line("vp_trace(cpu, " + A(rip) + ");");
        if (emit_x87(m)) return true;

        switch (m) {
        // Data movement
        case ZYDIS_MNEMONIC_MOV:
            if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER && ZydisRegisterGetClass(ops[0].reg.value) == ZYDIS_REGCLASS_SEGMENT) { unsupported("mov-seg"); return true; }
            if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER && ZydisRegisterGetClass(ops[1].reg.value) == ZYDIS_REGCLASS_SEGMENT) { unsupported("mov-seg"); return true; }
            line(wr(ops[0], bits, rd(ops[1], bits)));
            return true;
        case ZYDIS_MNEMONIC_MOVZX:
            line(wr(ops[0], bits, rd(ops[1], ops[1].size)));
            return true;
        case ZYDIS_MNEMONIC_MOVSX: case ZYDIS_MNEMONIC_MOVSXD:
            line(wr(ops[0], bits, fmt("(uint64_t)vp_sext(%d, %s)", ops[1].size, rd(ops[1], ops[1].size).c_str())));
            return true;
        case ZYDIS_MNEMONIC_LEA:
            line(wr(ops[0], bits, ea(ops[1])));
            return true;
        case ZYDIS_MNEMONIC_XCHG: {
            if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY || ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
                const ZydisDecodedOperand& mo = ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? ops[0] : ops[1];
                const ZydisDecodedOperand& ro = ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? ops[1] : ops[0];
                line("const uint64_t ea = " + ea(mo) + ";");
                line(fmt("const uint64_t x = vp_xchg%d(ea, (uint%d_t)%s);", bits, bits, rd(ro, bits).c_str()));
                line(wr(ro, bits, "x"));
                return true;
            }
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s, y = %s;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str()));
            line(wr(ops[0], bits, "y", a));
            line(wr(ops[1], bits, "x"));
            return true;
        }
        case ZYDIS_MNEMONIC_CMOVO: case ZYDIS_MNEMONIC_CMOVNO: case ZYDIS_MNEMONIC_CMOVB: case ZYDIS_MNEMONIC_CMOVNB:
        case ZYDIS_MNEMONIC_CMOVZ: case ZYDIS_MNEMONIC_CMOVNZ: case ZYDIS_MNEMONIC_CMOVBE: case ZYDIS_MNEMONIC_CMOVNBE:
        case ZYDIS_MNEMONIC_CMOVS: case ZYDIS_MNEMONIC_CMOVNS: case ZYDIS_MNEMONIC_CMOVP: case ZYDIS_MNEMONIC_CMOVNP:
        case ZYDIS_MNEMONIC_CMOVL: case ZYDIS_MNEMONIC_CMOVNL: case ZYDIS_MNEMONIC_CMOVLE: case ZYDIS_MNEMONIC_CMOVNLE:
            // The source is read even when the move does not happen (a faulting load faults).
            line(fmt("const uint64_t v = %s;", rd(ops[1], bits).c_str()));
            line(fmt("if (vp_cc(VP_FC, %d)) { %s } else { %s }", cc_of(m), wr(ops[0], bits, "v").c_str(),
                     bits == 32 ? wr(ops[0], 32, rd(ops[0], 32)).c_str() : ""));
            return true;
        case ZYDIS_MNEMONIC_SETO: case ZYDIS_MNEMONIC_SETNO: case ZYDIS_MNEMONIC_SETB: case ZYDIS_MNEMONIC_SETNB:
        case ZYDIS_MNEMONIC_SETZ: case ZYDIS_MNEMONIC_SETNZ: case ZYDIS_MNEMONIC_SETBE: case ZYDIS_MNEMONIC_SETNBE:
        case ZYDIS_MNEMONIC_SETS: case ZYDIS_MNEMONIC_SETNS: case ZYDIS_MNEMONIC_SETP: case ZYDIS_MNEMONIC_SETNP:
        case ZYDIS_MNEMONIC_SETL: case ZYDIS_MNEMONIC_SETNL: case ZYDIS_MNEMONIC_SETLE: case ZYDIS_MNEMONIC_SETNLE:
            line(wr(ops[0], 8, fmt("vp_cc(VP_FC, %d)", cc_of(m))));
            return true;
        case ZYDIS_MNEMONIC_PUSH:
            if (ops[0].size == 16) { unsupported("push16"); return true; }
            line(fmt("VP_PUSH(%s);", rd(ops[0], 64).c_str()));
            return true;
        case ZYDIS_MNEMONIC_POP:
            if (ops[0].size == 16) { unsupported("pop16"); return true; }
            if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
                line("const uint64_t v = VP_POP();");
                line(wr(ops[0], 64, "v"));
            } else {
                line(wr(ops[0], 64, "VP_POP()"));
            }
            return true;
        case ZYDIS_MNEMONIC_LEAVE:
            line("VP_W64(VP_RSP, VP_R64(VP_RBP));");
            line("VP_W64(VP_RBP, VP_POP());");
            return true;
        case ZYDIS_MNEMONIC_CDQE: line("VP_W64(VP_RAX, (uint64_t)(int64_t)(int32_t)VP_R32(VP_RAX));"); return true;
        case ZYDIS_MNEMONIC_CWDE: line("VP_W32(VP_RAX, (uint32_t)(int32_t)(int16_t)VP_R16(VP_RAX));"); return true;
        case ZYDIS_MNEMONIC_CBW: line("VP_W16(VP_RAX, (uint16_t)(int16_t)(int8_t)VP_R8(VP_RAX));"); return true;
        case ZYDIS_MNEMONIC_CQO: line("VP_W64(VP_RDX, (VP_R64(VP_RAX) >> 63) ? ~UINT64_C(0) : 0);"); return true;
        case ZYDIS_MNEMONIC_CDQ: line("VP_W32(VP_RDX, (VP_R32(VP_RAX) >> 31) ? 0xffffffffu : 0);"); return true;
        case ZYDIS_MNEMONIC_CWD: line("VP_W16(VP_RDX, (VP_R16(VP_RAX) >> 15) ? 0xffffu : 0);"); return true;
        // CET shadow-stack queries (libgcc's unwinder probes them): without CET they are NOPs and
        // rdssp leaves its register unchanged, which tells the code there is no shadow stack.
        case ZYDIS_MNEMONIC_RDSSPD: case ZYDIS_MNEMONIC_RDSSPQ: case ZYDIS_MNEMONIC_INCSSPD: case ZYDIS_MNEMONIC_INCSSPQ:
            return true;
        case ZYDIS_MNEMONIC_NOP: case ZYDIS_MNEMONIC_ENDBR64: case ZYDIS_MNEMONIC_PAUSE: case ZYDIS_MNEMONIC_FNOP:
        case ZYDIS_MNEMONIC_PREFETCHT0: case ZYDIS_MNEMONIC_PREFETCHT1: case ZYDIS_MNEMONIC_PREFETCHT2:
        case ZYDIS_MNEMONIC_PREFETCHNTA: case ZYDIS_MNEMONIC_PREFETCHW:
            return true;
        case ZYDIS_MNEMONIC_MFENCE: case ZYDIS_MNEMONIC_SFENCE: case ZYDIS_MNEMONIC_LFENCE:
            line("__atomic_thread_fence(__ATOMIC_SEQ_CST);");
            return true;

        // Arithmetic
        case ZYDIS_MNEMONIC_ADD: alu2("add", "+", true); return true;
        case ZYDIS_MNEMONIC_SUB: alu2("sub", "-", true); return true;
        case ZYDIS_MNEMONIC_CMP: alu2("sub", "-", false); return true;
        case ZYDIS_MNEMONIC_AND: alu2("logic", "&", true); return true;
        case ZYDIS_MNEMONIC_OR: alu2("logic", "|", true); return true;
        case ZYDIS_MNEMONIC_XOR: alu2("logic", "^", true); return true;
        case ZYDIS_MNEMONIC_TEST: alu2("logic", "&", false); return true;
        case ZYDIS_MNEMONIC_ADC: {
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s, y = %s, c = %s;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str(), lock_rmw ? "vp_cin" : "VP_FC->cf"));
            line(fmt("const uint64_t r = (x + y + c) & VP_MASK(%d);", bits));
            line(fmt("vp_flags_adc(VP_FC, %d, x, y, c, r);", bits));
            line(wr(ops[0], bits, "r", a));
            return true;
        }
        case ZYDIS_MNEMONIC_SBB: {
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s, y = %s, c = %s;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str(), lock_rmw ? "vp_cin" : "VP_FC->cf"));
            line(fmt("const uint64_t r = (x - y - c) & VP_MASK(%d);", bits));
            line(fmt("vp_flags_sbb(VP_FC, %d, x, y, c, r);", bits));
            line(wr(ops[0], bits, "r", a));
            return true;
        }
        case ZYDIS_MNEMONIC_INC: case ZYDIS_MNEMONIC_DEC: {
            const bool inc = m == ZYDIS_MNEMONIC_INC;
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s;", rd(ops[0], bits, a).c_str()));
            line(fmt("const uint64_t r = (x %s 1) & VP_MASK(%d);", inc ? "+" : "-", bits));
            line(fmt("vp_flags_%s(VP_FC, %d, x, r);", inc ? "inc" : "dec", bits));
            line(wr(ops[0], bits, "r", a));
            return true;
        }
        case ZYDIS_MNEMONIC_NEG: {
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s;", rd(ops[0], bits, a).c_str()));
            line(fmt("const uint64_t r = (0 - x) & VP_MASK(%d);", bits));
            line(fmt("vp_flags_sub(VP_FC, %d, 0, x, r);", bits));
            line(wr(ops[0], bits, "r", a));
            return true;
        }
        case ZYDIS_MNEMONIC_NOT: {
            const std::string a = bind_addr(ops[0]);
            line(wr(ops[0], bits, fmt("~%s", rd(ops[0], bits, a).c_str()), a));
            return true;
        }
        case ZYDIS_MNEMONIC_SHL: case ZYDIS_MNEMONIC_SALC: shift("shl"); return true;
        case ZYDIS_MNEMONIC_SHR: shift("shr"); return true;
        case ZYDIS_MNEMONIC_SAR: shift("sar"); return true;
        case ZYDIS_MNEMONIC_ROL: rotate(true); return true;
        case ZYDIS_MNEMONIC_ROR: rotate(false); return true;
        case ZYDIS_MNEMONIC_IMUL:
            if (nops == 1) { line(fmt("vp_mul1(cpu, %d, %s, 1);", bits, rd(ops[0], bits).c_str())); return true; }
            if (nops == 2) { line(wr(ops[0], bits, fmt("vp_imul(VP_FC, %d, %s, %s)", bits, rd(ops[0], bits).c_str(), rd(ops[1], bits).c_str()))); return true; }
            line(wr(ops[0], bits, fmt("vp_imul(VP_FC, %d, %s, %s)", bits, rd(ops[1], bits).c_str(), rd(ops[2], bits).c_str())));
            return true;
        case ZYDIS_MNEMONIC_MUL: line(fmt("vp_mul1(cpu, %d, %s, 0);", bits, rd(ops[0], bits).c_str())); return true;
        case ZYDIS_MNEMONIC_DIV: case ZYDIS_MNEMONIC_IDIV:
            line(fmt("if (vp_div1(cpu, %d, %s, %d)) { vp_divide_error(cpu, %s); return; }", bits, rd(ops[0], bits).c_str(),
                     m == ZYDIS_MNEMONIC_IDIV, A(rip).c_str()));
            return true;
        case ZYDIS_MNEMONIC_BT: case ZYDIS_MNEMONIC_BTS: case ZYDIS_MNEMONIC_BTR: case ZYDIS_MNEMONIC_BTC: {
            std::string a;
            if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                // A register bit offset is signed and reaches outside the operand.
                if (!lock_rmw) line("const uint64_t ea = " + bt_ea() + ";");
                a = "ea";
            } else {
                a = bind_addr(ops[0]);
            }
            line(fmt("const uint64_t x = %s; const unsigned b = (unsigned)(%s) & %u;", rd(ops[0], bits, a).c_str(), rd(ops[1], ops[1].size).c_str(), bits - 1));
            line("VP_FC->cf = (uint8_t)((x >> b) & 1);");
            if (m == ZYDIS_MNEMONIC_BTS) line(wr(ops[0], bits, "x | (UINT64_C(1) << b)", a));
            if (m == ZYDIS_MNEMONIC_BTR) line(wr(ops[0], bits, "x & ~(UINT64_C(1) << b)", a));
            if (m == ZYDIS_MNEMONIC_BTC) line(wr(ops[0], bits, "x ^ (UINT64_C(1) << b)", a));
            return true;
        }
        case ZYDIS_MNEMONIC_BSF: case ZYDIS_MNEMONIC_BSR: case ZYDIS_MNEMONIC_TZCNT: case ZYDIS_MNEMONIC_LZCNT: {
            line(fmt("const uint64_t x = %s & VP_MASK(%d);", rd(ops[1], bits).c_str(), bits));
            if (m == ZYDIS_MNEMONIC_BSF || m == ZYDIS_MNEMONIC_BSR) {
                line("VP_FC->zf = (x == 0);");
                line("if (x) { " + wr(ops[0], bits, m == ZYDIS_MNEMONIC_BSF ? "(uint64_t)__builtin_ctzll(x)" : "(uint64_t)(63 - __builtin_clzll(x))") + " }");
            } else if (m == ZYDIS_MNEMONIC_TZCNT) {
                line(fmt("VP_FC->cf = (x == 0); const uint64_t r = x ? (uint64_t)__builtin_ctzll(x) : %d; VP_FC->zf = (r == 0);", bits));
                line(wr(ops[0], bits, "r"));
            } else {
                line(fmt("VP_FC->cf = (x == 0); const uint64_t r = x ? (uint64_t)__builtin_clzll(x) - (64 - %d) : %d; VP_FC->zf = (r == 0);", bits, bits));
                line(wr(ops[0], bits, "r"));
            }
            return true;
        }
        case ZYDIS_MNEMONIC_POPCNT:
            line(fmt("const uint64_t r = (uint64_t)__builtin_popcountll(%s & VP_MASK(%d));", rd(ops[1], bits).c_str(), bits));
            line("VP_FC->cf = VP_FC->of = VP_FC->sf = VP_FC->pf = VP_FC->af = 0; VP_FC->zf = (r == 0);");
            line(wr(ops[0], bits, "r"));
            return true;
        case ZYDIS_MNEMONIC_BSWAP:
            line(wr(ops[0], bits, fmt(bits == 64 ? "__builtin_bswap64(%s)" : "__builtin_bswap32((uint32_t)%s)", rd(ops[0], bits).c_str())));
            return true;
        case ZYDIS_MNEMONIC_CLC: line("VP_FC->cf = 0;"); return true;
        case ZYDIS_MNEMONIC_STC: line("VP_FC->cf = 1;"); return true;
        case ZYDIS_MNEMONIC_CMC: line("VP_FC->cf ^= 1;"); return true;
        case ZYDIS_MNEMONIC_CLD: line("VP_FC->df = 0;"); return true;
        case ZYDIS_MNEMONIC_STD: line("VP_FC->df = 1;"); return true;

        // Control flow
        case ZYDIS_MNEMONIC_JMP: {
            if (uint64_t t = branch_target(*insn, ops, rip); t && opt.natives.count(t)) {
                line(fmt("vp_call_native(cpu, %s); return;", A(t).c_str())); // a tail call: the native pops our caller's return address
            } else if (t && img.is_code(t)) {
                line("goto " + label(t) + ";");
            } else if (current && current->jump_tables.count(rip)) {
                stats.jump_tables++;
                line(fmt("const uint64_t t = %s;", rd(ops[0], 64).c_str()));
                line(opt.pic ? "switch (t - VP_MOD.base) {" : "switch (t) {");
                for (uint64_t t : current->jump_tables.at(rip)) line(fmt("case 0x%" PRIx64 "ull: goto %s;", opt.pic ? t - img.base : t, label(t).c_str()));
                line("default: cpu->rip = t; vp_dispatch(cpu, t); return;");
                line("}");
            } else {
                stats.indirect_jumps++;
                if (stats.indirect_jump_sites.size() < 400) {
                    // The jump and up to four instructions before it in the block, for the report.
                    std::string context;
                    for (uint64_t q : block_recent) {
                        ZydisDecodedInstruction i2; ZydisDecodedOperand o2[ZYDIS_MAX_OPERAND_COUNT];
                        if (dec.decode(img, q, i2, o2)) context += format_insn(i2, o2, q) + " ; ";
                    }
                    stats.indirect_jump_sites.emplace_back(rip, context + format_insn(*insn, ops, rip));
                }
                line(fmt("cpu->rip = %s; vp_dispatch(cpu, cpu->rip); return;", rd(ops[0], 64).c_str()));
            }
            return false;
        }
        case ZYDIS_MNEMONIC_CALL: {
            // The target is read before the return address is pushed (it may be rsp-relative).
            if (uint64_t t = branch_target(*insn, ops, rip); t && opt.natives.count(t)) {
                line(fmt("VP_PUSH(%s);", A(next).c_str()));
                line(fmt("vp_call_native(cpu, %s);", A(t).c_str()));
            } else if (t && img.is_code(t)) {
                line(fmt("VP_PUSH(%s);", A(next).c_str()));
                line(fmt("%s(cpu, 0);", fn_name(t).c_str()));
            } else {
                stats.indirect_calls++;
                line(fmt("const uint64_t target = %s;", rd(ops[0], 64).c_str()));
                line(fmt("VP_PUSH(%s);", A(next).c_str()));
                line("vp_dispatch(cpu, target);");
            }
            // The callee returned to the address after the call unless it unwound somewhere else.
            line(fmt("if (cpu->rip != %s) return;", A(next).c_str()));
            return true;
        }
        case ZYDIS_MNEMONIC_RET:
            line("cpu->rip = VP_POP();");
            if (nops && ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) line(fmt("VP_W64(VP_RSP, VP_R64(VP_RSP) + %" PRIu64 ");", ops[0].imm.value.u));
            line("return;");
            return false;
        case ZYDIS_MNEMONIC_JCXZ: case ZYDIS_MNEMONIC_JECXZ: case ZYDIS_MNEMONIC_JRCXZ: {
            const uint64_t t = branch_target(*insn, ops, rip);
            const int w = m == ZYDIS_MNEMONIC_JRCXZ ? 64 : m == ZYDIS_MNEMONIC_JECXZ ? 32 : 16;
            line(fmt("if (VP_R%d(VP_RCX) == 0) goto %s;", w, label(t).c_str()));
            return true;
        }
        case ZYDIS_MNEMONIC_LOOP: case ZYDIS_MNEMONIC_LOOPE: case ZYDIS_MNEMONIC_LOOPNE: {
            const uint64_t t = branch_target(*insn, ops, rip);
            line("VP_W64(VP_RCX, VP_R64(VP_RCX) - 1);");
            const char* extra = m == ZYDIS_MNEMONIC_LOOPE ? " && VP_FC->zf" : m == ZYDIS_MNEMONIC_LOOPNE ? " && !VP_FC->zf" : "";
            line(fmt("if (VP_R64(VP_RCX) != 0%s) goto %s;", extra, label(t).c_str()));
            return true;
        }
        case ZYDIS_MNEMONIC_UD2: case ZYDIS_MNEMONIC_HLT: case ZYDIS_MNEMONIC_INT3:
            unsupported(name);
            return false;
        // Jcc: the conditional goto is written by emit_function after the body.
        case ZYDIS_MNEMONIC_JO: case ZYDIS_MNEMONIC_JNO: case ZYDIS_MNEMONIC_JB: case ZYDIS_MNEMONIC_JNB:
        case ZYDIS_MNEMONIC_JZ: case ZYDIS_MNEMONIC_JNZ: case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_JNBE:
        case ZYDIS_MNEMONIC_JS: case ZYDIS_MNEMONIC_JNS: case ZYDIS_MNEMONIC_JP: case ZYDIS_MNEMONIC_JNP:
        case ZYDIS_MNEMONIC_JL: case ZYDIS_MNEMONIC_JNL: case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_JNLE:
            return true;
        case ZYDIS_MNEMONIC_CPUID: line("vp_cpuid(cpu);"); return true;
        case ZYDIS_MNEMONIC_SYSCALL:
            // The host implements the system call from the registers; rcx/r11 are clobbered as on hardware.
            line(fmt("cpu->rip = %s; vp_syscall(cpu); VP_W64(1, %s); VP_W64(11, 0x202);", A(next).c_str(), A(next).c_str()));
            return true;
        case ZYDIS_MNEMONIC_RDTSCP:
            line("const uint64_t t = vp_rdtsc(cpu); VP_W32(VP_RAX, (uint32_t)t); VP_W32(VP_RDX, (uint32_t)(t >> 32)); VP_W32(VP_RCX, 0);");
            return true;
        case ZYDIS_MNEMONIC_XGETBV:
            line("VP_W32(VP_RAX, 0x7); VP_W32(VP_RDX, 0);"); // x87, SSE and AVX state enabled
            return true;
        case ZYDIS_MNEMONIC_VZEROUPPER: case ZYDIS_MNEMONIC_VZEROALL:
            if (m == ZYDIS_MNEMONIC_VZEROALL) line("memset(cpu->xmm, 0, sizeof cpu->xmm);");
            return true;
        case ZYDIS_MNEMONIC_RDTSC:
            line("const uint64_t t = vp_rdtsc(cpu); VP_W32(VP_RAX, (uint32_t)t); VP_W32(VP_RDX, (uint32_t)(t >> 32));");
            return true;
        case ZYDIS_MNEMONIC_LDMXCSR: line(fmt("cpu->mxcsr = %s; vp_apply_mxcsr(cpu);", rd(ops[0], 32).c_str())); return true;
        case ZYDIS_MNEMONIC_STMXCSR: line(wr(ops[0], 32, "cpu->mxcsr")); return true;

        // String instructions (the rep forms the compilers emit: movs, stos; one element otherwise).
        case ZYDIS_MNEMONIC_MOVSB: string_op(8, true); return true;
        case ZYDIS_MNEMONIC_MOVSW: string_op(16, true); return true;
        case ZYDIS_MNEMONIC_MOVSQ: string_op(64, true); return true;
        case ZYDIS_MNEMONIC_STOSB: string_op(8, false); return true;
        case ZYDIS_MNEMONIC_STOSW: string_op(16, false); return true;
        case ZYDIS_MNEMONIC_STOSD: string_op(32, false); return true;
        case ZYDIS_MNEMONIC_STOSQ: string_op(64, false); return true;
        case ZYDIS_MNEMONIC_LODSB: string_scan(8, 0); return true;
        case ZYDIS_MNEMONIC_LODSW: string_scan(16, 0); return true;
        case ZYDIS_MNEMONIC_LODSD: string_scan(32, 0); return true;
        case ZYDIS_MNEMONIC_LODSQ: string_scan(64, 0); return true;
        case ZYDIS_MNEMONIC_SCASB: string_scan(8, 1); return true;
        case ZYDIS_MNEMONIC_SCASW: string_scan(16, 1); return true;
        case ZYDIS_MNEMONIC_SCASD: string_scan(32, 1); return true;
        case ZYDIS_MNEMONIC_SCASQ: string_scan(64, 1); return true;
        case ZYDIS_MNEMONIC_CMPSB: string_scan(8, 2); return true;
        case ZYDIS_MNEMONIC_CMPSW: string_scan(16, 2); return true;
        case ZYDIS_MNEMONIC_CMPSQ: string_scan(64, 2); return true;

        // SSE moves
        case ZYDIS_MNEMONIC_MOVAPS: case ZYDIS_MNEMONIC_MOVUPS: case ZYDIS_MNEMONIC_MOVAPD: case ZYDIS_MNEMONIC_MOVUPD:
        case ZYDIS_MNEMONIC_MOVDQA: case ZYDIS_MNEMONIC_MOVDQU: case ZYDIS_MNEMONIC_LDDQU: case ZYDIS_MNEMONIC_MOVNTDQ:
        case ZYDIS_MNEMONIC_MOVNTPS: case ZYDIS_MNEMONIC_MOVNTPD:
            if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) line(reg_wr(ops[0].reg.value, 128, xmm_rd(ops[1])));
            else line("vp_st128(" + ea(ops[0]) + ", " + xmm_rd(ops[1]) + ");");
            return true;
        case ZYDIS_MNEMONIC_MOVSS: case ZYDIS_MNEMONIC_MOVSD: {
            if (m == ZYDIS_MNEMONIC_MOVSD && insn->meta.category == ZYDIS_CATEGORY_STRINGOP) { string_op(32, true); return true; }
            const bool dbl = m == ZYDIS_MNEMONIC_MOVSD;
            const int w = dbl ? 64 : 32;
            if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                line(fmt("%s.u%d[0] = %s.u%d[0];", xmm_dst().c_str(), w, xmm_rd(ops[1]).c_str(), w));
            } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                // From memory: the upper lanes are zeroed.
                line(fmt("VpXmm v = {{0}}; v.u%d[0] = vp_ld%d(%s); %s", w, w, ea(ops[1]).c_str(), reg_wr(ops[0].reg.value, 128, "v").c_str()));
            } else {
                line(fmt("vp_st%d(%s, %s.u%d[0]);", w, ea(ops[0]).c_str(), reg_rd(ops[1].reg.value, 128).c_str(), w));
            }
            return true;
        }
        case ZYDIS_MNEMONIC_MOVD: case ZYDIS_MNEMONIC_MOVQ: {
            const int w = m == ZYDIS_MNEMONIC_MOVD ? 32 : 64;
            if (xmm_index(ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER ? ops[0].reg.value : ZYDIS_REGISTER_NONE) >= 0) {
                std::string src = ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER && xmm_index(ops[1].reg.value) >= 0
                                      ? reg_rd(ops[1].reg.value, 128) + fmt(".u%d[0]", w) : rd(ops[1], w);
                line(fmt("VpXmm v = {{0}}; v.u%d[0] = %s; %s", w, src.c_str(), reg_wr(ops[0].reg.value, 128, "v").c_str()));
            } else {
                line(wr(ops[0], w, reg_rd(ops[1].reg.value, 128) + fmt(".u%d[0]", w)));
            }
            return true;
        }
        case ZYDIS_MNEMONIC_MOVLPS: case ZYDIS_MNEMONIC_MOVLPD:
            if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) line(fmt("%s.u64[0] = vp_ld64(%s);", xmm_dst().c_str(), ea(ops[1]).c_str()));
            else line(fmt("vp_st64(%s, %s.u64[0]);", ea(ops[0]).c_str(), reg_rd(ops[1].reg.value, 128).c_str()));
            return true;
        case ZYDIS_MNEMONIC_MOVHPS: case ZYDIS_MNEMONIC_MOVHPD:
            if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) line(fmt("%s.u64[1] = vp_ld64(%s);", xmm_dst().c_str(), ea(ops[1]).c_str()));
            else line(fmt("vp_st64(%s, %s.u64[1]);", ea(ops[0]).c_str(), reg_rd(ops[1].reg.value, 128).c_str()));
            return true;
        case ZYDIS_MNEMONIC_MOVLHPS: line(fmt("%s.u64[1] = %s.u64[0];", xmm_dst().c_str(), xmm_rd(ops[1]).c_str())); return true;
        case ZYDIS_MNEMONIC_MOVHLPS: line(fmt("%s.u64[0] = %s.u64[1];", xmm_dst().c_str(), xmm_rd(ops[1]).c_str())); return true;

        // SSE arithmetic
        case ZYDIS_MNEMONIC_ADDSS: sse_scalar("+", false); return true;
        case ZYDIS_MNEMONIC_SUBSS: sse_scalar("-", false); return true;
        case ZYDIS_MNEMONIC_MULSS: sse_scalar("*", false); return true;
        case ZYDIS_MNEMONIC_DIVSS: sse_scalar("/", false); return true;
        case ZYDIS_MNEMONIC_ADDSD: sse_scalar("+", true); return true;
        case ZYDIS_MNEMONIC_SUBSD: sse_scalar("-", true); return true;
        case ZYDIS_MNEMONIC_MULSD: sse_scalar("*", true); return true;
        case ZYDIS_MNEMONIC_DIVSD: sse_scalar("/", true); return true;
        case ZYDIS_MNEMONIC_ADDPS: sse_packed("+", false); return true;
        case ZYDIS_MNEMONIC_SUBPS: sse_packed("-", false); return true;
        case ZYDIS_MNEMONIC_MULPS: sse_packed("*", false); return true;
        case ZYDIS_MNEMONIC_DIVPS: sse_packed("/", false); return true;
        case ZYDIS_MNEMONIC_ADDPD: sse_packed("+", true); return true;
        case ZYDIS_MNEMONIC_SUBPD: sse_packed("-", true); return true;
        case ZYDIS_MNEMONIC_MULPD: sse_packed("*", true); return true;
        case ZYDIS_MNEMONIC_DIVPD: sse_packed("/", true); return true;
        case ZYDIS_MNEMONIC_MINSS: sse_minmax(false, false); return true;
        case ZYDIS_MNEMONIC_MAXSS: sse_minmax(true, false); return true;
        case ZYDIS_MNEMONIC_MINSD: sse_minmax(false, true); return true;
        case ZYDIS_MNEMONIC_MAXSD: sse_minmax(true, true); return true;
        case ZYDIS_MNEMONIC_SQRTSS: line(fmt("%s.f32[0] = sqrtf(%s);", xmm_dst().c_str(), f32_rd(ops[1]).c_str())); return true;
        case ZYDIS_MNEMONIC_SQRTSD: line(fmt("%s.f64[0] = sqrt(%s);", xmm_dst().c_str(), f64_rd(ops[1]).c_str())); return true;
        case ZYDIS_MNEMONIC_SQRTPS: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 4; ++i) " + xmm_dst() + ".f32[i] = sqrtf(s.f32[i]);"); return true;
        case ZYDIS_MNEMONIC_XORPS: case ZYDIS_MNEMONIC_XORPD: case ZYDIS_MNEMONIC_PXOR: sse_bitwise("^", false); return true;
        case ZYDIS_MNEMONIC_ANDPS: case ZYDIS_MNEMONIC_ANDPD: case ZYDIS_MNEMONIC_PAND: sse_bitwise("&", false); return true;
        case ZYDIS_MNEMONIC_ORPS: case ZYDIS_MNEMONIC_ORPD: case ZYDIS_MNEMONIC_POR: sse_bitwise("|", false); return true;
        case ZYDIS_MNEMONIC_ANDNPS: case ZYDIS_MNEMONIC_ANDNPD: case ZYDIS_MNEMONIC_PANDN: sse_bitwise("&", true); return true;
        case ZYDIS_MNEMONIC_UCOMISS: case ZYDIS_MNEMONIC_COMISS:
            line(fmt("vp_comiss(VP_FC, %s.f32[0], %s);", xmm_dst().c_str(), f32_rd(ops[1]).c_str()));
            return true;
        case ZYDIS_MNEMONIC_UCOMISD: case ZYDIS_MNEMONIC_COMISD:
            line(fmt("vp_comisd(VP_FC, %s.f64[0], %s);", xmm_dst().c_str(), f64_rd(ops[1]).c_str()));
            return true;
        case ZYDIS_MNEMONIC_CVTSI2SS:
            line(fmt("%s.f32[0] = (float)(int%d_t)%s;", xmm_dst().c_str(), ops[1].size, rd(ops[1], ops[1].size).c_str()));
            return true;
        case ZYDIS_MNEMONIC_CVTSI2SD:
            line(fmt("%s.f64[0] = (double)(int%d_t)%s;", xmm_dst().c_str(), ops[1].size, rd(ops[1], ops[1].size).c_str()));
            return true;
        case ZYDIS_MNEMONIC_CVTTSS2SI: line(wr(ops[0], bits, fmt("(uint64_t)vp_cvtt_f32_i%d(%s)", bits, f32_rd(ops[1]).c_str()))); return true;
        case ZYDIS_MNEMONIC_CVTTSD2SI: line(wr(ops[0], bits, fmt("(uint64_t)vp_cvtt_f64_i%d(%s)", bits, f64_rd(ops[1]).c_str()))); return true;
        case ZYDIS_MNEMONIC_CVTSS2SI: line(wr(ops[0], bits, fmt("(uint64_t)vp_cvt_f32_i%d(%s)", bits, f32_rd(ops[1]).c_str()))); return true;
        case ZYDIS_MNEMONIC_CVTSD2SI: line(wr(ops[0], bits, fmt("(uint64_t)vp_cvt_f64_i%d(%s)", bits, f64_rd(ops[1]).c_str()))); return true;
        case ZYDIS_MNEMONIC_CVTSS2SD: line(fmt("%s.f64[0] = (double)%s;", xmm_dst().c_str(), f32_rd(ops[1]).c_str())); return true;
        case ZYDIS_MNEMONIC_CVTSD2SS: line(fmt("%s.f32[0] = (float)%s;", xmm_dst().c_str(), f64_rd(ops[1]).c_str())); return true;
        case ZYDIS_MNEMONIC_CVTDQ2PS: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 4; ++i) " + xmm_dst() + ".f32[i] = (float)s.i32[i];"); return true;
        case ZYDIS_MNEMONIC_CVTTPS2DQ: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 4; ++i) " + xmm_dst() + ".i32[i] = vp_cvtt_f32_i32(s.f32[i]);"); return true;
        case ZYDIS_MNEMONIC_SHUFPS: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            line("const VpXmm a = " + xmm_dst() + ", b = " + xmm_rd(ops[1]) + ";");
            line(fmt("%s.u32[0] = a.u32[%u]; %s.u32[1] = a.u32[%u]; %s.u32[2] = b.u32[%u]; %s.u32[3] = b.u32[%u];",
                     xmm_dst().c_str(), imm & 3, xmm_dst().c_str(), (imm >> 2) & 3, xmm_dst().c_str(), (imm >> 4) & 3, xmm_dst().c_str(), (imm >> 6) & 3));
            return true;
        }
        case ZYDIS_MNEMONIC_PSHUFD: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            line("const VpXmm b = " + xmm_rd(ops[1]) + ";");
            line(fmt("%s.u32[0] = b.u32[%u]; %s.u32[1] = b.u32[%u]; %s.u32[2] = b.u32[%u]; %s.u32[3] = b.u32[%u];",
                     xmm_dst().c_str(), imm & 3, xmm_dst().c_str(), (imm >> 2) & 3, xmm_dst().c_str(), (imm >> 4) & 3, xmm_dst().c_str(), (imm >> 6) & 3));
            return true;
        }
        case ZYDIS_MNEMONIC_UNPCKLPS: line("const VpXmm a = " + xmm_dst() + ", b = " + xmm_rd(ops[1]) + "; " + xmm_dst() + ".u32[0] = a.u32[0]; " + xmm_dst() + ".u32[1] = b.u32[0]; " + xmm_dst() + ".u32[2] = a.u32[1]; " + xmm_dst() + ".u32[3] = b.u32[1];"); return true;
        case ZYDIS_MNEMONIC_UNPCKHPS: line("const VpXmm a = " + xmm_dst() + ", b = " + xmm_rd(ops[1]) + "; " + xmm_dst() + ".u32[0] = a.u32[2]; " + xmm_dst() + ".u32[1] = b.u32[2]; " + xmm_dst() + ".u32[2] = a.u32[3]; " + xmm_dst() + ".u32[3] = b.u32[3];"); return true;
        case ZYDIS_MNEMONIC_UNPCKLPD: case ZYDIS_MNEMONIC_PUNPCKLQDQ: line("const VpXmm b = " + xmm_rd(ops[1]) + "; " + xmm_dst() + ".u64[1] = b.u64[0];"); return true;
        case ZYDIS_MNEMONIC_UNPCKHPD: case ZYDIS_MNEMONIC_PUNPCKHQDQ: line("const VpXmm a = " + xmm_dst() + ", b = " + xmm_rd(ops[1]) + "; " + xmm_dst() + ".u64[0] = a.u64[1]; " + xmm_dst() + ".u64[1] = b.u64[1];"); return true;
        case ZYDIS_MNEMONIC_PADDD: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 4; ++i) " + xmm_dst() + ".u32[i] += s.u32[i];"); return true;
        case ZYDIS_MNEMONIC_PSUBD: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 4; ++i) " + xmm_dst() + ".u32[i] -= s.u32[i];"); return true;
        case ZYDIS_MNEMONIC_PADDQ: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 2; ++i) " + xmm_dst() + ".u64[i] += s.u64[i];"); return true;
        case ZYDIS_MNEMONIC_PSUBQ: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 2; ++i) " + xmm_dst() + ".u64[i] -= s.u64[i];"); return true;
        case ZYDIS_MNEMONIC_PCMPEQD: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 4; ++i) " + xmm_dst() + ".u32[i] = (" + xmm_dst() + ".u32[i] == s.u32[i]) ? 0xffffffffu : 0;"); return true;
        case ZYDIS_MNEMONIC_PCMPEQB: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 16; ++i) " + xmm_dst() + ".u8[i] = (" + xmm_dst() + ".u8[i] == s.u8[i]) ? 0xff : 0;"); return true;
        case ZYDIS_MNEMONIC_PMOVMSKB:
            line("const VpXmm s = " + xmm_rd(ops[1]) + "; uint32_t mk = 0; for (int i = 0; i < 16; ++i) mk |= (uint32_t)(s.u8[i] >> 7) << i;");
            line(wr(ops[0], bits, "mk"));
            return true;
        case ZYDIS_MNEMONIC_MOVMSKPS:
            line("const VpXmm s = " + xmm_rd(ops[1]) + "; uint32_t mk = 0; for (int i = 0; i < 4; ++i) mk |= (s.u32[i] >> 31) << i;");
            line(wr(ops[0], bits, "mk"));
            return true;
        case ZYDIS_MNEMONIC_CMPSS: case ZYDIS_MNEMONIC_CMPSD: {
            if (insn->meta.category == ZYDIS_CATEGORY_STRINGOP) { string_scan(32, 2); return true; }
            const bool dbl = m == ZYDIS_MNEMONIC_CMPSD;
            line(fmt("const int t = vp_fcmp_pred(%u, %s.%s, %s);", (unsigned)ops[2].imm.value.u, xmm_dst().c_str(), dbl ? "f64[0]" : "f32[0]",
                     (dbl ? f64_rd(ops[1]) : f32_rd(ops[1])).c_str()));
            line(fmt("%s.u%d[0] = t ? %s : 0;", xmm_dst().c_str(), dbl ? 64 : 32, dbl ? "~UINT64_C(0)" : "0xffffffffu"));
            return true;
        }
        case ZYDIS_MNEMONIC_PSLLDQ: case ZYDIS_MNEMONIC_PSRLDQ: {
            const unsigned n = std::min<unsigned>(16, (unsigned)ops[1].imm.value.u);
            line("VpXmm a = " + xmm_dst() + ", r = {{0}};");
            if (m == ZYDIS_MNEMONIC_PSLLDQ) line(fmt("for (int i = %u; i < 16; ++i) r.u8[i] = a.u8[i - %u];", n, n));
            else line(fmt("for (int i = 0; i + %u < 16; ++i) r.u8[i] = a.u8[i + %u];", n, n));
            line(xmm_dst() + " = r;");
            return true;
        }

        // SSE4.1 / SSSE3 / SSE3 packed integer and float operations
        case ZYDIS_MNEMONIC_BLENDVPS: case ZYDIS_MNEMONIC_BLENDVPD: case ZYDIS_MNEMONIC_PBLENDVB: {
            const int lanes = m == ZYDIS_MNEMONIC_BLENDVPS ? 4 : m == ZYDIS_MNEMONIC_BLENDVPD ? 2 : 16;
            const char* t = lanes == 4 ? "u32" : lanes == 2 ? "u64" : "u8";
            const int top = lanes == 4 ? 31 : lanes == 2 ? 63 : 7;
            if (vex_mask >= 0) line(fmt("const VpXmm s = %s, mk = vmask;", xmm_rd(ops[1]).c_str()));
            else line(fmt("const VpXmm s = %s, mk = cpu->xmm[0];", xmm_rd(ops[1]).c_str()));
            line(fmt("for (int i = 0; i < %d; ++i) if ((mk.%s[i] >> %d) & 1) %s.%s[i] = s.%s[i];", lanes, t, top, xmm_dst().c_str(), t, t));
            return true;
        }
        case ZYDIS_MNEMONIC_BLENDPS: case ZYDIS_MNEMONIC_BLENDPD: {
            const int lanes = m == ZYDIS_MNEMONIC_BLENDPS ? 4 : 2;
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            line("const VpXmm s = " + xmm_rd(ops[1]) + ";");
            for (int i = 0; i < lanes; ++i) if ((imm >> i) & 1) line(fmt("%s.%s[%d] = s.%s[%d];", xmm_dst().c_str(), lanes == 4 ? "u32" : "u64", i, lanes == 4 ? "u32" : "u64", i));
            return true;
        }
        case ZYDIS_MNEMONIC_PINSRB: case ZYDIS_MNEMONIC_PINSRW: case ZYDIS_MNEMONIC_PINSRD: case ZYDIS_MNEMONIC_PINSRQ: {
            const int w = m == ZYDIS_MNEMONIC_PINSRB ? 8 : m == ZYDIS_MNEMONIC_PINSRW ? 16 : m == ZYDIS_MNEMONIC_PINSRD ? 32 : 64;
            const unsigned idx = (unsigned)ops[2].imm.value.u & (16 / (w / 8) - 1);
            const int src_bits = ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY ? w : (w < 32 ? 32 : w);
            line(fmt("%s.u%d[%u] = (uint%d_t)%s;", xmm_dst().c_str(), w, idx, w, rd(ops[1], src_bits).c_str()));
            return true;
        }
        case ZYDIS_MNEMONIC_PEXTRB: case ZYDIS_MNEMONIC_PEXTRW: case ZYDIS_MNEMONIC_PEXTRD: case ZYDIS_MNEMONIC_PEXTRQ: {
            const int w = m == ZYDIS_MNEMONIC_PEXTRB ? 8 : m == ZYDIS_MNEMONIC_PEXTRW ? 16 : m == ZYDIS_MNEMONIC_PEXTRD ? 32 : 64;
            const unsigned idx = (unsigned)ops[2].imm.value.u & (16 / (w / 8) - 1);
            const int dst_bits = ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? w : (w < 32 ? 32 : w);
            line(wr(ops[0], dst_bits, fmt("%s.u%d[%u]", reg_rd(ops[1].reg.value, 128).c_str(), w, idx)));
            return true;
        }
        case ZYDIS_MNEMONIC_PSHUFB:
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; VpXmm r;");
            line("for (int i = 0; i < 16; ++i) r.u8[i] = (s.u8[i] & 0x80) ? 0 : a.u8[s.u8[i] & 15];");
            line(xmm_dst() + " = r;");
            return true;
        case ZYDIS_MNEMONIC_PMULLD: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 4; ++i) " + xmm_dst() + ".u32[i] *= s.u32[i];"); return true;
        case ZYDIS_MNEMONIC_PMULUDQ: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 2; ++i) " + xmm_dst() + ".u64[i] = (uint64_t)" + xmm_dst() + ".u32[2 * i] * s.u32[2 * i];"); return true;
        case ZYDIS_MNEMONIC_PMULLW: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 8; ++i) " + xmm_dst() + ".u16[i] = (uint16_t)(" + xmm_dst() + ".u16[i] * s.u16[i]);"); return true;
        case ZYDIS_MNEMONIC_PMULHW: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 8; ++i) " + xmm_dst() + ".u16[i] = (uint16_t)(((int32_t)(int16_t)" + xmm_dst() + ".u16[i] * (int16_t)s.u16[i]) >> 16);"); return true;
        case ZYDIS_MNEMONIC_PMULHUW: line("const VpXmm s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 8; ++i) " + xmm_dst() + ".u16[i] = (uint16_t)(((uint32_t)" + xmm_dst() + ".u16[i] * s.u16[i]) >> 16);"); return true;
        case ZYDIS_MNEMONIC_PMADDWD: line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < 4; ++i) " + xmm_dst() + ".i32[i] = (int32_t)(int16_t)a.u16[2*i] * (int16_t)s.u16[2*i] + (int32_t)(int16_t)a.u16[2*i+1] * (int16_t)s.u16[2*i+1];"); return true;
        case ZYDIS_MNEMONIC_ROUNDSS: case ZYDIS_MNEMONIC_ROUNDSD: case ZYDIS_MNEMONIC_ROUNDPS: case ZYDIS_MNEMONIC_ROUNDPD: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            // imm bit 2: MXCSR's mode (the host's, set by vp_apply_mxcsr); else the mode in bits 0-1.
            const char* fn = (imm & 4) ? "nearbyint" : (imm & 3) == 0 ? "__builtin_roundeven" : (imm & 3) == 1 ? "floor" : (imm & 3) == 2 ? "ceil" : "trunc";
            if (m == ZYDIS_MNEMONIC_ROUNDSS) line(fmt("%s.f32[0] = (float)%s(%s);", xmm_dst().c_str(), fn, f32_rd(ops[1]).c_str()));
            else if (m == ZYDIS_MNEMONIC_ROUNDSD) line(fmt("%s.f64[0] = %s(%s);", xmm_dst().c_str(), fn, f64_rd(ops[1]).c_str()));
            else if (m == ZYDIS_MNEMONIC_ROUNDPD) line(fmt("const VpXmm s = %s; for (int i = 0; i < 2; ++i) %s.f64[i] = %s(s.f64[i]);", xmm_rd(ops[1]).c_str(), xmm_dst().c_str(), fn));
            else line(fmt("const VpXmm s = %s; for (int i = 0; i < 4; ++i) %s.f32[i] = (float)%s(s.f32[i]);", xmm_rd(ops[1]).c_str(), xmm_dst().c_str(), fn));
            return true;
        }
#define VP_LANEOP(MN, LANES, T, EXPR) \
        case ZYDIS_MNEMONIC_##MN: line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; for (int i = 0; i < " #LANES "; ++i) " + xmm_dst() + "." T "[i] = " EXPR ";"); return true;
        VP_LANEOP(PMINSD, 4, "i32", "(a.i32[i] < s.i32[i]) ? a.i32[i] : s.i32[i]")
        VP_LANEOP(PMAXSD, 4, "i32", "(a.i32[i] > s.i32[i]) ? a.i32[i] : s.i32[i]")
        VP_LANEOP(PMINUD, 4, "u32", "(a.u32[i] < s.u32[i]) ? a.u32[i] : s.u32[i]")
        VP_LANEOP(PMAXUD, 4, "u32", "(a.u32[i] > s.u32[i]) ? a.u32[i] : s.u32[i]")
        VP_LANEOP(PMINSW, 8, "u16", "((int16_t)a.u16[i] < (int16_t)s.u16[i]) ? a.u16[i] : s.u16[i]")
        VP_LANEOP(PMAXSW, 8, "u16", "((int16_t)a.u16[i] > (int16_t)s.u16[i]) ? a.u16[i] : s.u16[i]")
        VP_LANEOP(PMINUB, 16, "u8", "(a.u8[i] < s.u8[i]) ? a.u8[i] : s.u8[i]")
        VP_LANEOP(PMAXUB, 16, "u8", "(a.u8[i] > s.u8[i]) ? a.u8[i] : s.u8[i]")
        VP_LANEOP(PCMPGTD, 4, "u32", "(a.i32[i] > s.i32[i]) ? 0xffffffffu : 0")
        VP_LANEOP(PCMPGTW, 8, "u16", "((int16_t)a.u16[i] > (int16_t)s.u16[i]) ? 0xffffu : 0")
        VP_LANEOP(PCMPGTB, 16, "u8", "((int8_t)a.u8[i] > (int8_t)s.u8[i]) ? 0xffu : 0")
        VP_LANEOP(PCMPEQW, 8, "u16", "(a.u16[i] == s.u16[i]) ? 0xffffu : 0")
        VP_LANEOP(PCMPEQQ, 2, "u64", "(a.u64[i] == s.u64[i]) ? ~UINT64_C(0) : 0")
        VP_LANEOP(PADDB, 16, "u8", "(uint8_t)(a.u8[i] + s.u8[i])")
        VP_LANEOP(PSUBB, 16, "u8", "(uint8_t)(a.u8[i] - s.u8[i])")
        VP_LANEOP(PADDW, 8, "u16", "(uint16_t)(a.u16[i] + s.u16[i])")
        VP_LANEOP(PSUBW, 8, "u16", "(uint16_t)(a.u16[i] - s.u16[i])")
        VP_LANEOP(PAVGB, 16, "u8", "(uint8_t)((a.u8[i] + s.u8[i] + 1) >> 1)")
        VP_LANEOP(PAVGW, 8, "u16", "(uint16_t)((a.u16[i] + s.u16[i] + 1) >> 1)")
        VP_LANEOP(MINPS, 4, "f32", "vp_minss(a.f32[i], s.f32[i])")
        VP_LANEOP(MAXPS, 4, "f32", "vp_maxss(a.f32[i], s.f32[i])")
        VP_LANEOP(MINPD, 2, "f64", "vp_minsd(a.f64[i], s.f64[i])")
        VP_LANEOP(MAXPD, 2, "f64", "vp_maxsd(a.f64[i], s.f64[i])")
        VP_LANEOP(SQRTPD, 2, "f64", "sqrt(s.f64[i])")
        VP_LANEOP(RCPPS, 4, "f32", "1.0f / s.f32[i]")
        VP_LANEOP(RSQRTPS, 4, "f32", "1.0f / sqrtf(s.f32[i])")
        VP_LANEOP(PABSD, 4, "u32", "(uint32_t)(s.i32[i] < 0 ? -(uint32_t)s.i32[i] : (uint32_t)s.i32[i])")
        VP_LANEOP(PABSW, 8, "u16", "(uint16_t)((int16_t)s.u16[i] < 0 ? -(uint16_t)s.u16[i] : s.u16[i])")
        VP_LANEOP(PABSB, 16, "u8", "(uint8_t)((int8_t)s.u8[i] < 0 ? -(uint8_t)s.u8[i] : s.u8[i])")
        VP_LANEOP(PSIGND, 4, "i32", "s.i32[i] < 0 ? -a.i32[i] : s.i32[i] == 0 ? 0 : a.i32[i]")
        VP_LANEOP(CVTPS2DQ, 4, "i32", "vp_cvt_f32_i32(s.f32[i])")
        VP_LANEOP(PUNPCKLBW, 16, "u8", "(i & 1) ? s.u8[i / 2] : a.u8[i / 2]")
        VP_LANEOP(PUNPCKHBW, 16, "u8", "(i & 1) ? s.u8[8 + i / 2] : a.u8[8 + i / 2]")
        VP_LANEOP(PUNPCKLWD, 8, "u16", "(i & 1) ? s.u16[i / 2] : a.u16[i / 2]")
        VP_LANEOP(PUNPCKHWD, 8, "u16", "(i & 1) ? s.u16[4 + i / 2] : a.u16[4 + i / 2]")
        VP_LANEOP(PUNPCKLDQ, 4, "u32", "(i & 1) ? s.u32[i / 2] : a.u32[i / 2]")
        VP_LANEOP(PUNPCKHDQ, 4, "u32", "(i & 1) ? s.u32[2 + i / 2] : a.u32[2 + i / 2]")
        VP_LANEOP(MOVSHDUP, 4, "u32", "s.u32[i | 1]")
        VP_LANEOP(MOVSLDUP, 4, "u32", "s.u32[i & ~1]")
        VP_LANEOP(MOVDDUP, 2, "u64", "s.u64[0]")
        VP_LANEOP(PMULHRSW, 8, "u16", "(uint16_t)((((int32_t)(int16_t)a.u16[i] * (int16_t)s.u16[i] >> 14) + 1) >> 1)")
        VP_LANEOP(PMADDUBSW, 8, "u16", "(uint16_t)vp_sat16((int32_t)a.u8[2*i] * (int8_t)s.u8[2*i] + (int32_t)a.u8[2*i+1] * (int8_t)s.u8[2*i+1])")
        VP_LANEOP(PHADDW, 8, "u16", "i < 4 ? (uint16_t)(a.u16[2*i] + a.u16[2*i+1]) : (uint16_t)(s.u16[2*(i-4)] + s.u16[2*(i-4)+1])")
        VP_LANEOP(PHSUBW, 8, "u16", "i < 4 ? (uint16_t)(a.u16[2*i] - a.u16[2*i+1]) : (uint16_t)(s.u16[2*(i-4)] - s.u16[2*(i-4)+1])")
        VP_LANEOP(PHADDSW, 8, "u16", "(uint16_t)vp_sat16(i < 4 ? (int16_t)a.u16[2*i] + (int16_t)a.u16[2*i+1] : (int16_t)s.u16[2*(i-4)] + (int16_t)s.u16[2*(i-4)+1])")
        VP_LANEOP(PHSUBSW, 8, "u16", "(uint16_t)vp_sat16(i < 4 ? (int16_t)a.u16[2*i] - (int16_t)a.u16[2*i+1] : (int16_t)s.u16[2*(i-4)] - (int16_t)s.u16[2*(i-4)+1])")
        VP_LANEOP(PHSUBD, 4, "u32", "i < 2 ? a.u32[2*i] - a.u32[2*i+1] : s.u32[2*(i-2)] - s.u32[2*(i-2)+1]")
        VP_LANEOP(PSIGNB, 16, "u8", "(int8_t)s.u8[i] < 0 ? (uint8_t)-a.u8[i] : s.u8[i] == 0 ? 0 : a.u8[i]")
        VP_LANEOP(PSIGNW, 8, "u16", "(int16_t)s.u16[i] < 0 ? (uint16_t)-a.u16[i] : s.u16[i] == 0 ? 0 : a.u16[i]")
        VP_LANEOP(PMULDQ, 2, "i64", "(int64_t)a.i32[2*i] * (int64_t)s.i32[2*i]")
        VP_LANEOP(PCMPGTQ, 2, "u64", "a.i64[i] > s.i64[i] ? ~UINT64_C(0) : 0")
        VP_LANEOP(PMINSB, 16, "u8", "(int8_t)a.u8[i] < (int8_t)s.u8[i] ? a.u8[i] : s.u8[i]")
        VP_LANEOP(PMAXSB, 16, "u8", "(int8_t)a.u8[i] > (int8_t)s.u8[i] ? a.u8[i] : s.u8[i]")
        VP_LANEOP(PMINUW, 8, "u16", "a.u16[i] < s.u16[i] ? a.u16[i] : s.u16[i]")
        VP_LANEOP(PMAXUW, 8, "u16", "a.u16[i] > s.u16[i] ? a.u16[i] : s.u16[i]")
        VP_LANEOP(HADDPD, 2, "f64", "i == 0 ? a.f64[0] + a.f64[1] : s.f64[0] + s.f64[1]")
        VP_LANEOP(HSUBPS, 4, "f32", "i < 2 ? a.f32[2*i] - a.f32[2*i+1] : s.f32[2*(i-2)] - s.f32[2*(i-2)+1]")
        VP_LANEOP(HSUBPD, 2, "f64", "i == 0 ? a.f64[0] - a.f64[1] : s.f64[0] - s.f64[1]")
        VP_LANEOP(ADDSUBPS, 4, "f32", "(i & 1) ? a.f32[i] + s.f32[i] : a.f32[i] - s.f32[i]")
        VP_LANEOP(ADDSUBPD, 2, "f64", "(i & 1) ? a.f64[i] + s.f64[i] : a.f64[i] - s.f64[i]")
        VP_LANEOP(HADDPS, 4, "f32", "i < 2 ? a.f32[2*i] + a.f32[2*i+1] : s.f32[2*(i-2)] + s.f32[2*(i-2)+1]")
        VP_LANEOP(PHADDD, 4, "i32", "i < 2 ? a.i32[2*i] + a.i32[2*i+1] : s.i32[2*(i-2)] + s.i32[2*(i-2)+1]")
        VP_LANEOP(CVTDQ2PD, 2, "f64", "(double)s.i32[i]")
        VP_LANEOP(CVTPS2PD, 2, "f64", "(double)s.f32[i]")
        VP_LANEOP(CVTTPD2DQ, 4, "i32", "i < 2 ? vp_cvtt_f64_i32(s.f64[i]) : 0")
        VP_LANEOP(CVTPD2PS, 4, "f32", "i < 2 ? (float)s.f64[i] : 0.0f")
        VP_LANEOP(PMOVZXBW, 8, "u16", "s.u8[i]")
        VP_LANEOP(PMOVZXBD, 4, "u32", "s.u8[i]")
        VP_LANEOP(PMOVZXWD, 4, "u32", "s.u16[i]")
        VP_LANEOP(PMOVZXDQ, 2, "u64", "s.u32[i]")
        VP_LANEOP(PMOVSXBW, 8, "u16", "(uint16_t)(int16_t)(int8_t)s.u8[i]")
        VP_LANEOP(PMOVSXBD, 4, "u32", "(uint32_t)(int32_t)(int8_t)s.u8[i]")
        VP_LANEOP(PMOVSXWD, 4, "u32", "(uint32_t)(int32_t)(int16_t)s.u16[i]")
        VP_LANEOP(PMOVSXDQ, 2, "u64", "(uint64_t)(int64_t)s.i32[i]")
#undef VP_LANEOP
        case ZYDIS_MNEMONIC_RSQRTSS: line(fmt("%s.f32[0] = 1.0f / sqrtf(%s);", xmm_dst().c_str(), f32_rd(ops[1]).c_str())); return true;
        case ZYDIS_MNEMONIC_RCPSS: line(fmt("%s.f32[0] = 1.0f / %s;", xmm_dst().c_str(), f32_rd(ops[1]).c_str())); return true;
        case ZYDIS_MNEMONIC_PSLLD: case ZYDIS_MNEMONIC_PSRLD: case ZYDIS_MNEMONIC_PSRAD: case ZYDIS_MNEMONIC_PSLLQ: case ZYDIS_MNEMONIC_PSRLQ:
        case ZYDIS_MNEMONIC_PSLLW: case ZYDIS_MNEMONIC_PSRLW: case ZYDIS_MNEMONIC_PSRAW: {
            const int w = (m == ZYDIS_MNEMONIC_PSLLQ || m == ZYDIS_MNEMONIC_PSRLQ) ? 64 : (m == ZYDIS_MNEMONIC_PSLLW || m == ZYDIS_MNEMONIC_PSRLW || m == ZYDIS_MNEMONIC_PSRAW) ? 16 : 32;
            const bool left = m == ZYDIS_MNEMONIC_PSLLD || m == ZYDIS_MNEMONIC_PSLLQ || m == ZYDIS_MNEMONIC_PSLLW;
            const bool arith = m == ZYDIS_MNEMONIC_PSRAD || m == ZYDIS_MNEMONIC_PSRAW;
            if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) line(fmt("const uint64_t n = %" PRIu64 ";", ops[1].imm.value.u));
            else line("const uint64_t n = " + xmm_rd(ops[1]) + ".u64[0];");
            const int lanes = 128 / w;
            if (arith) line(fmt("for (int i = 0; i < %d; ++i) %s.u%d[i] = (uint%d_t)((int%d_t)%s.u%d[i] >> (n > %d ? %d : n));", lanes, xmm_dst().c_str(), w, w, w, xmm_dst().c_str(), w, w - 1, w - 1));
            else line(fmt("for (int i = 0; i < %d; ++i) %s.u%d[i] = (n > %d) ? 0 : (uint%d_t)(%s.u%d[i] %s n);", lanes, xmm_dst().c_str(), w, w - 1, w, xmm_dst().c_str(), w, left ? "<<" : ">>"));
            return true;
        }
        case ZYDIS_MNEMONIC_PTEST:
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + ";");
            line("VP_FC->zf = ((a.u64[0] & s.u64[0]) | (a.u64[1] & s.u64[1])) == 0; VP_FC->cf = ((~a.u64[0] & s.u64[0]) | (~a.u64[1] & s.u64[1])) == 0; VP_FC->of = VP_FC->sf = VP_FC->pf = VP_FC->af = 0;");
            return true;
        case ZYDIS_MNEMONIC_PALIGNR: {
            const unsigned n = (unsigned)ops[2].imm.value.u;
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; uint8_t cat[32]; memcpy(cat, &s, 16); memcpy(cat + 16, &a, 16); VpXmm r = {{0}};");
            line(fmt("for (int i = 0; i < 16; ++i) if (i + %u < 32) r.u8[i] = cat[i + %u];", n, n));
            line(xmm_dst() + " = r;");
            return true;
        }
        case ZYDIS_MNEMONIC_SHUFPD: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            line("const VpXmm a = " + xmm_dst() + ", b = " + xmm_rd(ops[1]) + ";");
            line(fmt("%s.u64[0] = a.u64[%u]; %s.u64[1] = b.u64[%u];", xmm_dst().c_str(), imm & 1, xmm_dst().c_str(), (imm >> 1) & 1));
            return true;
        }
        case ZYDIS_MNEMONIC_PSHUFLW: case ZYDIS_MNEMONIC_PSHUFHW: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            const int base = m == ZYDIS_MNEMONIC_PSHUFLW ? 0 : 4;
            line("const VpXmm b = " + xmm_rd(ops[1]) + "; VpXmm r = b;");
            for (int i = 0; i < 4; ++i) line(fmt("r.u16[%d] = b.u16[%d];", base + i, base + ((imm >> (2 * i)) & 3)));
            line(xmm_dst() + " = r;");
            return true;
        }
        case ZYDIS_MNEMONIC_INSERTPS: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            const unsigned src_lane = ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY ? 0 : (imm >> 6) & 3, dst_lane = (imm >> 4) & 3;
            if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) line(fmt("const uint32_t v = vp_ld32(%s);", ea(ops[1]).c_str()));
            else line(fmt("const uint32_t v = %s.u32[%u];", xmm_rd(ops[1]).c_str(), src_lane));
            line(fmt("%s.u32[%u] = v;", xmm_dst().c_str(), dst_lane));
            for (int i = 0; i < 4; ++i) if ((imm >> i) & 1) line(fmt("%s.u32[%d] = 0;", xmm_dst().c_str(), i));
            return true;
        }
        case ZYDIS_MNEMONIC_EXTRACTPS:
            line(wr(ops[0], ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? 32 : bits, fmt("%s.u32[%u]", reg_rd(ops[1].reg.value, 128).c_str(), (unsigned)ops[2].imm.value.u & 3)));
            return true;
        case ZYDIS_MNEMONIC_DPPS: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            // As the hardware adds: the masked products (others +0), then (p0 + p1) + (p2 + p3).
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; float p[4];");
            for (int i = 0; i < 4; ++i) line(((imm >> (4 + i)) & 1) ? fmt("p[%d] = a.f32[%d] * s.f32[%d];", i, i, i) : fmt("p[%d] = 0.0f;", i));
            line("const float d = (p[0] + p[1]) + (p[2] + p[3]);");
            for (int i = 0; i < 4; ++i) line(fmt("%s.f32[%d] = %s;", xmm_dst().c_str(), i, ((imm >> i) & 1) ? "d" : "0.0f"));
            return true;
        }
        case ZYDIS_MNEMONIC_CMPPS: case ZYDIS_MNEMONIC_CMPPD: {
            const bool dbl = m == ZYDIS_MNEMONIC_CMPPD;
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + ";");
            if (dbl) line(fmt("for (int i = 0; i < 2; ++i) %s.u64[i] = vp_fcmp_pred(%u, a.f64[i], s.f64[i]) ? ~UINT64_C(0) : 0;", xmm_dst().c_str(), (unsigned)ops[2].imm.value.u));
            else line(fmt("for (int i = 0; i < 4; ++i) %s.u32[i] = vp_fcmp_pred(%u, a.f32[i], s.f32[i]) ? 0xffffffffu : 0;", xmm_dst().c_str(), (unsigned)ops[2].imm.value.u));
            return true;
        }
        case ZYDIS_MNEMONIC_PACKSSDW: case ZYDIS_MNEMONIC_PACKUSDW: case ZYDIS_MNEMONIC_PACKSSWB: case ZYDIS_MNEMONIC_PACKUSWB: {
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; VpXmm r;");
            if (m == ZYDIS_MNEMONIC_PACKSSDW) line("for (int i = 0; i < 8; ++i) { int32_t v = i < 4 ? a.i32[i] : s.i32[i - 4]; r.u16[i] = (uint16_t)(v < -32768 ? -32768 : v > 32767 ? 32767 : v); }");
            else if (m == ZYDIS_MNEMONIC_PACKUSDW) line("for (int i = 0; i < 8; ++i) { int32_t v = i < 4 ? a.i32[i] : s.i32[i - 4]; r.u16[i] = (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : v); }");
            else if (m == ZYDIS_MNEMONIC_PACKSSWB) line("for (int i = 0; i < 16; ++i) { int16_t v = (int16_t)(i < 8 ? a.u16[i] : s.u16[i - 8]); r.u8[i] = (uint8_t)(v < -128 ? -128 : v > 127 ? 127 : v); }");
            else line("for (int i = 0; i < 16; ++i) { int16_t v = (int16_t)(i < 8 ? a.u16[i] : s.u16[i - 8]); r.u8[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }");
            line(xmm_dst() + " = r;");
            return true;
        }
        case ZYDIS_MNEMONIC_PBLENDW: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            line("const VpXmm s = " + xmm_rd(ops[1]) + ";");
            for (int i = 0; i < 8; ++i) if ((imm >> i) & 1) line(fmt("%s.u16[%d] = s.u16[%d];", xmm_dst().c_str(), i, i));
            return true;
        }
        case ZYDIS_MNEMONIC_PHMINPOSUW:
            line("const VpXmm s = " + xmm_rd(ops[1]) + "; VpXmm r = {{0}}; r.u16[0] = s.u16[0];");
            line("for (int i = 1; i < 8; ++i) if (s.u16[i] < r.u16[0]) { r.u16[0] = s.u16[i]; r.u16[1] = (uint16_t)i; }");
            line(xmm_dst() + " = r;");
            return true;
        case ZYDIS_MNEMONIC_MPSADBW: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; VpXmm r;");
            line(fmt("for (int i = 0; i < 8; ++i) { unsigned t = 0; for (int k = 0; k < 4; ++k) { const int d = (int)a.u8[%u + i + k] - (int)s.u8[%u + k]; t += (unsigned)(d < 0 ? -d : d); } r.u16[i] = (uint16_t)t; }",
                     ((imm >> 2) & 1) * 4, (imm & 3) * 4));
            line(xmm_dst() + " = r;");
            return true;
        }
        case ZYDIS_MNEMONIC_AESENC: case ZYDIS_MNEMONIC_AESENCLAST:
            line(xmm_dst() + fmt(" = vp_aesenc(%s, %s, %d);", xmm_dst().c_str(), xmm_rd(ops[1]).c_str(), m == ZYDIS_MNEMONIC_AESENCLAST));
            return true;
        case ZYDIS_MNEMONIC_AESDEC: case ZYDIS_MNEMONIC_AESDECLAST:
            line(xmm_dst() + fmt(" = vp_aesdec(%s, %s, %d);", xmm_dst().c_str(), xmm_rd(ops[1]).c_str(), m == ZYDIS_MNEMONIC_AESDECLAST));
            return true;
        case ZYDIS_MNEMONIC_AESIMC: line(xmm_dst() + " = vp_aesimc(" + xmm_rd(ops[1]) + ");"); return true;
        case ZYDIS_MNEMONIC_AESKEYGENASSIST:
            line(xmm_dst() + fmt(" = vp_aeskeygenassist(%s, %uu);", xmm_rd(ops[1]).c_str(), (unsigned)(ops[2].imm.value.u & 0xff)));
            return true;
        case ZYDIS_MNEMONIC_PCLMULQDQ: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            line(xmm_dst() + fmt(" = vp_clmul(%s.u64[%u], %s.u64[%u]);", xmm_dst().c_str(), imm & 1, xmm_rd(ops[1]).c_str(), (imm >> 4) & 1));
            return true;
        }
        case ZYDIS_MNEMONIC_PCMPESTRI: case ZYDIS_MNEMONIC_PCMPESTRM: case ZYDIS_MNEMONIC_PCMPISTRI: case ZYDIS_MNEMONIC_PCMPISTRM: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            const bool expl = m == ZYDIS_MNEMONIC_PCMPESTRI || m == ZYDIS_MNEMONIC_PCMPESTRM;
            const bool index = m == ZYDIS_MNEMONIC_PCMPESTRI || m == ZYDIS_MNEMONIC_PCMPISTRI;
            const int n = (imm & 1) ? 8 : 16;
            if (expl) {
                // |rax| / |rdx| (eax/edx without REX.W), saturated to the element count.
                const bool w = insn->operand_width == 64;
                line(w ? "const int64_t la0 = (int64_t)VP_R64(VP_RAX), lb0 = (int64_t)VP_R64(VP_RDX);"
                       : "const int64_t la0 = (int32_t)VP_R32(VP_RAX), lb0 = (int32_t)VP_R32(VP_RDX);");
                line(fmt("const uint64_t la1 = la0 < 0 ? 0 - (uint64_t)la0 : (uint64_t)la0, lb1 = lb0 < 0 ? 0 - (uint64_t)lb0 : (uint64_t)lb0;"));
                line(fmt("const int64_t la = la1 > %d ? %d : (int64_t)la1, lb = lb1 > %d ? %d : (int64_t)lb1;", n, n, n, n));
            } else {
                line("const int64_t la = -1, lb = -1;");
            }
            line(fmt("const uint32_t res = vp_pcmpstr(%s, %s, la, lb, %uu, VP_FC);", reg_rd(ops[0].reg.value, 128).c_str(), xmm_rd(ops[1]).c_str(), imm));
            if (index) line(fmt("VP_W32(VP_RCX, vp_pcmpstr_index(res, %uu));", imm));
            else line(fmt("cpu->xmm[0] = vp_pcmpstr_mask(res, %uu);%s", imm, (insn->attributes & ZYDIS_ATTRIB_HAS_VEX) ? " cpu->ymmh[0] = (VpXmm){{0}};" : ""));
            return true;
        }
        case ZYDIS_MNEMONIC_CRC32: {
            const int sb = ops[1].size;
            line(fmt("const uint32_t c0 = VP_R32(%d);", gpr_index(ops[0].reg.value)));
            line(wr(ops[0], 32, fmt("vp_crc32c(c0, %s, %d)", rd(ops[1], sb).c_str(), sb / 8)));
            return true;
        }
        case ZYDIS_MNEMONIC_PUSHFQ:
            line("VP_PUSH((uint64_t)(0x202u | VP_FC->cf | VP_FC->pf << 2 | VP_FC->af << 4 | VP_FC->zf << 6 | VP_FC->sf << 7 | VP_FC->df << 10 | VP_FC->of << 11 | cpu->rflags_ac_id));");
            return true;
        case ZYDIS_MNEMONIC_POPFQ:
            line("{ const uint64_t f = VP_POP(); VP_FC->cf = f & 1; VP_FC->pf = (f >> 2) & 1; VP_FC->af = (f >> 4) & 1; VP_FC->zf = (f >> 6) & 1;"
                 " VP_FC->sf = (f >> 7) & 1; VP_FC->df = (f >> 10) & 1; VP_FC->of = (f >> 11) & 1; cpu->rflags_ac_id = (uint32_t)(f & 0x240000u); }");
            return true;
        case ZYDIS_MNEMONIC_LAHF:
            line("VP_W8H(VP_RAX, (uint8_t)(VP_FC->sf << 7 | VP_FC->zf << 6 | VP_FC->af << 4 | VP_FC->pf << 2 | 2 | VP_FC->cf));");
            return true;
        case ZYDIS_MNEMONIC_SAHF:
            line("{ const uint8_t h = VP_R8H(VP_RAX); VP_FC->sf = (h >> 7) & 1; VP_FC->zf = (h >> 6) & 1; VP_FC->af = (h >> 4) & 1; VP_FC->pf = (h >> 2) & 1; VP_FC->cf = h & 1; }");
            return true;
        case ZYDIS_MNEMONIC_XLAT:
            line("VP_W8(VP_RAX, vp_ld8(VP_R64(VP_RBX) + VP_R8(VP_RAX)));");
            return true;
        case ZYDIS_MNEMONIC_BEXTR: { // BMI1: start = src2[7:0], length = src2[15:8]
            line(fmt("const uint64_t v = %s, c = %s; const unsigned st = (unsigned)(c & 0xff), ln = (unsigned)((c >> 8) & 0xff);", rd(ops[1], bits).c_str(), rd(ops[2], bits).c_str()));
            line(fmt("uint64_t r = st >= %d ? 0 : v >> st; if (ln < 64) r &= (UINT64_C(1) << ln) - 1; r &= VP_MASK(%d);", bits, bits));
            // AF, SF, PF are undefined; AMD (the PS4's vendor) leaves PF as the parity of the result.
            line("VP_FC->zf = r == 0; VP_FC->cf = VP_FC->of = 0; VP_FC->sf = 0; VP_FC->pf = vp_parity(r); VP_FC->af = 0;");
            line(wr(ops[0], bits, "r"));
            return true;
        }
        case ZYDIS_MNEMONIC_MOVNTI: line(wr(ops[0], bits, rd(ops[1], bits))); return true;
        // SSE4a (AMD, Jaguar has it): bit-field extract / insert on the low quadword.
        case ZYDIS_MNEMONIC_EXTRQ:
            if (nops == 3) line(fmt("const unsigned ln = %uu, ix = %uu;", (unsigned)(ops[1].imm.value.u & 63), (unsigned)(ops[2].imm.value.u & 63)));
            else line("const VpXmm c = " + xmm_rd(ops[1]) + "; const unsigned ln = c.u8[0] & 63, ix = c.u8[1] & 63;");
            line("{ const uint64_t mk = ln ? (UINT64_C(1) << ln) - 1 : ~UINT64_C(0); " + xmm_dst() + ".u64[0] = (" + xmm_dst() + ".u64[0] >> ix) & mk; " + xmm_dst() + ".u64[1] = 0; }"); // AMD zeroes the upper half
            return true;
        case ZYDIS_MNEMONIC_INSERTQ:
            line("const VpXmm s = " + xmm_rd(ops[1]) + ";");
            if (nops == 4) line(fmt("const unsigned ln = %uu, ix = %uu;", (unsigned)(ops[2].imm.value.u & 63), (unsigned)(ops[3].imm.value.u & 63)));
            else line("const unsigned ln = s.u8[8] & 63, ix = s.u8[9] & 63;");
            line("{ const uint64_t mk = (ln ? (UINT64_C(1) << ln) - 1 : ~UINT64_C(0)) << ix; " + xmm_dst() + ".u64[0] = (" + xmm_dst() +
                 ".u64[0] & ~mk) | ((s.u64[0] << ix) & mk); " + xmm_dst() + ".u64[1] = 0; }"); // AMD zeroes the upper half
            return true;
        case ZYDIS_MNEMONIC_MOVNTSS: line("vp_st32(" + ea(ops[0]) + ", " + xmm_rd(ops[1]) + ".u32[0]);"); return true;
        case ZYDIS_MNEMONIC_MOVNTSD: line("vp_st64(" + ea(ops[0]) + ", " + xmm_rd(ops[1]) + ".u64[0]);"); return true;
        case ZYDIS_MNEMONIC_CLFLUSH: case ZYDIS_MNEMONIC_CLFLUSHOPT: case ZYDIS_MNEMONIC_PREFETCH: return true;
        case ZYDIS_MNEMONIC_PSADBW:
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; VpXmm r = {{0}};");
            line("for (int i = 0; i < 16; ++i) r.u64[i / 8] += (uint64_t)(a.u8[i] > s.u8[i] ? a.u8[i] - s.u8[i] : s.u8[i] - a.u8[i]);");
            line(xmm_dst() + " = r;");
            return true;

        // Atomics and double shifts
        case ZYDIS_MNEMONIC_XADD: {
            const bool lock = insn->attributes & ZYDIS_ATTRIB_HAS_LOCK;
            if (lock && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
                line("const uint64_t ea = " + ea(ops[0]) + ";");
                line(fmt("const uint64_t y = %s;", rd(ops[1], bits).c_str()));
                line(fmt("const uint64_t x = vp_fetch_add%d(ea, (uint%d_t)y);", bits, bits));
                line(fmt("vp_flags_add(VP_FC, %d, x, y, (x + y) & VP_MASK(%d));", bits, bits));
                line(wr(ops[1], bits, "x"));
            } else {
                const std::string a = bind_addr(ops[0]);
                line(fmt("const uint64_t x = %s, y = %s;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str()));
                line(fmt("const uint64_t r = (x + y) & VP_MASK(%d);", bits));
                line(fmt("vp_flags_add(VP_FC, %d, x, y, r);", bits));
                line(wr(ops[1], bits, "x"));
                line(wr(ops[0], bits, "r", a));
            }
            return true;
        }
        case ZYDIS_MNEMONIC_CMPXCHG: {
            const bool lock = insn->attributes & ZYDIS_ATTRIB_HAS_LOCK;
            line(fmt("uint64_t expected = VP_R%d(VP_RAX); const uint64_t desired = %s;", bits, rd(ops[1], bits).c_str()));
            if (lock && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
                line("const uint64_t ea = " + ea(ops[0]) + ";");
                line(fmt("uint%d_t exp = (uint%d_t)expected;", bits, bits));
                line(fmt("const int ok = vp_cas%d(ea, &exp, (uint%d_t)desired);", bits, bits));
                line(fmt("vp_flags_sub(VP_FC, %d, expected, exp, (expected - exp) & VP_MASK(%d));", bits, bits));
                line(fmt("if (!ok) VP_W%d(VP_RAX, (uint%d_t)exp);", bits, bits));
            } else {
                const std::string a = bind_addr(ops[0]);
                line(fmt("const uint64_t cur = %s;", rd(ops[0], bits, a).c_str()));
                line(fmt("vp_flags_sub(VP_FC, %d, expected, cur, (expected - cur) & VP_MASK(%d));", bits, bits));
                line("if (cur == expected) { " + wr(ops[0], bits, "desired", a) + " } else { " + fmt("VP_W%d(VP_RAX, (uint%d_t)cur);", bits, bits) + " }");
            }
            return true;
        }
        case ZYDIS_MNEMONIC_CMPXCHG16B: {
            line("const uint64_t ea = " + ea(ops[0]) + ";");
            line("unsigned __int128 expected = ((unsigned __int128)VP_R64(VP_RDX) << 64) | VP_R64(VP_RAX);");
            line("const unsigned __int128 desired = ((unsigned __int128)VP_R64(VP_RCX) << 64) | VP_R64(VP_RBX);");
            line("const int ok = __atomic_compare_exchange_n((unsigned __int128*)(uintptr_t)ea, &expected, desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);");
            line("VP_FC->zf = (uint8_t)ok; if (!ok) { VP_W64(VP_RAX, (uint64_t)expected); VP_W64(VP_RDX, (uint64_t)(expected >> 64)); }");
            return true;
        }
        case ZYDIS_MNEMONIC_SHLD: case ZYDIS_MNEMONIC_SHRD: {
            const bool left = m == ZYDIS_MNEMONIC_SHLD;
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s & VP_MASK(%d), y = %s & VP_MASK(%d);", rd(ops[0], bits, a).c_str(), bits, rd(ops[1], bits).c_str(), bits));
            line(fmt("const unsigned n = (unsigned)(%s) & %u;", rd(ops[2], 8).c_str(), bits == 64 ? 63u : 31u));
            line("if (n) {");
            if (left) line(fmt("    const uint64_t r = ((x << n) | (y >> (%d - n))) & VP_MASK(%d);", bits, bits));
            else line(fmt("    const uint64_t r = ((x >> n) | (y << (%d - n))) & VP_MASK(%d);", bits, bits));
            line(fmt("    vp_flags_%s(VP_FC, %d, x, n, r);", left ? "shl" : "shr", bits));
            line("    " + wr(ops[0], bits, "r", a));
            line("}");
            if (bits == 32) line("else { " + wr(ops[0], 32, "x", a) + " }");
            return true;
        }
        case ZYDIS_MNEMONIC_MOVMSKPD:
            line("const VpXmm s = " + xmm_rd(ops[1]) + "; const uint32_t mk = (uint32_t)((s.u64[0] >> 63) | ((s.u64[1] >> 63) << 1));");
            line(wr(ops[0], bits, "mk"));
            return true;
        case ZYDIS_MNEMONIC_MOVBE:
            if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) line(wr(ops[0], bits, fmt(bits == 64 ? "__builtin_bswap64(%s)" : bits == 32 ? "__builtin_bswap32((uint32_t)%s)" : "__builtin_bswap16((uint16_t)%s)", rd(ops[1], bits).c_str())));
            else line(wr(ops[0], bits, fmt(bits == 64 ? "__builtin_bswap64(%s)" : bits == 32 ? "__builtin_bswap32((uint32_t)%s)" : "__builtin_bswap16((uint16_t)%s)", rd(ops[1], bits).c_str())));
            return true;
        case ZYDIS_MNEMONIC_ANDN: // BMI1: dst = ~src1 & src2
            line(fmt("const uint64_t r = (~%s & %s) & VP_MASK(%d);", rd(ops[1], bits).c_str(), rd(ops[2], bits).c_str(), bits));
            line(fmt("vp_flags_logic(VP_FC, %d, r);", bits));
            line(wr(ops[0], bits, "r"));
            return true;
        case ZYDIS_MNEMONIC_BLSR: case ZYDIS_MNEMONIC_BLSI: case ZYDIS_MNEMONIC_BLSMSK: {
            line(fmt("const uint64_t x = %s & VP_MASK(%d);", rd(ops[1], bits).c_str(), bits));
            if (m == ZYDIS_MNEMONIC_BLSR) line(fmt("const uint64_t r = (x & (x - 1)) & VP_MASK(%d);", bits));
            else if (m == ZYDIS_MNEMONIC_BLSI) line(fmt("const uint64_t r = (x & (0 - x)) & VP_MASK(%d);", bits));
            else line(fmt("const uint64_t r = (x ^ (x - 1)) & VP_MASK(%d);", bits));
            line(fmt("vp_flags_result(VP_FC, %d, r); VP_FC->cf = (x == 0)%s; VP_FC->of = 0;", bits, m == ZYDIS_MNEMONIC_BLSI ? " ? 0 : 1" : ""));
            line(wr(ops[0], bits, "r"));
            return true;
        }
        case ZYDIS_MNEMONIC_SARX: case ZYDIS_MNEMONIC_SHLX: case ZYDIS_MNEMONIC_SHRX: { // BMI2: no flags
            line(fmt("const uint64_t x = %s; const unsigned n = (unsigned)(%s) & %u;", rd(ops[1], bits).c_str(), rd(ops[2], bits).c_str(), bits == 64 ? 63u : 31u));
            if (m == ZYDIS_MNEMONIC_SHLX) line(wr(ops[0], bits, fmt("(x << n) & VP_MASK(%d)", bits)));
            else if (m == ZYDIS_MNEMONIC_SHRX) line(wr(ops[0], bits, fmt("(x & VP_MASK(%d)) >> n", bits)));
            else line(wr(ops[0], bits, fmt("vp_sar(%d, x, n)", bits)));
            return true;
        }
        case ZYDIS_MNEMONIC_RORX:
            line(wr(ops[0], bits, fmt("vp_ror(%d, %s, %u)", bits, rd(ops[1], bits).c_str(), (unsigned)ops[2].imm.value.u)));
            return true;
        case ZYDIS_MNEMONIC_BZHI:
            line(fmt("const uint64_t x = %s & VP_MASK(%d); const unsigned n = (unsigned)(%s) & 0xff;", rd(ops[1], bits).c_str(), bits, rd(ops[2], bits).c_str()));
            line(fmt("const uint64_t r = n >= %d ? x : x & ((UINT64_C(1) << n) - 1); VP_FC->cf = (n >= %d); vp_flags_result(VP_FC, %d, r); VP_FC->of = 0;", bits, bits, bits));
            line(wr(ops[0], bits, "r"));
            return true;
        case ZYDIS_MNEMONIC_MULX: { // BMI2: hi:lo = rdx * src, no flags
            line(fmt("const unsigned __int128 p = (unsigned __int128)(VP_R%d(VP_RDX)) * (%s & VP_MASK(%d));", bits, rd(ops[2], bits).c_str(), bits));
            line(wr(ops[1], bits, fmt("(uint64_t)p & VP_MASK(%d)", bits)));
            line(wr(ops[0], bits, fmt("(uint64_t)(p >> %d) & VP_MASK(%d)", bits, bits)));
            return true;
        }
        case ZYDIS_MNEMONIC_PDEP: case ZYDIS_MNEMONIC_PEXT: {
            line(fmt("const uint64_t src = %s & VP_MASK(%d), mask = %s & VP_MASK(%d); uint64_t r = 0;", rd(ops[1], bits).c_str(), bits, rd(ops[2], bits).c_str(), bits));
            if (m == ZYDIS_MNEMONIC_PDEP) line("for (uint64_t mm = mask, k = 0; mm; mm &= mm - 1, ++k) if ((src >> k) & 1) r |= mm & (0 - mm);");
            else line("for (uint64_t mm = mask, k = 0; mm; mm &= mm - 1, ++k) if (src & mm & (0 - mm)) r |= UINT64_C(1) << k;");
            line(wr(ops[0], bits, "r"));
            return true;
        }
        case ZYDIS_MNEMONIC_RDSEED: case ZYDIS_MNEMONIC_RDRAND:
            line("VP_FC->cf = 1; VP_FC->of = VP_FC->sf = VP_FC->zf = VP_FC->af = VP_FC->pf = 0;");
            line(wr(ops[0], bits, "vp_rdtsc(cpu) * 0x9e3779b97f4a7c15ull"));
            return true;
        default:
            break;
        }
        unsupported(name);
        return true;
    }

    const char* linkage = "static ";

    static void fappend(std::string& t, const char* f, ...) {
        va_list ap;
        va_start(ap, f);
        t += vfmt(f, ap);
        va_end(ap);
    }

    // The resume points that get their own label and entry: not the entry, not a landing pad.
    std::set<uint64_t> resumes_of(const Function& f) const {
        std::set<uint64_t> r;
        for (uint64_t e : f.resume_points) if (e != f.entry && !f.extra_entries.count(e) && f.blocks.size()) r.insert(e);
        return r;
    }

    std::set<uint64_t> cur_resumes;
    std::map<uint64_t, std::set<uint64_t>> resumes_emitted; // function -> resume labels written
    void emit_function(const Function& f) {
        current = &f;
        cur_resumes = resumes_of(f);
        std::string ftext;
        if (opt.regcache) {
            fappend(ftext, "%svoid %s(VpCpu* restrict cpu, uint32_t entry) {\n    VP_DECL();\n", linkage, fn_name(f.entry).c_str());
        } else {
            fappend(ftext, "%svoid %s(VpCpu* restrict cpu, uint32_t entry) {\n", linkage, fn_name(f.entry).c_str());
        }
        // The entry switch is written once the body is done: it lists only the resume points
        // whose call was actually emitted (a block boundary inside an instruction can make the
        // emitter decode a different sequence than discovery did).
        const size_t switch_at = ftext.size();
        std::set<uint64_t>& emitted_resumes = resumes_emitted[f.entry];
        emitted_resumes.clear();
        // Blocks are emitted in address order; parts of the function below its entry (hot/cold
        // splitting puts .text.unlikely before .text) must not run first.
        if (!f.blocks.empty() && *f.blocks.begin() != f.entry) fappend(ftext, "    goto %s;\n", label(f.entry).c_str());
        std::set<uint64_t> emitted;
        // Blocks in address order; a block runs until a transfer or the next block start.
        for (auto it = f.blocks.begin(); it != f.blocks.end(); ++it) {
            uint64_t a = *it;
            auto nx = std::next(it);
            const uint64_t limit = nx == f.blocks.end() ? UINT64_MAX : *nx;
            fappend(ftext, "%s:\n", label(a).c_str());
            if (f.tail_blocks.count(a)) {
                // Another function starts here: go there instead of copying its code.
                if (opt.regcache) fappend(ftext, "    VP_OUT();\n");
                fappend(ftext, "    cpu->rip = %s; vp_dispatch(cpu, cpu->rip); return;\n", A(a).c_str());
                continue;
            }
            bool falls = true;
            // Flag liveness inside the block (backward): which of the flags an instruction writes
            // are read by a later one before being written again. Everything is live at the end.
            std::vector<unsigned> needed;
            if (opt.lazy_flags) {
                std::vector<std::pair<unsigned, unsigned>> rw; // (written, tested) per instruction
                uint64_t q = a;
                while (q < limit) {
                    ZydisDecodedInstruction i2;
                    ZydisDecodedOperand o2[ZYDIS_MAX_OPERAND_COUNT];
                    if (!dec.decode(img, q, i2, o2)) break;
                    const ZydisAccessedFlags* fl = i2.cpu_flags;
                    const unsigned written = fl ? (unsigned)(fl->modified | fl->set_0 | fl->set_1 | fl->undefined) : 0xffffu;
                    unsigned tested = fl ? (unsigned)fl->tested : 0xffffu;
                    // An instruction that writes its flags only sometimes (a shift or rotate by a
                    // count that may be 0, a rep-prefixed string compare with rcx = 0) leaves the
                    // older values in place otherwise: they stay live through it.
                    if (writes_flags_conditionally(i2, o2)) tested |= written;
                    rw.emplace_back(written, tested);
                    q += i2.length;
                    if (ends_block(i2)) break;
                }
                needed.assign(rw.size(), 0xffffu);
                unsigned live = 0xffffu;
                for (size_t k = rw.size(); k-- > 0;) {
                    needed[k] = live & rw[k].first;
                    live = (live & ~rw[k].first) | rw[k].second;
                }
            }
            size_t index_in_block = 0;
            block_recent.clear();
            while (a < limit) {
                ZydisDecodedInstruction insn_;
                ZydisDecodedOperand ops_[ZYDIS_MAX_OPERAND_COUNT];
                if (!dec.decode(img, a, insn_, ops_)) {
                    fappend(ftext, "    vp_unsupported(cpu, %s, \"undecodable\"); return;\n", A(a).c_str());
                    falls = false;
                    break;
                }
                rip = a;
                next = a + insn_.length;
                insn = &insn_;
                ops = ops_;
                body.clear();
                if (block_recent.size() == 4) block_recent.erase(block_recent.begin());
                bool cont = true;
                try {
                    cont = emit_insn();
                } catch (const std::exception& e) {
                    body.clear();
                    unsupported(e.what());
                }
                char text[256];
                ZydisFormatter fm;
                ZydisFormatterInit(&fm, ZYDIS_FORMATTER_STYLE_INTEL);
                ZydisFormatterFormatInstruction(&fm, &insn_, ops_, insn_.operand_count_visible, text, sizeof text, a, nullptr);
                const unsigned mk = (opt.lazy_flags && index_in_block < needed.size()) ? (needed[index_in_block] & 0x8d5u) : 0x8d5u;
                ++index_in_block;
                if (mk != 0x8d5u) fappend(ftext, "#undef VP_FLAG_MASK\n#define VP_FLAG_MASK 0x%xu\n", mk);
                // Register cache: a body that calls something which reads or writes cpu->r (stack
                // helpers, mul/div, calls, dispatch, natives, faults) runs on the struct between a
                // write-back and a reload; every other body works on the locals.
                const bool helper = opt.regcache && (body.find("vp_mul1") != std::string::npos || body.find("vp_div1") != std::string::npos ||
                                                     body.find("vp_cpuid") != std::string::npos || body.find("vp_rdtsc") != std::string::npos || body.find("vp_syscall") != std::string::npos ||
                                                     body.find("vp_dispatch") != std::string::npos || body.find("vp_call_native") != std::string::npos ||
                                                     body.find(opt.symbol_prefix) != std::string::npos || body.find("vp_unsupported") != std::string::npos ||
                                                     body.find("vp_divide_error") != std::string::npos || body.find("return") != std::string::npos);
                if (helper) fappend(ftext, "#undef VP_LOCAL\n#define VP_LOCAL 0\n    VP_OUT();\n");
                fappend(ftext, "    /* %s: %s */\n    {\n%s    }\n", hex(a).c_str(), text, body.c_str());
                // A resume point: entered with the state in cpu, it reloads the locals (VP_IN below).
                if (insn_.mnemonic == ZYDIS_MNEMONIC_CALL && cur_resumes.count(next) && emitted_resumes.insert(next).second)
                    fappend(ftext, "R_%" PRIx64 ":;\n", next);
                if (helper) fappend(ftext, "    VP_IN();\n#undef VP_LOCAL\n#define VP_LOCAL 1\n");
                if (mk != 0x8d5u) fappend(ftext, "#undef VP_FLAG_MASK\n#define VP_FLAG_MASK VP_F_ALL\n");
                if (is_jcc(insn_.mnemonic) && insn_.mnemonic != ZYDIS_MNEMONIC_JCXZ && insn_.mnemonic != ZYDIS_MNEMONIC_JECXZ &&
                    insn_.mnemonic != ZYDIS_MNEMONIC_JRCXZ) {
                    const uint64_t t = branch_target(insn_, ops_, a);
                    fappend(ftext, "    if (vp_cc(VP_FC, %d)) goto %s;\n", cc_of(insn_.mnemonic), label(t).c_str());
                }
                block_recent.push_back(rip);
                a = next;
                if (!cont) { falls = false; break; }
            }
            if (falls && limit != UINT64_MAX && a >= limit) {
                // Fall-through into the next block in address order: nothing to emit.
            } else if (falls) {
                if (opt.regcache) fappend(ftext, "    VP_OUT();\n");
                fappend(ftext, "    cpu->rip = %s; vp_dispatch(cpu, cpu->rip); return;\n", A(a).c_str());
            }
        }
        fappend(ftext, "}\n\n");
        std::string sw;
        if (!f.extra_entries.empty() || !emitted_resumes.empty()) {
            fappend(sw, "    switch (entry) {\n");
            for (uint64_t e : f.extra_entries) fappend(sw, "    case %u: goto %s;\n", (unsigned)(e - f.entry), label(e).c_str());
            for (uint64_t e : emitted_resumes) fappend(sw, "    case %u: goto R_%" PRIx64 ";\n", (unsigned)(e - f.entry), e);
            fappend(sw, "    default: break;\n    }\n");
        } else {
            fappend(sw, "    (void)entry;\n");
        }
        ftext.insert(switch_at, sw);
        fputs(ftext.c_str(), out);
    }
};

} // namespace

void emit_c(const Image& img, const std::map<uint64_t, Function>& functions, const Options& opt,
            const std::string& out_path, Stats& stats) {
    // One translation unit, or several of `opt.split` functions each plus a header: a game's
    // executable is millions of instructions and a single C file would take the compiler hours.
    const bool split = opt.split > 0 && functions.size() > opt.split;
    std::string stem = out_path;
    if (stem.size() > 2 && stem.compare(stem.size() - 2, 2, ".c") == 0) stem.resize(stem.size() - 2);
    const char* linkage = split ? "" : "static ";
    auto fname = [&](uint64_t a) { return opt.symbol_prefix + fmt("%" PRIx64, a); };
    std::string header_name = stem + "_decl.h";
    // The module the generated code belongs to: its base is where image addresses are taken from.
    const std::string module_var = "vp_module_" + opt.module;
    const std::string module_decl = "extern __attribute__((visibility(\"hidden\"))) VpModule " + module_var + ";\n#define VP_MOD " + module_var + "\n\n";
    if (split) {
        FILE* h = fopen(header_name.c_str(), "w");
        if (!h) throw std::runtime_error("cannot write " + header_name);
        fprintf(h, "/* Generated by vpaot. Do not edit. */\n#pragma once\n#include \"vp_cpu.h\"\n#include \"vp_emit.h\"\n%s\n%s", opt.regcache ? "#define VP_REGCACHE 1\n#include \"vp_regs.h\"" : "#include \"vp_regs.h\"", module_decl.c_str());
        for (auto& [a, fn] : functions) fprintf(h, "__attribute__((visibility(\"hidden\"))) void %s(VpCpu* cpu, uint32_t entry);\n", fname(a).c_str());
        fclose(h);
    }
    FILE* f = nullptr;
    size_t index = 0, in_file = 0;
    std::vector<std::string> written;
    auto open_unit = [&]() {
        std::string path = split ? fmt("%s_%03zu.c", stem.c_str(), index++) : out_path;
        f = fopen(path.c_str(), "w");
        if (!f) throw std::runtime_error("cannot write " + path);
        written.push_back(path);
        if (split) {
            const size_t slash = header_name.find_last_of('/');
            fprintf(f, "/* Generated by vpaot. Do not edit. */\n#include \"%s\"\n\n", header_name.substr(slash == std::string::npos ? 0 : slash + 1).c_str());
        } else {
            fprintf(f, "/* Generated by vpaot. Do not edit. */\n#include \"vp_cpu.h\"\n#include \"vp_emit.h\"\n%s\n%s", opt.regcache ? "#define VP_REGCACHE 1\n#include \"vp_regs.h\"" : "#include \"vp_regs.h\"", module_decl.c_str());
            for (auto& [a, fn] : functions) fprintf(f, "static void %s(VpCpu* cpu, uint32_t entry);\n", fname(a).c_str());
            fprintf(f, "\n");
        }
        in_file = 0;
    };
    open_unit();
    Emitter e(img, opt, stats, f);
    e.functions = &functions;
    for (auto& [a, fn] : functions) {
        if (split && in_file == opt.split) { fclose(f); open_unit(); e.out = f; }
        e.linkage = linkage;
        e.emit_function(fn);
        ++in_file;
    }
    if (split) { fclose(f); open_unit(); }
    // The table the host uses to enter translated code: sorted by guest address.
    // Table addresses: absolute, or offsets from the link base in --pic output.
    auto T = [&](uint64_t a) { return opt.pic ? fmt("0x%" PRIx64 "ull", a - img.base) : hex(a); };
    const std::string M = opt.module;
    fprintf(f, "static const VpEntry %s_entries[] = {\n", M.c_str());
    for (auto& [a, fn] : functions) fprintf(f, "    { %s, %s },\n", T(a).c_str(), fname(a).c_str());
    if (functions.empty()) fprintf(f, "    { 0, 0 },\n");
    fprintf(f, "};\n");
    // Mid-function entries (landing pads, resume points): guest address -> function and entry
    // offset, sorted by address for the runtime's binary search.
    size_t extra = 0;
    std::vector<std::pair<uint64_t, uint64_t>> mids; // (address, function)
    for (auto& [a, fn] : functions) {
        for (uint64_t x : fn.extra_entries) mids.emplace_back(x, a);
        for (uint64_t x : e.resumes_emitted[a]) mids.emplace_back(x, a);
    }
    std::sort(mids.begin(), mids.end());
    mids.erase(std::unique(mids.begin(), mids.end(), [](auto& x, auto& y) { return x.first == y.first; }), mids.end());
    fprintf(f, "static const VpExtraEntry %s_extra[] = {\n", M.c_str());
    for (auto& [e, a] : mids) {
        fprintf(f, "    { %s, %s, %u },\n", T(e).c_str(), fname(a).c_str(), (unsigned)(e - a));
        ++extra;
    }
    if (!extra) fprintf(f, "    { 0, 0, 0 },\n");
    fprintf(f, "};\n");
    // Imports: the slots the runtime fills with the addresses of its native implementations.
    fprintf(f, "static const VpImport %s_imports[] = {\n", M.c_str());
    for (const auto& im : img.imports) {
        std::string esc;
        for (char ch : im.name) { if (ch == '"' || ch == '\\') esc += '\\'; esc += (ch >= 32 && ch < 127) ? ch : '?'; }
        fprintf(f, "    { \"%s\", %s, %d },\n", esc.c_str(), T(im.slot).c_str(), im.function ? 1 : 0);
    }
    if (img.imports.empty()) fprintf(f, "    { 0, 0, 0 },\n");
    fprintf(f, "};\n");
    // Code ranges and relocated fields, and the fingerprint of the code as translated (the same
    // FNV-1a 64 as vp_fingerprint in runtime/vp_host.c, over the image at its link base).
    std::vector<std::pair<uint64_t, uint64_t>> code;
    for (const auto& r : img.executable) code.emplace_back(r.start - img.base, r.end - r.start);
    std::vector<uint64_t> relocs;
    for (uint64_t a : img.loader_written) if (a >= img.base) relocs.push_back(a - img.base);
    std::sort(relocs.begin(), relocs.end());
    relocs.erase(std::unique(relocs.begin(), relocs.end()), relocs.end());
    uint64_t h = 1469598103934665603ull;
    {
        size_t r = 0;
        for (auto [start, size] : code) {
            for (uint64_t k = 0; k < size; ++k) {
                const uint64_t off = start + k;
                while (r < relocs.size() && relocs[r] + 8 <= off) ++r;
                const bool in_reloc = r < relocs.size() && relocs[r] <= off && off < relocs[r] + 8;
                const uint8_t byte = img.mapped(img.base + off) ? *img.at(img.base + off) : 0;
                h = (h ^ (in_reloc ? 0 : byte)) * 1099511628211ull;
            }
        }
    }
    fprintf(f, "static const VpRange %s_code[] = {\n", M.c_str());
    for (auto [start, size] : code) fprintf(f, "    { 0x%" PRIx64 "ull, 0x%" PRIx64 "ull },\n", start, size);
    if (code.empty()) fprintf(f, "    { 0, 0 },\n");
    fprintf(f, "};\nstatic const uint64_t %s_relocs[] = {\n", M.c_str());
    for (size_t i = 0; i < relocs.size(); ++i) fprintf(f, "%s0x%" PRIx64 "ull,%s", i % 8 ? "" : "    ", relocs[i], i % 8 == 7 ? "\n" : " ");
    if (relocs.empty()) fprintf(f, "    0");
    fprintf(f, "\n};\n");
    fprintf(f, "__attribute__((visibility(\"hidden\"))) VpModule %s = {\n    \"%s\", 0x%" PRIx64 "ull, 0x%" PRIx64 "ull, 0x%" PRIx64 "ull, %d,\n", module_var.c_str(), M.c_str(),
            img.base, img.base, img.end() - img.base, opt.pic ? 1 : 0);
    fprintf(f, "    %s_entries, %zu, %s_extra, %zu, %s_imports, %zu,\n", M.c_str(), functions.size(), M.c_str(), extra, M.c_str(), img.imports.size());
    fprintf(f, "    %s_code, %zu, %s_relocs, %zu, 0x%" PRIx64 "ull, %d, 0\n};\n", M.c_str(), code.size(), M.c_str(), relocs.size(), h, opt.pic ? 0 : 1);
    fprintf(f, "__attribute__((constructor)) static void %s_register(void) { vp_register_module(&%s); }\n", M.c_str(), module_var.c_str());
    fclose(f);
    if (split) {
        // A list of the units for the build system.
        FILE* l = fopen((stem + "_files.txt").c_str(), "w");
        if (l) { for (auto& w : written) fprintf(l, "%s\n", w.c_str()); fclose(l); }
    }
}

} // namespace vpaot
