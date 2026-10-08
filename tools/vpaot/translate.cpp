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

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
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
                                      const ZydisDecodedOperand* jmp_ops) {
    std::vector<uint64_t> out;
    uint64_t table = 0;
    bool relative = false;
    uint64_t count = 0;
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
        }
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

Function explore(const Image& img, const Decoder& dec, uint64_t entry, std::vector<uint64_t>& callees, const Options& opt) {
    Function f;
    f.entry = entry;
    f.blocks.insert(entry);
    std::vector<uint64_t> work{entry};
    std::set<uint64_t> seen;
    // The instructions before a block that falls through into it: jump-table shapes span the
    // conditional branch that bounds the index.
    std::map<uint64_t, std::vector<uint64_t>> recent_at;
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
            if (insn.mnemonic == ZYDIS_MNEMONIC_JMP && ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                auto targets = read_jump_table(img, dec, recent, a, insn, ops);
                if (!targets.empty()) {
                    for (uint64_t t : targets) { f.blocks.insert(t); work.push_back(t); }
                    f.jump_tables[a] = std::move(targets);
                }
            }
            recent.push_back(a);
            if (recent.size() > 12) recent.erase(recent.begin());
            if (insn.mnemonic == ZYDIS_MNEMONIC_CALL) {
                if (uint64_t t = branch_target(insn, ops, a); t && img.is_code(t)) callees.push_back(t);
            } else if (insn.mnemonic == ZYDIS_MNEMONIC_LEA && ops[1].mem.base == ZYDIS_REGISTER_RIP && ops[1].mem.disp.size) {
                // A function pointer taken in position-independent code.
                const uint64_t t = next + (uint64_t)ops[1].mem.disp.value;
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
        }
    }
    return f;
}

} // namespace

std::map<uint64_t, Function> discover(const Image& img, const std::vector<uint64_t>& roots, Stats& stats,
                                      const Options& opt) {
    Decoder dec;
    std::map<uint64_t, Function> out;
    std::vector<uint64_t> work(roots);
    for (uint64_t a : img.code_pointers) work.push_back(a);
    for (uint64_t a : img.eh_frame_starts) work.push_back(a);
    while (!work.empty() && out.size() < opt.max_functions) {
        const uint64_t e = work.back();
        work.pop_back();
        if (out.count(e) || !img.is_code(e) || opt.natives.count(e)) continue;
        std::vector<uint64_t> callees;
        Function f = explore(img, dec, e, callees, opt);
        out.emplace(e, std::move(f));
        for (uint64_t c : callees) if (!out.count(c)) work.push_back(c);
    }
    stats.functions = out.size();
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
    int vex_mask = -1;     // VEX blendv: the explicit mask register (legacy forms use xmm0)
    bool vex_src2 = false; // VEX: the second source was copied to `vsrc2` before the destination changed
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
        return -1;
    }
    static const char* ctype(int bits) {
        switch (bits) { case 8: return "uint8_t"; case 16: return "uint16_t"; case 32: return "uint32_t"; default: return "uint64_t"; }
    }

    std::string reg_rd(ZydisRegister r, int bits) {
        if (r == ZYDIS_REGISTER_RIP) return hex(next);
        if (int x = xmm_index(r); x >= 0) return fmt("cpu->xmm[%d]", x);
        const int i = gpr_index(r);
        if (i < 0) throw std::runtime_error(fmt("register %s", ZydisRegisterGetString(r)));
        if (bits == 8 && is_high8(r)) return fmt("vp_r8h(cpu, %d)", i);
        return fmt("vp_r%d(cpu, %d)", bits, i);
    }
    std::string reg_wr(ZydisRegister r, int bits, const std::string& v) {
        if (int x = xmm_index(r); x >= 0) return fmt("cpu->xmm[%d] = %s;", x, v.c_str());
        const int i = gpr_index(r);
        if (i < 0) throw std::runtime_error(fmt("register %s", ZydisRegisterGetString(r)));
        if (bits == 8 && is_high8(r)) return fmt("vp_w8h(cpu, %d, (uint8_t)(%s));", i, v.c_str());
        return fmt("vp_w%d(cpu, %d, (%s)(%s));", bits, i, ctype(bits), v.c_str());
    }

    // -- memory -----------------------------------------------------------------------------------
    std::string ea(const ZydisDecodedOperand& op) {
        const auto& m = op.mem;
        std::string e;
        if (m.base == ZYDIS_REGISTER_RIP) {
            e = hex(next);
        } else if (m.base != ZYDIS_REGISTER_NONE) {
            e = reg_rd(m.base, 64);
        }
        if (m.index != ZYDIS_REGISTER_NONE && m.scale) {
            const std::string ix = reg_rd(m.index, 64) + (m.scale > 1 ? fmt(" * %u", m.scale) : "");
            e = e.empty() ? ix : e + " + " + ix;
        }
        if (m.disp.size && m.disp.value) {
            e = e.empty() ? fmt("(uint64_t)%" PRId64, m.disp.value)
                          : e + fmt(" + (uint64_t)%" PRId64 "ll", (long long)m.disp.value);
        }
        if (m.segment == ZYDIS_REGISTER_FS) e = "cpu->fs_base" + (e.empty() ? "" : " + " + e);
        else if (m.segment == ZYDIS_REGISTER_GS) e = "cpu->gs_base" + (e.empty() ? "" : " + " + e);
        if (e.empty()) e = "0";
        if (insn->address_width == 32) e = "(uint64_t)(uint32_t)(" + e + ")";
        return "(" + e + ")";
    }

    // Reads operand `op` as an unsigned integer of `bits`; `addr` is a precomputed EA variable.
    std::string rd(const ZydisDecodedOperand& op, int bits, const std::string& addr = "") {
        switch (op.type) {
        case ZYDIS_OPERAND_TYPE_REGISTER: return reg_rd(op.reg.value, bits);
        case ZYDIS_OPERAND_TYPE_MEMORY: return fmt("vp_ld%d(%s)", bits, addr.empty() ? ea(op).c_str() : addr.c_str());
        case ZYDIS_OPERAND_TYPE_IMMEDIATE:
            if (op.imm.is_signed) return fmt("((%s)(int64_t)%" PRId64 "ll)", ctype(bits), (long long)op.imm.value.s);
            return fmt("((%s)%" PRIu64 "ull)", ctype(bits), (unsigned long long)op.imm.value.u);
        default: throw std::runtime_error("operand type");
        }
    }
    std::string wr(const ZydisDecodedOperand& op, int bits, const std::string& v, const std::string& addr = "") {
        switch (op.type) {
        case ZYDIS_OPERAND_TYPE_REGISTER: return reg_wr(op.reg.value, bits, v);
        case ZYDIS_OPERAND_TYPE_MEMORY: return fmt("vp_st%d(%s, (%s)(%s));", bits, addr.empty() ? ea(op).c_str() : addr.c_str(), ctype(bits), v.c_str());
        default: throw std::runtime_error("write to immediate");
        }
    }

    // For read-modify-write operands the address is computed once.
    std::string bind_addr(const ZydisDecodedOperand& op) {
        if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) return "";
        line("const uint64_t ea = " + ea(op) + ";");
        return "ea";
    }

    std::string fn_name(uint64_t a) const { return opt.symbol_prefix + fmt("%" PRIx64, a); }
    std::string label(uint64_t a) const { return fmt("L_%" PRIx64, a); }

    void unsupported(const char* what) {
        stats.unsupported++;
        stats.unsupported_by_mnemonic[what]++;
        line(fmt("vp_unsupported(cpu, %s, \"%s\"); return;", hex(rip).c_str(), what));
    }

    // -- integer ALU ------------------------------------------------------------------------------
    void alu2(const char* flags, const char* cop, bool store) {
        const int bits = ops[0].size;
        if (store && (insn->attributes & ZYDIS_ATTRIB_HAS_LOCK) && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            // lock add/sub/and/or/xor: one atomic read-modify-write; the flags come from the values seen.
            const char* fn = !strcmp(cop, "+") ? "add" : !strcmp(cop, "-") ? "sub" : !strcmp(cop, "&") ? "and" : !strcmp(cop, "|") ? "or" : "xor";
            line("const uint64_t ea = " + ea(ops[0]) + ";");
            line(fmt("const uint64_t y = %s;", rd(ops[1], bits).c_str()));
            line(fmt("const uint64_t x = __atomic_fetch_%s((uint%d_t*)(uintptr_t)ea, (uint%d_t)y, __ATOMIC_SEQ_CST);", fn, bits, bits));
            line(fmt("const uint64_t r = (x %s y) & VP_MASK(%d);", cop, bits));
            if (flags[0] == 'l') line(fmt("vp_flags_logic(cpu, %d, r);", bits));
            else line(fmt("vp_flags_%s(cpu, %d, x, y, r);", flags, bits));
            return;
        }
        const std::string a = bind_addr(ops[0]);
        line(fmt("const uint64_t x = %s, y = %s;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str()));
        line(fmt("const uint64_t r = (x %s y) & VP_MASK(%d);", cop, bits));
        if (flags[0] == 'l') line(fmt("vp_flags_logic(cpu, %d, r);", bits));
        else line(fmt("vp_flags_%s(cpu, %d, x, y, r);", flags, bits));
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
        line(fmt("    vp_flags_%s(cpu, %d, x, n, r);", kind, bits));
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
        if (left) line("    cpu->cf = (uint8_t)(r & 1);"), line(fmt("    cpu->of = (uint8_t)(cpu->cf ^ VP_SIGN(%d, r));", bits));
        else line(fmt("    cpu->cf = VP_SIGN(%d, r);", bits)), line(fmt("    cpu->of = (uint8_t)(VP_SIGN(%d, r) ^ ((r >> (%d - 2)) & 1));", bits, bits));
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
    void string_op(int w, bool movs) {
        const bool rep = insn->attributes & ZYDIS_ATTRIB_HAS_REP;
        line(fmt("const int64_t step = cpu->df ? -%d : %d;", w / 8, w / 8));
        if (rep) line("while (cpu->r[VP_RCX]) {");
        if (movs) line(fmt("    vp_st%d(cpu->r[VP_RDI], vp_ld%d(cpu->r[VP_RSI])); cpu->r[VP_RSI] += step; cpu->r[VP_RDI] += step;", w, w));
        else line(fmt("    vp_st%d(cpu->r[VP_RDI], vp_r%d(cpu, VP_RAX)); cpu->r[VP_RDI] += step;", w, w));
        if (rep) line("    cpu->r[VP_RCX] -= 1; }");
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
        V(CMPPS) V(CMPPD) V(PABSD) V(PABSW) V(PABSB) V(PSIGND) V(PHADDD) V(PAVGB) V(PAVGW) V(PSADBW) V(PCMPISTRI)
#undef V
        default: return ZYDIS_MNEMONIC_INVALID;
        }
    }

    // -- one instruction --------------------------------------------------------------------------
    // Returns false when the instruction ended the block with a transfer (nothing falls through).
    bool emit_insn() {
        if ((insn->attributes & ZYDIS_ATTRIB_HAS_VEX) && legacy_of(insn->mnemonic) != ZYDIS_MNEMONIC_INVALID) {
            return emit_vex();
        }
        return emit_legacy(insn->mnemonic, insn->operand_count_visible);
    }

    // VEX: 256-bit (ymm) forms are not modelled (VpXmm is 128 bits); 128-bit forms are the legacy
    // operation with the result in a separate destination: dst = src1, then dst op= src2. Scalar
    // VEX ops take their upper lanes from src1, which the copy provides; VEX zeroes ymm's upper
    // half, which does not exist here.
    bool emit_vex() {
        const ZydisMnemonic legacy = legacy_of(insn->mnemonic);
        const int n = insn->operand_count_visible;
        for (int i = 0; i < n; ++i) {
            if (ops[i].type == ZYDIS_OPERAND_TYPE_REGISTER && ZydisRegisterGetClass(ops[i].reg.value) == ZYDIS_REGCLASS_YMM) {
                stats.instructions++;
                stats.by_mnemonic[ZydisMnemonicGetString(insn->mnemonic)]++;
                unsupported("ymm");
                return true;
            }
        }
        ZydisDecodedOperand remapped[ZYDIS_MAX_OPERAND_COUNT];
        std::memcpy(remapped, ops, sizeof remapped);
        const ZydisDecodedOperand* saved = ops;
        int count = n;
        const bool three = n >= 3 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER &&
                           xmm_index(ops[0].reg.value) >= 0 && xmm_index(ops[1].reg.value) >= 0 &&
                           !(ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE && n == 3);
        if (three) {
            // dst = src1 (when they differ), then the legacy 2-operand form dst op= src2 [, imm].
            // Every source is read before the destination changes: the destination may also be
            // a source (vblendvpd xmm0, xmm15, xmm14, xmm0), so the sources are copied first.
            line(fmt("const VpXmm vsrc1 = cpu->xmm[%d];", xmm_index(ops[1].reg.value)));
            remapped[1] = ops[2];
            if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER && xmm_index(ops[2].reg.value) >= 0) {
                line(fmt("const VpXmm vsrc2 = cpu->xmm[%d];", xmm_index(ops[2].reg.value)));
                vex_src2 = true;
            }
            if (n >= 4) remapped[2] = ops[3];
            count = n - 1;
            vex_mask = (n >= 4 && ops[3].type == ZYDIS_OPERAND_TYPE_REGISTER && xmm_index(ops[3].reg.value) >= 0) ? xmm_index(ops[3].reg.value) : -1;
            if (vex_mask >= 0) line(fmt("const VpXmm vmask = cpu->xmm[%d];", vex_mask));
            line(fmt("cpu->xmm[%d] = vsrc1;", xmm_index(ops[0].reg.value)));
        }
        ops = remapped;
        const bool r = emit_legacy(legacy, count);
        ops = saved;
        vex_mask = -1;
        vex_src2 = false;
        return r;
    }

    bool emit_legacy(ZydisMnemonic mnemonic, int operand_count) {
        using M = ZydisMnemonic;
        const M m = mnemonic;
        const int nops = operand_count;
        const int bits = nops ? ops[0].size : 0;
        const char* name = ZydisMnemonicGetString(insn->mnemonic);
        stats.instructions++;
        stats.by_mnemonic[name]++;
        if (opt.emit_rip_updates) line("cpu->rip = " + hex(rip) + ";");
        if (opt.trace) line("vp_trace(cpu, " + hex(rip) + ");");

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
                line(fmt("const uint64_t x = __atomic_exchange_n((uint%d_t*)(uintptr_t)ea, (uint%d_t)%s, __ATOMIC_SEQ_CST);", bits, bits, rd(ro, bits).c_str()));
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
            line(fmt("if (vp_cc(cpu, %d)) { %s } else { %s }", cc_of(m), wr(ops[0], bits, "v").c_str(),
                     bits == 32 ? wr(ops[0], 32, rd(ops[0], 32)).c_str() : ""));
            return true;
        case ZYDIS_MNEMONIC_SETO: case ZYDIS_MNEMONIC_SETNO: case ZYDIS_MNEMONIC_SETB: case ZYDIS_MNEMONIC_SETNB:
        case ZYDIS_MNEMONIC_SETZ: case ZYDIS_MNEMONIC_SETNZ: case ZYDIS_MNEMONIC_SETBE: case ZYDIS_MNEMONIC_SETNBE:
        case ZYDIS_MNEMONIC_SETS: case ZYDIS_MNEMONIC_SETNS: case ZYDIS_MNEMONIC_SETP: case ZYDIS_MNEMONIC_SETNP:
        case ZYDIS_MNEMONIC_SETL: case ZYDIS_MNEMONIC_SETNL: case ZYDIS_MNEMONIC_SETLE: case ZYDIS_MNEMONIC_SETNLE:
            line(wr(ops[0], 8, fmt("vp_cc(cpu, %d)", cc_of(m))));
            return true;
        case ZYDIS_MNEMONIC_PUSH:
            line(fmt("vp_push64(cpu, %s);", rd(ops[0], 64).c_str()));
            return true;
        case ZYDIS_MNEMONIC_POP:
            if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
                line("const uint64_t v = vp_pop64(cpu);");
                line(wr(ops[0], 64, "v"));
            } else {
                line(wr(ops[0], 64, "vp_pop64(cpu)"));
            }
            return true;
        case ZYDIS_MNEMONIC_LEAVE:
            line("cpu->r[VP_RSP] = cpu->r[VP_RBP];");
            line("cpu->r[VP_RBP] = vp_pop64(cpu);");
            return true;
        case ZYDIS_MNEMONIC_CDQE: line("vp_w64(cpu, VP_RAX, (uint64_t)(int64_t)(int32_t)vp_r32(cpu, VP_RAX));"); return true;
        case ZYDIS_MNEMONIC_CWDE: line("vp_w32(cpu, VP_RAX, (uint32_t)(int32_t)(int16_t)vp_r16(cpu, VP_RAX));"); return true;
        case ZYDIS_MNEMONIC_CBW: line("vp_w16(cpu, VP_RAX, (uint16_t)(int16_t)(int8_t)vp_r8(cpu, VP_RAX));"); return true;
        case ZYDIS_MNEMONIC_CQO: line("vp_w64(cpu, VP_RDX, (cpu->r[VP_RAX] >> 63) ? ~UINT64_C(0) : 0);"); return true;
        case ZYDIS_MNEMONIC_CDQ: line("vp_w32(cpu, VP_RDX, (vp_r32(cpu, VP_RAX) >> 31) ? 0xffffffffu : 0);"); return true;
        case ZYDIS_MNEMONIC_CWD: line("vp_w16(cpu, VP_RDX, (vp_r16(cpu, VP_RAX) >> 15) ? 0xffffu : 0);"); return true;
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
            line(fmt("const uint64_t x = %s, y = %s, c = cpu->cf;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str()));
            line(fmt("const uint64_t r = (x + y + c) & VP_MASK(%d);", bits));
            line(fmt("vp_flags_adc(cpu, %d, x, y, c, r);", bits));
            line(wr(ops[0], bits, "r", a));
            return true;
        }
        case ZYDIS_MNEMONIC_SBB: {
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s, y = %s, c = cpu->cf;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str()));
            line(fmt("const uint64_t r = (x - y - c) & VP_MASK(%d);", bits));
            line(fmt("vp_flags_sbb(cpu, %d, x, y, c, r);", bits));
            line(wr(ops[0], bits, "r", a));
            return true;
        }
        case ZYDIS_MNEMONIC_INC: case ZYDIS_MNEMONIC_DEC: {
            const bool inc = m == ZYDIS_MNEMONIC_INC;
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s;", rd(ops[0], bits, a).c_str()));
            line(fmt("const uint64_t r = (x %s 1) & VP_MASK(%d);", inc ? "+" : "-", bits));
            line(fmt("vp_flags_%s(cpu, %d, x, r);", inc ? "inc" : "dec", bits));
            line(wr(ops[0], bits, "r", a));
            return true;
        }
        case ZYDIS_MNEMONIC_NEG: {
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s;", rd(ops[0], bits, a).c_str()));
            line(fmt("const uint64_t r = (0 - x) & VP_MASK(%d);", bits));
            line(fmt("vp_flags_sub(cpu, %d, 0, x, r);", bits));
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
            if (nops == 2) { line(wr(ops[0], bits, fmt("vp_imul(cpu, %d, %s, %s)", bits, rd(ops[0], bits).c_str(), rd(ops[1], bits).c_str()))); return true; }
            line(wr(ops[0], bits, fmt("vp_imul(cpu, %d, %s, %s)", bits, rd(ops[1], bits).c_str(), rd(ops[2], bits).c_str())));
            return true;
        case ZYDIS_MNEMONIC_MUL: line(fmt("vp_mul1(cpu, %d, %s, 0);", bits, rd(ops[0], bits).c_str())); return true;
        case ZYDIS_MNEMONIC_DIV: case ZYDIS_MNEMONIC_IDIV:
            line(fmt("if (vp_div1(cpu, %d, %s, %d)) { vp_divide_error(cpu, %s); return; }", bits, rd(ops[0], bits).c_str(),
                     m == ZYDIS_MNEMONIC_IDIV, hex(rip).c_str()));
            return true;
        case ZYDIS_MNEMONIC_BT: case ZYDIS_MNEMONIC_BTS: case ZYDIS_MNEMONIC_BTR: case ZYDIS_MNEMONIC_BTC: {
            if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) { unsupported("bt-mem-reg"); return true; }
            const std::string a = bind_addr(ops[0]);
            line(fmt("const uint64_t x = %s; const unsigned b = (unsigned)(%s) & %u;", rd(ops[0], bits, a).c_str(), rd(ops[1], ops[1].size).c_str(), bits - 1));
            line("cpu->cf = (uint8_t)((x >> b) & 1);");
            if (m == ZYDIS_MNEMONIC_BTS) line(wr(ops[0], bits, "x | (UINT64_C(1) << b)", a));
            if (m == ZYDIS_MNEMONIC_BTR) line(wr(ops[0], bits, "x & ~(UINT64_C(1) << b)", a));
            if (m == ZYDIS_MNEMONIC_BTC) line(wr(ops[0], bits, "x ^ (UINT64_C(1) << b)", a));
            return true;
        }
        case ZYDIS_MNEMONIC_BSF: case ZYDIS_MNEMONIC_BSR: case ZYDIS_MNEMONIC_TZCNT: case ZYDIS_MNEMONIC_LZCNT: {
            line(fmt("const uint64_t x = %s & VP_MASK(%d);", rd(ops[1], bits).c_str(), bits));
            if (m == ZYDIS_MNEMONIC_BSF || m == ZYDIS_MNEMONIC_BSR) {
                line("cpu->zf = (x == 0);");
                line("if (x) { " + wr(ops[0], bits, m == ZYDIS_MNEMONIC_BSF ? "(uint64_t)__builtin_ctzll(x)" : "(uint64_t)(63 - __builtin_clzll(x))") + " }");
            } else if (m == ZYDIS_MNEMONIC_TZCNT) {
                line(fmt("cpu->cf = (x == 0); const uint64_t r = x ? (uint64_t)__builtin_ctzll(x) : %d; cpu->zf = (r == 0);", bits));
                line(wr(ops[0], bits, "r"));
            } else {
                line(fmt("cpu->cf = (x == 0); const uint64_t r = x ? (uint64_t)__builtin_clzll(x) - (64 - %d) : %d; cpu->zf = (r == 0);", bits, bits));
                line(wr(ops[0], bits, "r"));
            }
            return true;
        }
        case ZYDIS_MNEMONIC_POPCNT:
            line(fmt("const uint64_t r = (uint64_t)__builtin_popcountll(%s & VP_MASK(%d));", rd(ops[1], bits).c_str(), bits));
            line("cpu->cf = cpu->of = cpu->sf = cpu->pf = cpu->af = 0; cpu->zf = (r == 0);");
            line(wr(ops[0], bits, "r"));
            return true;
        case ZYDIS_MNEMONIC_BSWAP:
            line(wr(ops[0], bits, fmt(bits == 64 ? "__builtin_bswap64(%s)" : "__builtin_bswap32((uint32_t)%s)", rd(ops[0], bits).c_str())));
            return true;
        case ZYDIS_MNEMONIC_CLC: line("cpu->cf = 0;"); return true;
        case ZYDIS_MNEMONIC_STC: line("cpu->cf = 1;"); return true;
        case ZYDIS_MNEMONIC_CMC: line("cpu->cf ^= 1;"); return true;
        case ZYDIS_MNEMONIC_CLD: line("cpu->df = 0;"); return true;
        case ZYDIS_MNEMONIC_STD: line("cpu->df = 1;"); return true;

        // Control flow
        case ZYDIS_MNEMONIC_JMP: {
            if (uint64_t t = branch_target(*insn, ops, rip); t && opt.natives.count(t)) {
                line(fmt("vp_call_native(cpu, %s); return;", hex(t).c_str())); // a tail call: the native pops our caller's return address
            } else if (t && img.is_code(t)) {
                line("goto " + label(t) + ";");
            } else if (current && current->jump_tables.count(rip)) {
                stats.jump_tables++;
                line(fmt("const uint64_t t = %s;", rd(ops[0], 64).c_str()));
                line("switch (t) {");
                for (uint64_t t : current->jump_tables.at(rip)) line(fmt("case %s: goto %s;", hex(t).c_str(), label(t).c_str()));
                line("default: cpu->rip = t; vp_dispatch(cpu, t); return;");
                line("}");
            } else {
                stats.indirect_jumps++;
                line(fmt("cpu->rip = %s; vp_dispatch(cpu, cpu->rip); return;", rd(ops[0], 64).c_str()));
            }
            return false;
        }
        case ZYDIS_MNEMONIC_CALL: {
            // The target is read before the return address is pushed (it may be rsp-relative).
            if (uint64_t t = branch_target(*insn, ops, rip); t && opt.natives.count(t)) {
                line(fmt("vp_push64(cpu, %s);", hex(next).c_str()));
                line(fmt("vp_call_native(cpu, %s);", hex(t).c_str()));
            } else if (t && img.is_code(t)) {
                line(fmt("vp_push64(cpu, %s);", hex(next).c_str()));
                line(fmt("%s(cpu, 0);", fn_name(t).c_str()));
            } else {
                stats.indirect_calls++;
                line(fmt("const uint64_t target = %s;", rd(ops[0], 64).c_str()));
                line(fmt("vp_push64(cpu, %s);", hex(next).c_str()));
                line("vp_dispatch(cpu, target);");
            }
            // The callee returned to the address after the call unless it unwound somewhere else.
            line(fmt("if (cpu->rip != %s) return;", hex(next).c_str()));
            return true;
        }
        case ZYDIS_MNEMONIC_RET:
            line("cpu->rip = vp_pop64(cpu);");
            if (nops && ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) line(fmt("cpu->r[VP_RSP] += %" PRIu64 ";", ops[0].imm.value.u));
            line("return;");
            return false;
        case ZYDIS_MNEMONIC_JCXZ: case ZYDIS_MNEMONIC_JECXZ: case ZYDIS_MNEMONIC_JRCXZ: {
            const uint64_t t = branch_target(*insn, ops, rip);
            const int w = m == ZYDIS_MNEMONIC_JRCXZ ? 64 : m == ZYDIS_MNEMONIC_JECXZ ? 32 : 16;
            line(fmt("if (vp_r%d(cpu, VP_RCX) == 0) goto %s;", w, label(t).c_str()));
            return true;
        }
        case ZYDIS_MNEMONIC_LOOP: case ZYDIS_MNEMONIC_LOOPE: case ZYDIS_MNEMONIC_LOOPNE: {
            const uint64_t t = branch_target(*insn, ops, rip);
            line("cpu->r[VP_RCX] -= 1;");
            const char* extra = m == ZYDIS_MNEMONIC_LOOPE ? " && cpu->zf" : m == ZYDIS_MNEMONIC_LOOPNE ? " && !cpu->zf" : "";
            line(fmt("if (cpu->r[VP_RCX] != 0%s) goto %s;", extra, label(t).c_str()));
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
        case ZYDIS_MNEMONIC_RDTSC:
            line("const uint64_t t = vp_rdtsc(cpu); vp_w32(cpu, VP_RAX, (uint32_t)t); vp_w32(cpu, VP_RDX, (uint32_t)(t >> 32));");
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
                line(fmt("%s.u%d[0] = %s.u%d[0];", xmm_dst().c_str(), w, reg_rd(ops[1].reg.value, 128).c_str(), w));
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
        case ZYDIS_MNEMONIC_MOVLHPS: line(fmt("%s.u64[1] = %s.u64[0];", xmm_dst().c_str(), reg_rd(ops[1].reg.value, 128).c_str())); return true;
        case ZYDIS_MNEMONIC_MOVHLPS: line(fmt("%s.u64[0] = %s.u64[1];", xmm_dst().c_str(), reg_rd(ops[1].reg.value, 128).c_str())); return true;

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
            line(fmt("vp_comiss(cpu, %s.f32[0], %s);", xmm_dst().c_str(), f32_rd(ops[1]).c_str()));
            return true;
        case ZYDIS_MNEMONIC_UCOMISD: case ZYDIS_MNEMONIC_COMISD:
            line(fmt("vp_comisd(cpu, %s.f64[0], %s);", xmm_dst().c_str(), f64_rd(ops[1]).c_str()));
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
            if (insn->meta.category == ZYDIS_CATEGORY_STRINGOP) { unsupported("cmpsd-string"); return true; }
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
        case ZYDIS_MNEMONIC_ROUNDSS: case ZYDIS_MNEMONIC_ROUNDSD: case ZYDIS_MNEMONIC_ROUNDPS: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            const char* fn = (imm & 4) ? "nearbyint" : (imm & 3) == 0 ? "rint" : (imm & 3) == 1 ? "floor" : (imm & 3) == 2 ? "ceil" : "trunc";
            if (m == ZYDIS_MNEMONIC_ROUNDSS) line(fmt("%s.f32[0] = (float)%s(%s);", xmm_dst().c_str(), fn, f32_rd(ops[1]).c_str()));
            else if (m == ZYDIS_MNEMONIC_ROUNDSD) line(fmt("%s.f64[0] = %s(%s);", xmm_dst().c_str(), fn, f64_rd(ops[1]).c_str()));
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
            line("cpu->zf = ((a.u64[0] & s.u64[0]) | (a.u64[1] & s.u64[1])) == 0; cpu->cf = ((~a.u64[0] & s.u64[0]) | (~a.u64[1] & s.u64[1])) == 0; cpu->of = cpu->sf = cpu->pf = cpu->af = 0;");
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
            else line(fmt("const uint32_t v = %s.u32[%u];", reg_rd(ops[1].reg.value, 128).c_str(), src_lane));
            line(fmt("%s.u32[%u] = v;", xmm_dst().c_str(), dst_lane));
            for (int i = 0; i < 4; ++i) if ((imm >> i) & 1) line(fmt("%s.u32[%d] = 0;", xmm_dst().c_str(), i));
            return true;
        }
        case ZYDIS_MNEMONIC_EXTRACTPS:
            line(wr(ops[0], ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? 32 : bits, fmt("%s.u32[%u]", reg_rd(ops[1].reg.value, 128).c_str(), (unsigned)ops[2].imm.value.u & 3)));
            return true;
        case ZYDIS_MNEMONIC_DPPS: {
            const unsigned imm = (unsigned)ops[2].imm.value.u;
            line("const VpXmm a = " + xmm_dst() + ", s = " + xmm_rd(ops[1]) + "; float d = 0.0f;");
            for (int i = 0; i < 4; ++i) if ((imm >> (4 + i)) & 1) line(fmt("d += a.f32[%d] * s.f32[%d];", i, i));
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
                line(fmt("const uint64_t x = __atomic_fetch_add((uint%d_t*)(uintptr_t)ea, (uint%d_t)y, __ATOMIC_SEQ_CST);", bits, bits));
                line(fmt("vp_flags_add(cpu, %d, x, y, (x + y) & VP_MASK(%d));", bits, bits));
                line(wr(ops[1], bits, "x"));
            } else {
                const std::string a = bind_addr(ops[0]);
                line(fmt("const uint64_t x = %s, y = %s;", rd(ops[0], bits, a).c_str(), rd(ops[1], bits).c_str()));
                line(fmt("const uint64_t r = (x + y) & VP_MASK(%d);", bits));
                line(fmt("vp_flags_add(cpu, %d, x, y, r);", bits));
                line(wr(ops[1], bits, "x"));
                line(wr(ops[0], bits, "r", a));
            }
            return true;
        }
        case ZYDIS_MNEMONIC_CMPXCHG: {
            const bool lock = insn->attributes & ZYDIS_ATTRIB_HAS_LOCK;
            line(fmt("uint64_t expected = vp_r%d(cpu, VP_RAX); const uint64_t desired = %s;", bits, rd(ops[1], bits).c_str()));
            if (lock && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
                line("const uint64_t ea = " + ea(ops[0]) + ";");
                line(fmt("uint%d_t exp = (uint%d_t)expected;", bits, bits));
                line(fmt("const int ok = __atomic_compare_exchange_n((uint%d_t*)(uintptr_t)ea, &exp, (uint%d_t)desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);", bits, bits));
                line(fmt("vp_flags_sub(cpu, %d, expected, exp, (expected - exp) & VP_MASK(%d));", bits, bits));
                line(fmt("if (!ok) vp_w%d(cpu, VP_RAX, (uint%d_t)exp);", bits, bits));
            } else {
                const std::string a = bind_addr(ops[0]);
                line(fmt("const uint64_t cur = %s;", rd(ops[0], bits, a).c_str()));
                line(fmt("vp_flags_sub(cpu, %d, expected, cur, (expected - cur) & VP_MASK(%d));", bits, bits));
                line("if (cur == expected) { " + wr(ops[0], bits, "desired", a) + " } else { " + fmt("vp_w%d(cpu, VP_RAX, (uint%d_t)cur);", bits, bits) + " }");
            }
            return true;
        }
        case ZYDIS_MNEMONIC_CMPXCHG16B: {
            line("const uint64_t ea = " + ea(ops[0]) + ";");
            line("unsigned __int128 expected = ((unsigned __int128)cpu->r[VP_RDX] << 64) | cpu->r[VP_RAX];");
            line("const unsigned __int128 desired = ((unsigned __int128)cpu->r[VP_RCX] << 64) | cpu->r[VP_RBX];");
            line("const int ok = __atomic_compare_exchange_n((unsigned __int128*)(uintptr_t)ea, &expected, desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);");
            line("cpu->zf = (uint8_t)ok; if (!ok) { cpu->r[VP_RAX] = (uint64_t)expected; cpu->r[VP_RDX] = (uint64_t)(expected >> 64); }");
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
            line(fmt("    vp_flags_%s(cpu, %d, x, n, r);", left ? "shl" : "shr", bits));
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
            line(fmt("vp_flags_logic(cpu, %d, r);", bits));
            line(wr(ops[0], bits, "r"));
            return true;
        case ZYDIS_MNEMONIC_BLSR: case ZYDIS_MNEMONIC_BLSI: case ZYDIS_MNEMONIC_BLSMSK: {
            line(fmt("const uint64_t x = %s & VP_MASK(%d);", rd(ops[1], bits).c_str(), bits));
            if (m == ZYDIS_MNEMONIC_BLSR) line(fmt("const uint64_t r = (x & (x - 1)) & VP_MASK(%d);", bits));
            else if (m == ZYDIS_MNEMONIC_BLSI) line(fmt("const uint64_t r = (x & (0 - x)) & VP_MASK(%d);", bits));
            else line(fmt("const uint64_t r = (x ^ (x - 1)) & VP_MASK(%d);", bits));
            line(fmt("vp_flags_result(cpu, %d, r); cpu->cf = (x == 0)%s; cpu->of = 0;", bits, m == ZYDIS_MNEMONIC_BLSI ? " ? 0 : 1" : ""));
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
            line(fmt("const uint64_t r = n >= %d ? x : x & ((UINT64_C(1) << n) - 1); cpu->cf = (n >= %d); vp_flags_result(cpu, %d, r); cpu->of = 0;", bits, bits, bits));
            line(wr(ops[0], bits, "r"));
            return true;
        case ZYDIS_MNEMONIC_MULX: { // BMI2: hi:lo = rdx * src, no flags
            line(fmt("const unsigned __int128 p = (unsigned __int128)(vp_r%d(cpu, VP_RDX)) * (%s & VP_MASK(%d));", bits, rd(ops[2], bits).c_str(), bits));
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
            line("cpu->cf = 1; cpu->of = cpu->sf = cpu->zf = cpu->af = cpu->pf = 0;");
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
        char buf[4096];
        va_list ap;
        va_start(ap, f);
        vsnprintf(buf, sizeof buf, f, ap);
        va_end(ap);
        t += buf;
    }

    void emit_function(const Function& f) {
        current = &f;
        std::string ftext;
        if (opt.locals) {
            // The state lives in a local for the whole function: the compiler keeps it in machine
            // registers, spills it around calls (which receive its address) and the returns
            // copy it back to the caller's state.
            fappend(ftext, "%svoid %s(VpCpu* restrict cpu_in, uint32_t entry) {\n    VpCpu L = *cpu_in; VpCpu* const cpu = &L;\n", linkage, fn_name(f.entry).c_str());
        } else {
            fappend(ftext, "%svoid %s(VpCpu* restrict cpu, uint32_t entry) {\n", linkage, fn_name(f.entry).c_str());
        }
        if (!f.extra_entries.empty()) {
            fappend(ftext, "    switch (entry) {\n");
            for (uint64_t e : f.extra_entries) fappend(ftext, "    case %u: goto %s;\n", (unsigned)(e - f.entry), label(e).c_str());
            fappend(ftext, "    default: break;\n    }\n");
        } else {
            fappend(ftext, "    (void)entry;\n");
        }
        std::set<uint64_t> emitted;
        // Blocks in address order; a block runs until a transfer or the next block start.
        for (auto it = f.blocks.begin(); it != f.blocks.end(); ++it) {
            uint64_t a = *it;
            auto nx = std::next(it);
            const uint64_t limit = nx == f.blocks.end() ? UINT64_MAX : *nx;
            fappend(ftext, "%s:\n", label(a).c_str());
            bool falls = true;
            while (a < limit) {
                ZydisDecodedInstruction insn_;
                ZydisDecodedOperand ops_[ZYDIS_MAX_OPERAND_COUNT];
                if (!dec.decode(img, a, insn_, ops_)) {
                    fappend(ftext, "    vp_unsupported(cpu, %s, \"undecodable\"); return;\n", hex(a).c_str());
                    falls = false;
                    break;
                }
                rip = a;
                next = a + insn_.length;
                insn = &insn_;
                ops = ops_;
                body.clear();
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
                fappend(ftext, "    /* %s: %s */\n    {\n%s    }\n", hex(a).c_str(), text, body.c_str());
                if (is_jcc(insn_.mnemonic) && insn_.mnemonic != ZYDIS_MNEMONIC_JCXZ && insn_.mnemonic != ZYDIS_MNEMONIC_JECXZ &&
                    insn_.mnemonic != ZYDIS_MNEMONIC_JRCXZ) {
                    const uint64_t t = branch_target(insn_, ops_, a);
                    fappend(ftext, "    if (vp_cc(cpu, %d)) goto %s;\n", cc_of(insn_.mnemonic), label(t).c_str());
                }
                a = next;
                if (!cont) { falls = false; break; }
            }
            if (falls && limit != UINT64_MAX && a >= limit) {
                // Fall-through into the next block in address order: nothing to emit.
            } else if (falls) {
                fappend(ftext, "    cpu->rip = %s; vp_dispatch(cpu, cpu->rip); return;\n", hex(a).c_str());
            }
        }
        fappend(ftext, "}\n\n");
        if (opt.locals) {
            std::string r;
            size_t pos = 0;
            for (;;) {
                const size_t k = ftext.find("return;", pos);
                if (k == std::string::npos) { r += ftext.substr(pos); break; }
                r += ftext.substr(pos, k - pos) + "{ *cpu_in = L; return; }";
                pos = k + 7;
            }
            ftext = r;
        }
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
    if (split) {
        FILE* h = fopen(header_name.c_str(), "w");
        if (!h) throw std::runtime_error("cannot write " + header_name);
        fprintf(h, "/* Generated by vpaot. Do not edit. */\n#pragma once\n#include \"vp_cpu.h\"\n#include \"vp_emit.h\"\n\n");
        for (auto& [a, fn] : functions) fprintf(h, "void %s(VpCpu* cpu, uint32_t entry);\n", fname(a).c_str());
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
            fprintf(f, "/* Generated by vpaot. Do not edit. */\n#include \"vp_cpu.h\"\n#include \"vp_emit.h\"\n\n");
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
    fprintf(f, "const VpEntry vp_entries[] = {\n");
    for (auto& [a, fn] : functions) fprintf(f, "    { %s, %s },\n", hex(a).c_str(), fname(a).c_str());
    fprintf(f, "};\nconst size_t vp_entry_count = %zu;\n", functions.size());
    fclose(f);
    if (split) {
        // A list of the units for the build system.
        FILE* l = fopen((stem + "_files.txt").c_str(), "w");
        if (l) { for (auto& w : written) fprintf(l, "%s\n", w.c_str()); fclose(l); }
    }
}

} // namespace vpaot
