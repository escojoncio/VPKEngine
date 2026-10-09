// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// Every instruction form of the PS4's CPU (AMD Jaguar: x86-64 with x87, MMX, SSE to SSE4.2 and
// SSE4a, AVX, F16C, AES, PCLMUL, BMI1, LZCNT, POPCNT, MOVBE, XSAVE/XSAVEOPT), found by decoding
// every opcode of every map with every prefix, ModRM register field, register and memory operand,
// and VEX length, width and source register; one of each form (mnemonic, operand kinds and
// sizes, vector length) is kept. Writes:
//   code.bin   each form followed by a ret, at 16-byte steps
//   roots.txt  their offsets (vpaot --roots)
//   forms.txt  offset, ISA extension, mnemonic, the instruction (Intel syntax), its bytes (hex),
//              the RFLAGS bits it leaves undefined (hex), and "test" when a differential test can run it
//              alone (no transfer, stack, I/O, system, time or division; rsp and rdi not written)
// Memory operands are [rdi+0x40]; registers rax/rcx and xmm0/xmm1/xmm14.
//   enumerate OUT_DIR
#include <Zydis/Zydis.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

static bool jaguar(ZydisISAExt e) {
    switch (e) {
    case ZYDIS_ISA_EXT_BASE: case ZYDIS_ISA_EXT_LONGMODE: case ZYDIS_ISA_EXT_X87: case ZYDIS_ISA_EXT_MMX:
    case ZYDIS_ISA_EXT_SSE: case ZYDIS_ISA_EXT_SSE2: case ZYDIS_ISA_EXT_SSE3: case ZYDIS_ISA_EXT_SSSE3:
    case ZYDIS_ISA_EXT_SSE4: case ZYDIS_ISA_EXT_SSE4A: case ZYDIS_ISA_EXT_AVX: case ZYDIS_ISA_EXT_AVXAES:
    case ZYDIS_ISA_EXT_F16C: case ZYDIS_ISA_EXT_AES: case ZYDIS_ISA_EXT_PCLMULQDQ: case ZYDIS_ISA_EXT_BMI1:
    case ZYDIS_ISA_EXT_LZCNT: case ZYDIS_ISA_EXT_MOVBE: case ZYDIS_ISA_EXT_XSAVE: case ZYDIS_ISA_EXT_XSAVEOPT:
    case ZYDIS_ISA_EXT_CLFSH: case ZYDIS_ISA_EXT_PAUSE: case ZYDIS_ISA_EXT_RDTSCP: case ZYDIS_ISA_EXT_AMD3DNOW_PREFETCH:
        return true;
    default:
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: enumerate OUT_DIR\n"); return 2; }
    const std::string out = argv[1];
    ZydisDecoder decoder;
    ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    ZydisFormatter formatter;
    ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);

    std::set<std::string> seen;
    std::vector<uint8_t> code;
    FILE* roots = fopen((out + "/roots.txt").c_str(), "w");
    FILE* forms = fopen((out + "/forms.txt").c_str(), "w");
    if (!roots || !forms) { perror(out.c_str()); return 1; }
    std::map<std::string, int> per_ext;
    size_t decoded = 0;

    auto take = [&](const uint8_t* bytes, size_t avail) {
        ZydisDecodedInstruction insn;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        if (ZYAN_FAILED(ZydisDecoderDecodeFull(&decoder, bytes, avail, &insn, ops))) return;
        ++decoded;
        if (!jaguar(insn.meta.isa_ext) || (insn.attributes & ZYDIS_ATTRIB_IS_PRIVILEGED)) return;
        // One of each form: mnemonic, vector length, each visible operand's kind, size and register class.
        std::string key = std::string(ZydisMnemonicGetString(insn.mnemonic)) + "/" + std::to_string(insn.avx.vector_length);
        for (int i = 0; i < insn.operand_count_visible; ++i) {
            const auto& o = ops[i];
            key += " " + std::to_string(o.type) + ":" + std::to_string(o.size);
            if (o.type == ZYDIS_OPERAND_TYPE_REGISTER) key += ":" + std::to_string(ZydisRegisterGetClass(o.reg.value));
        }
        if (!seen.insert(key).second) return;
        while (code.size() % 16) code.push_back(0xcc);
        const size_t at = code.size();
        code.insert(code.end(), bytes, bytes + insn.length);
        code.push_back(0xc3);
        char text[256];
        ZydisFormatterFormatInstruction(&formatter, &insn, ops, insn.operand_count_visible, text, sizeof text, 0x400000 + at, nullptr);
        fprintf(roots, "0x%zx\n", at);
        bool testable = true;
        switch (insn.meta.category) {
        case ZYDIS_CATEGORY_COND_BR: case ZYDIS_CATEGORY_UNCOND_BR: case ZYDIS_CATEGORY_CALL: case ZYDIS_CATEGORY_RET:
        case ZYDIS_CATEGORY_SYSCALL: case ZYDIS_CATEGORY_INTERRUPT: case ZYDIS_CATEGORY_SYSTEM: case ZYDIS_CATEGORY_IO:
        case ZYDIS_CATEGORY_IOSTRINGOP: case ZYDIS_CATEGORY_X87_ALU: case ZYDIS_CATEGORY_SEGOP: case ZYDIS_CATEGORY_STRINGOP:
        case ZYDIS_CATEGORY_XSAVE: case ZYDIS_CATEGORY_XSAVEOPT: case ZYDIS_CATEGORY_POP: case ZYDIS_CATEGORY_PUSH:
        case ZYDIS_CATEGORY_FLAGOP: case ZYDIS_CATEGORY_MISC:
            testable = false; break;
        default: break;
        }
        switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_DIV: case ZYDIS_MNEMONIC_IDIV: case ZYDIS_MNEMONIC_RDTSC: case ZYDIS_MNEMONIC_RDTSCP:
        case ZYDIS_MNEMONIC_CPUID: case ZYDIS_MNEMONIC_RDPMC: case ZYDIS_MNEMONIC_XGETBV: case ZYDIS_MNEMONIC_ENTER:
        case ZYDIS_MNEMONIC_LEAVE: case ZYDIS_MNEMONIC_FXSAVE: case ZYDIS_MNEMONIC_FXSAVE64: case ZYDIS_MNEMONIC_FXRSTOR:
        case ZYDIS_MNEMONIC_FXRSTOR64: case ZYDIS_MNEMONIC_LDMXCSR: case ZYDIS_MNEMONIC_VLDMXCSR: case ZYDIS_MNEMONIC_EMMS:
            testable = false; break;
        default: break;
        }
        if (insn.meta.isa_ext == ZYDIS_ISA_EXT_X87) testable = false;
        for (int i = 0; i < insn.operand_count; ++i) {
            const auto& o = ops[i];
            if (o.type == ZYDIS_OPERAND_TYPE_REGISTER && (o.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE)) {
                const ZydisRegister big = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, o.reg.value);
                if (big == ZYDIS_REGISTER_RSP || big == ZYDIS_REGISTER_RDI || big == ZYDIS_REGISTER_RIP) testable = false;
            }
            if (o.type == ZYDIS_OPERAND_TYPE_REGISTER && ZydisRegisterGetClass(o.reg.value) == ZYDIS_REGCLASS_SEGMENT) testable = false;
        }
        std::string hex;
        for (int i = 0; i < insn.length; ++i) { char b[4]; snprintf(b, sizeof b, "%02x", bytes[i]); hex += b; }
        const unsigned undefined = insn.cpu_flags ? insn.cpu_flags->undefined : 0;
        fprintf(forms, "%zx\t%s\t%s\t%s\t%s\t%x\t%s\n", at, ZydisISAExtGetString(insn.meta.isa_ext), ZydisMnemonicGetString(insn.mnemonic), text,
                hex.c_str(), undefined, testable ? "test" : "-");
        per_ext[ZydisISAExtGetString(insn.meta.isa_ext)]++;
    };

    // ModRM: register forms with every reg field (group opcodes), and [rdi+0x40].
    std::vector<std::vector<uint8_t>> modrms;
    for (int r = 0; r < 8; ++r) {
        modrms.push_back({(uint8_t)(0xc0 | r << 3 | 1)});
        modrms.push_back({(uint8_t)(0x40 | r << 3 | 7), 0x40});
    }
    const uint8_t imm[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto emit = [&](const std::vector<uint8_t>& head, int op) {
        for (const auto& m : modrms) {
            std::vector<uint8_t> b = head;
            b.push_back((uint8_t)op);
            b.insert(b.end(), m.begin(), m.end());
            b.insert(b.end(), imm, imm + 8);
            take(b.data(), b.size());
        }
    };
    // Legacy encodings.
    const std::vector<std::vector<uint8_t>> prefixes = {{}, {0x66}, {0xf2}, {0xf3}};
    const std::vector<std::vector<uint8_t>> maps = {{}, {0x0f}, {0x0f, 0x38}, {0x0f, 0x3a}};
    for (const auto& p : prefixes)
        for (int rexw = 0; rexw < 2; ++rexw)
            for (const auto& mp : maps)
                for (int op = 0; op < 256; ++op) {
                    std::vector<uint8_t> h = p;
                    if (rexw) h.push_back(0x48);
                    h.insert(h.end(), mp.begin(), mp.end());
                    emit(h, op);
                }
    // VEX (three-byte form): map 1-3, pp, L, W, vvvv none / xmm1 / xmm14.
    for (int map = 1; map <= 3; ++map)
        for (int pp = 0; pp < 4; ++pp)
            for (int l = 0; l < 2; ++l)
                for (int w = 0; w < 2; ++w)
                    for (int v : {15, 14, 1})
                        for (int op = 0; op < 256; ++op) {
                            const std::vector<uint8_t> h = {0xc4, (uint8_t)(0xe0 | map), (uint8_t)(w << 7 | v << 3 | l << 2 | pp)};
                            emit(h, op);
                        }
    FILE* f = fopen((out + "/code.bin").c_str(), "wb");
    if (!f || fwrite(code.data(), 1, code.size(), f) != code.size()) { perror("code.bin"); return 1; }
    fclose(f);
    fclose(roots);
    fclose(forms);
    fprintf(stderr, "enumerate: %zu decoded, %zu forms of the Jaguar's instructions:", decoded, seen.size());
    for (auto& [e, n] : per_ext) fprintf(stderr, " %s %d", e.c_str(), n);
    fprintf(stderr, "\n");
    return 0;
}
