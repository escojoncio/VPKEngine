// GCN 2 (Sea Islands / gfx7) opcode tables, per encoding, from AMD's public Sea Islands ISA
// reference.
//
// X(mnemonic, opcode, signature). Scalar ALU signature: one character per operand (dst, src0, src1),
// '-' = absent, a digit = size in dwords of an operand that takes registers, inline constants and
// literals; 'k' = 64-bit without literals; 'r' = 32-bit registers and src_vccz/execz/scc only;
// 'R' = 64-bit registers only.
#pragma once

#define VPGCN_SOP2(X)                                                                              \
    X(s_add_u32, 0, "111") X(s_sub_u32, 1, "111") X(s_add_i32, 2, "111") X(s_sub_i32, 3, "111")    \
    X(s_addc_u32, 4, "111") X(s_subb_u32, 5, "111") X(s_min_i32, 6, "111")                         \
    X(s_min_u32, 7, "111") X(s_max_i32, 8, "111") X(s_max_u32, 9, "111")                           \
    X(s_cselect_b32, 10, "111") X(s_cselect_b64, 11, "222") X(s_and_b32, 14, "111")               \
    X(s_and_b64, 15, "222") X(s_or_b32, 16, "111") X(s_or_b64, 17, "222")                          \
    X(s_xor_b32, 18, "111") X(s_xor_b64, 19, "222") X(s_andn2_b32, 20, "111")                      \
    X(s_andn2_b64, 21, "222") X(s_orn2_b32, 22, "111") X(s_orn2_b64, 23, "222")                    \
    X(s_nand_b32, 24, "111") X(s_nand_b64, 25, "222") X(s_nor_b32, 26, "111")                      \
    X(s_nor_b64, 27, "222") X(s_xnor_b32, 28, "111") X(s_xnor_b64, 29, "222")                      \
    X(s_lshl_b32, 30, "111") X(s_lshl_b64, 31, "221") X(s_lshr_b32, 32, "111")                     \
    X(s_lshr_b64, 33, "221") X(s_ashr_i32, 34, "111") X(s_ashr_i64, 35, "221")                     \
    X(s_bfm_b32, 36, "111") X(s_bfm_b64, 37, "211") X(s_mul_i32, 38, "111")                        \
    X(s_bfe_u32, 39, "111") X(s_bfe_i32, 40, "111") X(s_bfe_u64, 41, "221")                        \
    X(s_bfe_i64, 42, "221") X(s_cbranch_g_fork, 43, "-kk") X(s_absdiff_i32, 44, "111")

#define VPGCN_SOP1(X)                                                                              \
    X(s_mov_b32, 3, "11") X(s_mov_b64, 4, "22") X(s_cmov_b32, 5, "11") X(s_cmov_b64, 6, "22")      \
    X(s_not_b32, 7, "11") X(s_not_b64, 8, "22") X(s_wqm_b32, 9, "11") X(s_wqm_b64, 10, "22")       \
    X(s_brev_b32, 11, "11") X(s_brev_b64, 12, "22") X(s_bcnt0_i32_b32, 13, "11")                   \
    X(s_bcnt0_i32_b64, 14, "12") X(s_bcnt1_i32_b32, 15, "11") X(s_bcnt1_i32_b64, 16, "12")         \
    X(s_ff0_i32_b32, 17, "11") X(s_ff0_i32_b64, 18, "12") X(s_ff1_i32_b32, 19, "11")               \
    X(s_ff1_i32_b64, 20, "12") X(s_flbit_i32_b32, 21, "11") X(s_flbit_i32_b64, 22, "12")           \
    X(s_flbit_i32, 23, "11") X(s_flbit_i32_i64, 24, "12") X(s_sext_i32_i8, 25, "11")               \
    X(s_sext_i32_i16, 26, "11") X(s_bitset0_b32, 27, "11") X(s_bitset0_b64, 28, "21")              \
    X(s_bitset1_b32, 29, "11") X(s_bitset1_b64, 30, "21") X(s_getpc_b64, 31, "2-")                 \
    X(s_setpc_b64, 32, "-R") X(s_swappc_b64, 33, "22") X(s_rfe_b64, 34, "-R")                      \
    X(s_and_saveexec_b64, 36, "22") X(s_or_saveexec_b64, 37, "22")                                 \
    X(s_xor_saveexec_b64, 38, "22") X(s_andn2_saveexec_b64, 39, "22")                              \
    X(s_orn2_saveexec_b64, 40, "22") X(s_nand_saveexec_b64, 41, "22")                              \
    X(s_nor_saveexec_b64, 42, "22") X(s_xnor_saveexec_b64, 43, "22")                               \
    X(s_quadmask_b32, 44, "11") X(s_quadmask_b64, 45, "22") X(s_movrels_b32, 46, "1r")             \
    X(s_movrels_b64, 47, "2R") X(s_movreld_b32, 48, "11") X(s_movreld_b64, 49, "22")               \
    X(s_cbranch_join, 50, "-r") X(s_abs_i32, 52, "11")

#define VPGCN_SOPC(X)                                                                              \
    X(s_cmp_eq_i32, 0, "-11") X(s_cmp_lg_i32, 1, "-11") X(s_cmp_gt_i32, 2, "-11")                  \
    X(s_cmp_ge_i32, 3, "-11") X(s_cmp_lt_i32, 4, "-11") X(s_cmp_le_i32, 5, "-11")                  \
    X(s_cmp_eq_u32, 6, "-11") X(s_cmp_lg_u32, 7, "-11") X(s_cmp_gt_u32, 8, "-11")                  \
    X(s_cmp_ge_u32, 9, "-11") X(s_cmp_lt_u32, 10, "-11") X(s_cmp_le_u32, 11, "-11")                \
    X(s_bitcmp0_b32, 12, "-11") X(s_bitcmp1_b32, 13, "-11") X(s_bitcmp0_b64, 14, "-21")            \
    X(s_bitcmp1_b64, 15, "-21") X(s_setvskip, 16, "-11")

// SOPK signature letter: m = movk-style hex imm, k = cmpk/addk, h = getreg, H = setreg,
// I = setreg_imm32 (literal follows), b = cbranch_i_fork.
#define VPGCN_SOPK(X)                                                                              \
    X(s_movk_i32, 0, "m") X(s_cmovk_i32, 2, "m") X(s_cmpk_eq_i32, 3, "k")                          \
    X(s_cmpk_lg_i32, 4, "k") X(s_cmpk_gt_i32, 5, "k") X(s_cmpk_ge_i32, 6, "k")                     \
    X(s_cmpk_lt_i32, 7, "k") X(s_cmpk_le_i32, 8, "k") X(s_cmpk_eq_u32, 9, "k")                     \
    X(s_cmpk_lg_u32, 10, "k") X(s_cmpk_gt_u32, 11, "k") X(s_cmpk_ge_u32, 12, "k")                  \
    X(s_cmpk_lt_u32, 13, "k") X(s_cmpk_le_u32, 14, "k") X(s_addk_i32, 15, "k")                     \
    X(s_mulk_i32, 16, "k") X(s_cbranch_i_fork, 17, "b") X(s_getreg_b32, 18, "h")                   \
    X(s_setreg_b32, 19, "H") X(s_setreg_imm32_b32, 21, "I")

// SOPP signature letter: - = no operand (simm16 must be 0), e = optional decimal imm, d = imm,
// j = branch offset, w = waitcnt, m = sendmsg.
#define VPGCN_SOPP(X)                                                                              \
    X(s_nop, 0, "d") X(s_endpgm, 1, "e") X(s_branch, 2, "j") X(s_cbranch_scc0, 4, "j")             \
    X(s_cbranch_scc1, 5, "j") X(s_cbranch_vccz, 6, "j") X(s_cbranch_vccnz, 7, "j")                 \
    X(s_cbranch_execz, 8, "j") X(s_cbranch_execnz, 9, "j") X(s_barrier, 10, "-")                   \
    X(s_waitcnt, 12, "w") X(s_sethalt, 13, "d") X(s_sleep, 14, "d") X(s_setprio, 15, "d")          \
    X(s_sendmsg, 16, "m") X(s_sendmsghalt, 17, "m") X(s_trap, 18, "d") X(s_icache_inv, 19, "-")    \
    X(s_incperflevel, 20, "d") X(s_decperflevel, 21, "d") X(s_ttracedata, 22, "-")                 \
    X(s_cbranch_cdbgsys, 23, "j") X(s_cbranch_cdbguser, 24, "j")                                   \
    X(s_cbranch_cdbgsys_or_user, 25, "j") X(s_cbranch_cdbgsys_and_user, 26, "j")

// SMRD signature: dst dwords, base dwords ('-' = absent; 'G' = 16).
#define VPGCN_SMRD(X)                                                                              \
    X(s_load_dword, 0, "12") X(s_load_dwordx2, 1, "22") X(s_load_dwordx4, 2, "42")                 \
    X(s_load_dwordx8, 3, "82") X(s_load_dwordx16, 4, "G2") X(s_buffer_load_dword, 8, "14")         \
    X(s_buffer_load_dwordx2, 9, "24") X(s_buffer_load_dwordx4, 10, "44")                           \
    X(s_buffer_load_dwordx8, 11, "84") X(s_buffer_load_dwordx16, 12, "G4")                         \
    X(s_dcache_inv_vol, 29, "--") X(s_memtime, 30, "2-") X(s_dcache_inv, 31, "--")
