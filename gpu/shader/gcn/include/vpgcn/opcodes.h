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

// Vector ALU signature: "<operands>[/<VOP3 modifiers>]". Operands, in printed order: dst then
// sources. A digit = VGPR dst / vector source of that many dwords; '-' = no dst; C = carry-out or
// compare result (vcc in VOP1/2/C form, an SGPR pair in VOP3); c = carry-in / select (vcc, or an
// SGPR pair in VOP3); s = SGPR dst, or an 8-bit scalar source; S = scalar source incl. literal;
// v = VGPR or lds_direct; V = VGPR only; k = 32-bit literal constant; + / M = implicit vcc / m0
// read (both use the constant bus). The '/' part exists
// iff the op also has a VOP3 form: per source a = abs+neg, n = neg, - = none; then c = clamp,
// o = output modifier.
#define VPGCN_VOP2(X)                                                                               \
    X(v_cndmask_b32, 0, "111c/aa-") X(v_readlane_b32, 1, "svs") X(v_writelane_b32, 2, "1Ss")        \
    X(v_add_f32, 3, "111/aaco") X(v_sub_f32, 4, "111/aaco") X(v_subrev_f32, 5, "111/aaco")          \
    X(v_mac_legacy_f32, 6, "111/aaco") X(v_mul_legacy_f32, 7, "111/aaco")                           \
    X(v_mul_f32, 8, "111/aaco") X(v_mul_i32_i24, 9, "111/--") X(v_mul_hi_i32_i24, 10, "111/--")     \
    X(v_mul_u32_u24, 11, "111/--") X(v_mul_hi_u32_u24, 12, "111/--")                                \
    X(v_min_legacy_f32, 13, "111/aaco") X(v_max_legacy_f32, 14, "111/aaco")                         \
    X(v_min_f32, 15, "111/aaco") X(v_max_f32, 16, "111/aaco") X(v_min_i32, 17, "111/--")            \
    X(v_max_i32, 18, "111/--") X(v_min_u32, 19, "111/--") X(v_max_u32, 20, "111/--")                \
    X(v_lshr_b32, 21, "111/--") X(v_lshrrev_b32, 22, "111/--") X(v_ashr_i32, 23, "111/--")          \
    X(v_ashrrev_i32, 24, "111/--") X(v_lshl_b32, 25, "111/--") X(v_lshlrev_b32, 26, "111/--")       \
    X(v_and_b32, 27, "111/--") X(v_or_b32, 28, "111/--") X(v_xor_b32, 29, "111/--")                 \
    X(v_bfm_b32, 30, "111/--") X(v_mac_f32, 31, "111/aaco") X(v_madmk_f32, 32, "11k1")              \
    X(v_madak_f32, 33, "111k") X(v_bcnt_u32_b32, 34, "111/--")                                      \
    X(v_mbcnt_lo_u32_b32, 35, "111/--") X(v_mbcnt_hi_u32_b32, 36, "111/--")                         \
    X(v_add_i32, 37, "1C11/--") X(v_sub_i32, 38, "1C11/--") X(v_subrev_i32, 39, "1C11/--")          \
    X(v_addc_u32, 40, "1C11c/---") X(v_subb_u32, 41, "1C11c/---")                                   \
    X(v_subbrev_u32, 42, "1C11c/---") X(v_ldexp_f32, 43, "111/a-co")                                \
    X(v_cvt_pkaccum_u8_f32, 44, "111/a-") X(v_cvt_pknorm_i16_f32, 45, "111/aa")                     \
    X(v_cvt_pknorm_u16_f32, 46, "111/aa") X(v_cvt_pkrtz_f16_f32, 47, "111/aaco")                    \
    X(v_cvt_pk_u16_u32, 48, "111/--") X(v_cvt_pk_i16_i32, 49, "111/--")

#define VPGCN_VOP1(X)                                                                               \
    X(v_nop, 0, "-/") X(v_mov_b32, 1, "11/-") X(v_readfirstlane_b32, 2, "sv")                       \
    X(v_cvt_i32_f64, 3, "12/ao") X(v_cvt_f64_i32, 4, "21/-co") X(v_cvt_f32_i32, 5, "11/-co")        \
    X(v_cvt_f32_u32, 6, "11/-co") X(v_cvt_u32_f32, 7, "11/ao") X(v_cvt_i32_f32, 8, "11/ao")         \
    X(v_cvt_f16_f32, 10, "11/aco") X(v_cvt_f32_f16, 11, "11/aco")                                   \
    X(v_cvt_rpi_i32_f32, 12, "11/a") X(v_cvt_flr_i32_f32, 13, "11/a")                               \
    X(v_cvt_off_f32_i4, 14, "11/-co") X(v_cvt_f32_f64, 15, "12/aco")                                \
    X(v_cvt_f64_f32, 16, "21/aco") X(v_cvt_f32_ubyte0, 17, "11/-co")                                \
    X(v_cvt_f32_ubyte1, 18, "11/-co") X(v_cvt_f32_ubyte2, 19, "11/-co")                             \
    X(v_cvt_f32_ubyte3, 20, "11/-co") X(v_cvt_u32_f64, 21, "12/ao")                                 \
    X(v_cvt_f64_u32, 22, "21/-co") X(v_trunc_f64, 23, "22/aco") X(v_ceil_f64, 24, "22/aco")         \
    X(v_rndne_f64, 25, "22/aco") X(v_floor_f64, 26, "22/aco") X(v_fract_f32, 32, "11/aco")          \
    X(v_trunc_f32, 33, "11/aco") X(v_ceil_f32, 34, "11/aco") X(v_rndne_f32, 35, "11/aco")           \
    X(v_floor_f32, 36, "11/aco") X(v_exp_f32, 37, "11/aco") X(v_log_clamp_f32, 38, "11/aco")        \
    X(v_log_f32, 39, "11/aco") X(v_rcp_clamp_f32, 40, "11/aco") X(v_rcp_legacy_f32, 41, "11/aco")   \
    X(v_rcp_f32, 42, "11/aco") X(v_rcp_iflag_f32, 43, "11/aco") X(v_rsq_clamp_f32, 44, "11/aco")    \
    X(v_rsq_legacy_f32, 45, "11/aco") X(v_rsq_f32, 46, "11/aco") X(v_rcp_f64, 47, "22/aco")         \
    X(v_rcp_clamp_f64, 48, "22/aco") X(v_rsq_f64, 49, "22/aco") X(v_rsq_clamp_f64, 50, "22/aco")    \
    X(v_sqrt_f32, 51, "11/aco") X(v_sqrt_f64, 52, "22/aco") X(v_sin_f32, 53, "11/aco")              \
    X(v_cos_f32, 54, "11/aco") X(v_not_b32, 55, "11/-") X(v_bfrev_b32, 56, "11/-")                  \
    X(v_ffbh_u32, 57, "11/-") X(v_ffbl_b32, 58, "11/-") X(v_ffbh_i32, 59, "11/-")                   \
    X(v_frexp_exp_i32_f64, 60, "12/ao") X(v_frexp_mant_f64, 61, "22/aco")                           \
    X(v_fract_f64, 62, "22/aco") X(v_frexp_exp_i32_f32, 63, "11/a")                                 \
    X(v_frexp_mant_f32, 64, "11/aco") X(v_clrexcp, 65, "-/") X(v_movreld_b32, 66, "11M/-")           \
    X(v_movrels_b32, 67, "1VM/-") X(v_movrelsd_b32, 68, "1VM/-") X(v_log_legacy_f32, 69, "11/aco")    \
    X(v_exp_legacy_f32, 70, "11/aco")

#define VPGCN_VOPC(X)                                                                               \
    X(v_cmp_f_f32, 0, "-C11/aa") X(v_cmp_lt_f32, 1, "-C11/aa") X(v_cmp_eq_f32, 2, "-C11/aa")        \
    X(v_cmp_le_f32, 3, "-C11/aa") X(v_cmp_gt_f32, 4, "-C11/aa") X(v_cmp_lg_f32, 5, "-C11/aa")       \
    X(v_cmp_ge_f32, 6, "-C11/aa") X(v_cmp_o_f32, 7, "-C11/aa") X(v_cmp_u_f32, 8, "-C11/aa")         \
    X(v_cmp_nge_f32, 9, "-C11/aa") X(v_cmp_nlg_f32, 10, "-C11/aa")                                  \
    X(v_cmp_ngt_f32, 11, "-C11/aa") X(v_cmp_nle_f32, 12, "-C11/aa")                                 \
    X(v_cmp_neq_f32, 13, "-C11/aa") X(v_cmp_nlt_f32, 14, "-C11/aa")                                 \
    X(v_cmp_tru_f32, 15, "-C11/aa") X(v_cmpx_f_f32, 16, "-C11/aa")                                  \
    X(v_cmpx_lt_f32, 17, "-C11/aa") X(v_cmpx_eq_f32, 18, "-C11/aa")                                 \
    X(v_cmpx_le_f32, 19, "-C11/aa") X(v_cmpx_gt_f32, 20, "-C11/aa")                                 \
    X(v_cmpx_lg_f32, 21, "-C11/aa") X(v_cmpx_ge_f32, 22, "-C11/aa")                                 \
    X(v_cmpx_o_f32, 23, "-C11/aa") X(v_cmpx_u_f32, 24, "-C11/aa")                                   \
    X(v_cmpx_nge_f32, 25, "-C11/aa") X(v_cmpx_nlg_f32, 26, "-C11/aa")                               \
    X(v_cmpx_ngt_f32, 27, "-C11/aa") X(v_cmpx_nle_f32, 28, "-C11/aa")                               \
    X(v_cmpx_neq_f32, 29, "-C11/aa") X(v_cmpx_nlt_f32, 30, "-C11/aa")                               \
    X(v_cmpx_tru_f32, 31, "-C11/aa") X(v_cmp_f_f64, 32, "-C22/aa") X(v_cmp_lt_f64, 33, "-C22/aa")   \
    X(v_cmp_eq_f64, 34, "-C22/aa") X(v_cmp_le_f64, 35, "-C22/aa") X(v_cmp_gt_f64, 36, "-C22/aa")    \
    X(v_cmp_lg_f64, 37, "-C22/aa") X(v_cmp_ge_f64, 38, "-C22/aa") X(v_cmp_o_f64, 39, "-C22/aa")     \
    X(v_cmp_u_f64, 40, "-C22/aa") X(v_cmp_nge_f64, 41, "-C22/aa") X(v_cmp_nlg_f64, 42, "-C22/aa")   \
    X(v_cmp_ngt_f64, 43, "-C22/aa") X(v_cmp_nle_f64, 44, "-C22/aa")                                 \
    X(v_cmp_neq_f64, 45, "-C22/aa") X(v_cmp_nlt_f64, 46, "-C22/aa")                                 \
    X(v_cmp_tru_f64, 47, "-C22/aa") X(v_cmpx_f_f64, 48, "-C22/aa")                                  \
    X(v_cmpx_lt_f64, 49, "-C22/aa") X(v_cmpx_eq_f64, 50, "-C22/aa")                                 \
    X(v_cmpx_le_f64, 51, "-C22/aa") X(v_cmpx_gt_f64, 52, "-C22/aa")                                 \
    X(v_cmpx_lg_f64, 53, "-C22/aa") X(v_cmpx_ge_f64, 54, "-C22/aa")                                 \
    X(v_cmpx_o_f64, 55, "-C22/aa") X(v_cmpx_u_f64, 56, "-C22/aa")                                   \
    X(v_cmpx_nge_f64, 57, "-C22/aa") X(v_cmpx_nlg_f64, 58, "-C22/aa")                               \
    X(v_cmpx_ngt_f64, 59, "-C22/aa") X(v_cmpx_nle_f64, 60, "-C22/aa")                               \
    X(v_cmpx_neq_f64, 61, "-C22/aa") X(v_cmpx_nlt_f64, 62, "-C22/aa")                               \
    X(v_cmpx_tru_f64, 63, "-C22/aa") X(v_cmps_f_f32, 64, "-C11/aa")                                 \
    X(v_cmps_lt_f32, 65, "-C11/aa") X(v_cmps_eq_f32, 66, "-C11/aa")                                 \
    X(v_cmps_le_f32, 67, "-C11/aa") X(v_cmps_gt_f32, 68, "-C11/aa")                                 \
    X(v_cmps_lg_f32, 69, "-C11/aa") X(v_cmps_ge_f32, 70, "-C11/aa")                                 \
    X(v_cmps_o_f32, 71, "-C11/aa") X(v_cmps_u_f32, 72, "-C11/aa")                                   \
    X(v_cmps_nge_f32, 73, "-C11/aa") X(v_cmps_nlg_f32, 74, "-C11/aa")                               \
    X(v_cmps_ngt_f32, 75, "-C11/aa") X(v_cmps_nle_f32, 76, "-C11/aa")                               \
    X(v_cmps_neq_f32, 77, "-C11/aa") X(v_cmps_nlt_f32, 78, "-C11/aa")                               \
    X(v_cmps_tru_f32, 79, "-C11/aa") X(v_cmpsx_f_f32, 80, "-C11/aa")                                \
    X(v_cmpsx_lt_f32, 81, "-C11/aa") X(v_cmpsx_eq_f32, 82, "-C11/aa")                               \
    X(v_cmpsx_le_f32, 83, "-C11/aa") X(v_cmpsx_gt_f32, 84, "-C11/aa")                               \
    X(v_cmpsx_lg_f32, 85, "-C11/aa") X(v_cmpsx_ge_f32, 86, "-C11/aa")                               \
    X(v_cmpsx_o_f32, 87, "-C11/aa") X(v_cmpsx_u_f32, 88, "-C11/aa")                                 \
    X(v_cmpsx_nge_f32, 89, "-C11/aa") X(v_cmpsx_nlg_f32, 90, "-C11/aa")                             \
    X(v_cmpsx_ngt_f32, 91, "-C11/aa") X(v_cmpsx_nle_f32, 92, "-C11/aa")                             \
    X(v_cmpsx_neq_f32, 93, "-C11/aa") X(v_cmpsx_nlt_f32, 94, "-C11/aa")                             \
    X(v_cmpsx_tru_f32, 95, "-C11/aa") X(v_cmps_f_f64, 96, "-C22/aa")                                \
    X(v_cmps_lt_f64, 97, "-C22/aa") X(v_cmps_eq_f64, 98, "-C22/aa")                                 \
    X(v_cmps_le_f64, 99, "-C22/aa") X(v_cmps_gt_f64, 100, "-C22/aa")                                \
    X(v_cmps_lg_f64, 101, "-C22/aa") X(v_cmps_ge_f64, 102, "-C22/aa")                               \
    X(v_cmps_o_f64, 103, "-C22/aa") X(v_cmps_u_f64, 104, "-C22/aa")                                 \
    X(v_cmps_nge_f64, 105, "-C22/aa") X(v_cmps_nlg_f64, 106, "-C22/aa")                             \
    X(v_cmps_ngt_f64, 107, "-C22/aa") X(v_cmps_nle_f64, 108, "-C22/aa")                             \
    X(v_cmps_neq_f64, 109, "-C22/aa") X(v_cmps_nlt_f64, 110, "-C22/aa")                             \
    X(v_cmps_tru_f64, 111, "-C22/aa") X(v_cmpsx_f_f64, 112, "-C22/aa")                              \
    X(v_cmpsx_lt_f64, 113, "-C22/aa") X(v_cmpsx_eq_f64, 114, "-C22/aa")                             \
    X(v_cmpsx_le_f64, 115, "-C22/aa") X(v_cmpsx_gt_f64, 116, "-C22/aa")                             \
    X(v_cmpsx_lg_f64, 117, "-C22/aa") X(v_cmpsx_ge_f64, 118, "-C22/aa")                             \
    X(v_cmpsx_o_f64, 119, "-C22/aa") X(v_cmpsx_u_f64, 120, "-C22/aa")                               \
    X(v_cmpsx_nge_f64, 121, "-C22/aa") X(v_cmpsx_nlg_f64, 122, "-C22/aa")                           \
    X(v_cmpsx_ngt_f64, 123, "-C22/aa") X(v_cmpsx_nle_f64, 124, "-C22/aa")                           \
    X(v_cmpsx_neq_f64, 125, "-C22/aa") X(v_cmpsx_nlt_f64, 126, "-C22/aa")                           \
    X(v_cmpsx_tru_f64, 127, "-C22/aa") X(v_cmp_f_i32, 128, "-C11/--")                               \
    X(v_cmp_lt_i32, 129, "-C11/--") X(v_cmp_eq_i32, 130, "-C11/--")                                 \
    X(v_cmp_le_i32, 131, "-C11/--") X(v_cmp_gt_i32, 132, "-C11/--")                                 \
    X(v_cmp_ne_i32, 133, "-C11/--") X(v_cmp_ge_i32, 134, "-C11/--")                                 \
    X(v_cmp_t_i32, 135, "-C11/--") X(v_cmp_class_f32, 136, "-C11/a-")                               \
    X(v_cmpx_f_i32, 144, "-C11/--") X(v_cmpx_lt_i32, 145, "-C11/--")                                \
    X(v_cmpx_eq_i32, 146, "-C11/--") X(v_cmpx_le_i32, 147, "-C11/--")                               \
    X(v_cmpx_gt_i32, 148, "-C11/--") X(v_cmpx_ne_i32, 149, "-C11/--")                               \
    X(v_cmpx_ge_i32, 150, "-C11/--") X(v_cmpx_t_i32, 151, "-C11/--")                                \
    X(v_cmpx_class_f32, 152, "-C11/a-") X(v_cmp_f_i64, 160, "-C22/--")                              \
    X(v_cmp_lt_i64, 161, "-C22/--") X(v_cmp_eq_i64, 162, "-C22/--")                                 \
    X(v_cmp_le_i64, 163, "-C22/--") X(v_cmp_gt_i64, 164, "-C22/--")                                 \
    X(v_cmp_ne_i64, 165, "-C22/--") X(v_cmp_ge_i64, 166, "-C22/--")                                 \
    X(v_cmp_t_i64, 167, "-C22/--") X(v_cmp_class_f64, 168, "-C21/a-")                               \
    X(v_cmpx_f_i64, 176, "-C22/--") X(v_cmpx_lt_i64, 177, "-C22/--")                                \
    X(v_cmpx_eq_i64, 178, "-C22/--") X(v_cmpx_le_i64, 179, "-C22/--")                               \
    X(v_cmpx_gt_i64, 180, "-C22/--") X(v_cmpx_ne_i64, 181, "-C22/--")                               \
    X(v_cmpx_ge_i64, 182, "-C22/--") X(v_cmpx_t_i64, 183, "-C22/--")                                \
    X(v_cmpx_class_f64, 184, "-C21/a-") X(v_cmp_f_u32, 192, "-C11/--")                              \
    X(v_cmp_lt_u32, 193, "-C11/--") X(v_cmp_eq_u32, 194, "-C11/--")                                 \
    X(v_cmp_le_u32, 195, "-C11/--") X(v_cmp_gt_u32, 196, "-C11/--")                                 \
    X(v_cmp_ne_u32, 197, "-C11/--") X(v_cmp_ge_u32, 198, "-C11/--")                                 \
    X(v_cmp_t_u32, 199, "-C11/--") X(v_cmpx_f_u32, 208, "-C11/--")                                  \
    X(v_cmpx_lt_u32, 209, "-C11/--") X(v_cmpx_eq_u32, 210, "-C11/--")                               \
    X(v_cmpx_le_u32, 211, "-C11/--") X(v_cmpx_gt_u32, 212, "-C11/--")                               \
    X(v_cmpx_ne_u32, 213, "-C11/--") X(v_cmpx_ge_u32, 214, "-C11/--")                               \
    X(v_cmpx_t_u32, 215, "-C11/--") X(v_cmp_f_u64, 224, "-C22/--")                                  \
    X(v_cmp_lt_u64, 225, "-C22/--") X(v_cmp_eq_u64, 226, "-C22/--")                                 \
    X(v_cmp_le_u64, 227, "-C22/--") X(v_cmp_gt_u64, 228, "-C22/--")                                 \
    X(v_cmp_ne_u64, 229, "-C22/--") X(v_cmp_ge_u64, 230, "-C22/--")                                 \
    X(v_cmp_t_u64, 231, "-C22/--") X(v_cmpx_f_u64, 240, "-C22/--")                                  \
    X(v_cmpx_lt_u64, 241, "-C22/--") X(v_cmpx_eq_u64, 242, "-C22/--")                               \
    X(v_cmpx_le_u64, 243, "-C22/--") X(v_cmpx_gt_u64, 244, "-C22/--")                               \
    X(v_cmpx_ne_u64, 245, "-C22/--") X(v_cmpx_ge_u64, 246, "-C22/--")                               \
    X(v_cmpx_t_u64, 247, "-C22/--")

#define VPGCN_VOP3(X)                                                                               \
    X(v_mad_legacy_f32, 320, "1111/aaaco") X(v_mad_f32, 321, "1111/aaaco")                          \
    X(v_mad_i32_i24, 322, "1111/---") X(v_mad_u32_u24, 323, "1111/---")                             \
    X(v_cubeid_f32, 324, "1111/aaaco") X(v_cubesc_f32, 325, "1111/aaaco")                           \
    X(v_cubetc_f32, 326, "1111/aaaco") X(v_cubema_f32, 327, "1111/aaaco")                           \
    X(v_bfe_u32, 328, "1111/---") X(v_bfe_i32, 329, "1111/---") X(v_bfi_b32, 330, "1111/---")       \
    X(v_fma_f32, 331, "1111/aaaco") X(v_fma_f64, 332, "2222/aaaco") X(v_lerp_u8, 333, "1111/---")   \
    X(v_alignbit_b32, 334, "1111/---") X(v_alignbyte_b32, 335, "1111/---")                          \
    X(v_mullit_f32, 336, "1111/aaaco") X(v_min3_f32, 337, "1111/aaaco")                             \
    X(v_min3_i32, 338, "1111/---") X(v_min3_u32, 339, "1111/---")                                   \
    X(v_max3_f32, 340, "1111/aaaco") X(v_max3_i32, 341, "1111/---")                                 \
    X(v_max3_u32, 342, "1111/---") X(v_med3_f32, 343, "1111/aaaco")                                 \
    X(v_med3_i32, 344, "1111/---") X(v_med3_u32, 345, "1111/---") X(v_sad_u8, 346, "1111/---")      \
    X(v_sad_hi_u8, 347, "1111/---") X(v_sad_u16, 348, "1111/---") X(v_sad_u32, 349, "1111/---")     \
    X(v_cvt_pk_u8_f32, 350, "1111/a--") X(v_div_fixup_f32, 351, "1111/aaaco")                       \
    X(v_div_fixup_f64, 352, "2222/aaaco") X(v_lshl_b64, 353, "221/--")                              \
    X(v_lshr_b64, 354, "221/--") X(v_ashr_i64, 355, "221/--") X(v_add_f64, 356, "222/aaco")         \
    X(v_mul_f64, 357, "222/aaco") X(v_min_f64, 358, "222/aaco") X(v_max_f64, 359, "222/aaco")       \
    X(v_ldexp_f64, 360, "221/a-co") X(v_mul_lo_u32, 361, "111/--") X(v_mul_hi_u32, 362, "111/--")   \
    X(v_mul_lo_i32, 363, "111/--") X(v_mul_hi_i32, 364, "111/--")                                   \
    X(v_div_scale_f32, 365, "1C111/nnno") X(v_div_scale_f64, 366, "2C222/nnno")                     \
    X(v_div_fmas_f32, 367, "1111+/aaaco") X(v_div_fmas_f64, 368, "2222+/aaaco")                     \
    X(v_msad_u8, 369, "1111/---") X(v_qsad_pk_u16_u8, 370, "2212/---")                              \
    X(v_mqsad_pk_u16_u8, 371, "2212/---") X(v_trig_preop_f64, 372, "221/a-co")                      \
    X(v_mqsad_u32_u8, 373, "4214/---") X(v_mad_u64_u32, 374, "2C112/---")

// ---- Memory, export and interpolation encodings ---------------------------------------------------
//
// DS signature: "<dst><addr><data0><data1><offset>": dword counts ('-' = absent; the address is one
// VGPR), then o = one 16-bit offset, 2 = two 8-bit offsets (offset0/offset1), z = swizzle pattern.
// MUBUF / MTBUF / FLAT signature: L = load, S = store, A = atomic (vdata also returns with glc),
// N = no operands; then vdata's dword count (cmpswap carries compare + swap value).
// GWS ops (ds_gws_init, _sema_br, _barrier) take their value in the address field here, as llvm-mc
// encodes them for gfx7; graphics shaders do not use GWS.
// MIMG signature: kind L load, W store, A atomic, R resinfo, S sample, G gather4, O get_lod; the
// address VGPR count of the op's 2D, non-array form (the real count depends on the image's
// dimension, known only from its descriptor: the lifter recomputes it); 's' = takes a sampler;
// C = compare-swap (data = 2 x dmask). Data dwords: popcount(dmask) (gather4: 4), + 1 with tfe.

#define VPGCN_DS(X) \
    X(ds_add_u32, 0, "-11-o") X(ds_sub_u32, 1, "-11-o") X(ds_rsub_u32, 2, "-11-o")                 \
    X(ds_inc_u32, 3, "-11-o") X(ds_dec_u32, 4, "-11-o") X(ds_min_i32, 5, "-11-o")                  \
    X(ds_max_i32, 6, "-11-o") X(ds_min_u32, 7, "-11-o") X(ds_max_u32, 8, "-11-o")                  \
    X(ds_and_b32, 9, "-11-o") X(ds_or_b32, 10, "-11-o") X(ds_xor_b32, 11, "-11-o")                 \
    X(ds_mskor_b32, 12, "-111o") X(ds_write_b32, 13, "-11-o") X(ds_write2_b32, 14, "-1112")        \
    X(ds_write2st64_b32, 15, "-1112") X(ds_cmpst_b32, 16, "-111o") X(ds_cmpst_f32, 17, "-111o")    \
    X(ds_min_f32, 18, "-11-o") X(ds_max_f32, 19, "-11-o") X(ds_nop, 20, "----o")                   \
    X(ds_gws_sema_release_all, 24, "----o") X(ds_gws_init, 25, "-1--o")                            \
    X(ds_gws_sema_v, 26, "----o") X(ds_gws_sema_br, 27, "-1--o") X(ds_gws_sema_p, 28, "----o")     \
    X(ds_gws_barrier, 29, "-1--o") X(ds_write_b8, 30, "-11-o") X(ds_write_b16, 31, "-11-o")        \
    X(ds_add_rtn_u32, 32, "111-o") X(ds_sub_rtn_u32, 33, "111-o") X(ds_rsub_rtn_u32, 34, "111-o")  \
    X(ds_inc_rtn_u32, 35, "111-o") X(ds_dec_rtn_u32, 36, "111-o") X(ds_min_rtn_i32, 37, "111-o")   \
    X(ds_max_rtn_i32, 38, "111-o") X(ds_min_rtn_u32, 39, "111-o") X(ds_max_rtn_u32, 40, "111-o")   \
    X(ds_and_rtn_b32, 41, "111-o") X(ds_or_rtn_b32, 42, "111-o") X(ds_xor_rtn_b32, 43, "111-o")    \
    X(ds_mskor_rtn_b32, 44, "1111o") X(ds_wrxchg_rtn_b32, 45, "111-o")                             \
    X(ds_wrxchg2_rtn_b32, 46, "21112") X(ds_wrxchg2st64_rtn_b32, 47, "21112")                      \
    X(ds_cmpst_rtn_b32, 48, "1111o") X(ds_cmpst_rtn_f32, 49, "1111o")                              \
    X(ds_min_rtn_f32, 50, "111-o") X(ds_max_rtn_f32, 51, "111-o") X(ds_wrap_rtn_b32, 52, "1111o")  \
    X(ds_swizzle_b32, 53, "11--z") X(ds_read_b32, 54, "11--o") X(ds_read2_b32, 55, "21--2")        \
    X(ds_read2st64_b32, 56, "21--2") X(ds_read_i8, 57, "11--o") X(ds_read_u8, 58, "11--o")         \
    X(ds_read_i16, 59, "11--o") X(ds_read_u16, 60, "11--o") X(ds_consume, 61, "1---o")             \
    X(ds_append, 62, "1---o") X(ds_ordered_count, 63, "11--o") X(ds_add_u64, 64, "-12-o")          \
    X(ds_sub_u64, 65, "-12-o") X(ds_rsub_u64, 66, "-12-o") X(ds_inc_u64, 67, "-12-o")              \
    X(ds_dec_u64, 68, "-12-o") X(ds_min_i64, 69, "-12-o") X(ds_max_i64, 70, "-12-o")               \
    X(ds_min_u64, 71, "-12-o") X(ds_max_u64, 72, "-12-o") X(ds_and_b64, 73, "-12-o")               \
    X(ds_or_b64, 74, "-12-o") X(ds_xor_b64, 75, "-12-o") X(ds_mskor_b64, 76, "-122o")              \
    X(ds_write_b64, 77, "-12-o") X(ds_write2_b64, 78, "-1222") X(ds_write2st64_b64, 79, "-1222")   \
    X(ds_cmpst_b64, 80, "-122o") X(ds_cmpst_f64, 81, "-122o") X(ds_min_f64, 82, "-12-o")           \
    X(ds_max_f64, 83, "-12-o") X(ds_add_rtn_u64, 96, "212-o") X(ds_sub_rtn_u64, 97, "212-o")       \
    X(ds_rsub_rtn_u64, 98, "212-o") X(ds_inc_rtn_u64, 99, "212-o") X(ds_dec_rtn_u64, 100, "212-o") \
    X(ds_min_rtn_i64, 101, "212-o") X(ds_max_rtn_i64, 102, "212-o")                                \
    X(ds_min_rtn_u64, 103, "212-o") X(ds_max_rtn_u64, 104, "212-o")                                \
    X(ds_and_rtn_b64, 105, "212-o") X(ds_or_rtn_b64, 106, "212-o") X(ds_xor_rtn_b64, 107, "212-o") \
    X(ds_mskor_rtn_b64, 108, "2122o") X(ds_wrxchg_rtn_b64, 109, "212-o")                           \
    X(ds_wrxchg2_rtn_b64, 110, "41222") X(ds_wrxchg2st64_rtn_b64, 111, "41222")                    \
    X(ds_cmpst_rtn_b64, 112, "2122o") X(ds_cmpst_rtn_f64, 113, "2122o")                            \
    X(ds_min_rtn_f64, 114, "212-o") X(ds_max_rtn_f64, 115, "212-o") X(ds_read_b64, 118, "21--o")   \
    X(ds_read2_b64, 119, "41--2") X(ds_read2st64_b64, 120, "41--2")                                \
    X(ds_condxchg32_rtn_b64, 126, "212-o") X(ds_add_src2_u32, 128, "-1--o")                        \
    X(ds_sub_src2_u32, 129, "-1--o") X(ds_rsub_src2_u32, 130, "-1--o")                             \
    X(ds_inc_src2_u32, 131, "-1--o") X(ds_dec_src2_u32, 132, "-1--o")                              \
    X(ds_min_src2_i32, 133, "-1--o") X(ds_max_src2_i32, 134, "-1--o")                              \
    X(ds_min_src2_u32, 135, "-1--o") X(ds_max_src2_u32, 136, "-1--o")                              \
    X(ds_and_src2_b32, 137, "-1--o") X(ds_or_src2_b32, 138, "-1--o")                               \
    X(ds_xor_src2_b32, 139, "-1--o") X(ds_write_src2_b32, 141, "-1--o")                            \
    X(ds_min_src2_f32, 146, "-1--o") X(ds_max_src2_f32, 147, "-1--o")                              \
    X(ds_add_src2_u64, 192, "-1--o") X(ds_sub_src2_u64, 193, "-1--o")                              \
    X(ds_rsub_src2_u64, 194, "-1--o") X(ds_inc_src2_u64, 195, "-1--o")                             \
    X(ds_dec_src2_u64, 196, "-1--o") X(ds_min_src2_i64, 197, "-1--o")                              \
    X(ds_max_src2_i64, 198, "-1--o") X(ds_min_src2_u64, 199, "-1--o")                              \
    X(ds_max_src2_u64, 200, "-1--o") X(ds_and_src2_b64, 201, "-1--o")                              \
    X(ds_or_src2_b64, 202, "-1--o") X(ds_xor_src2_b64, 203, "-1--o")                               \
    X(ds_write_src2_b64, 205, "-1--o") X(ds_min_src2_f64, 210, "-1--o")                            \
    X(ds_max_src2_f64, 211, "-1--o") X(ds_write_b96, 222, "-13-o") X(ds_write_b128, 223, "-14-o")  \
    X(ds_read_b96, 254, "31--o") X(ds_read_b128, 255, "41--o")

#define VPGCN_MUBUF(X) \
    X(buffer_load_format_x, 0, "L1") X(buffer_load_format_xy, 1, "L2")                             \
    X(buffer_load_format_xyz, 2, "L3") X(buffer_load_format_xyzw, 3, "L4")                         \
    X(buffer_store_format_x, 4, "S1") X(buffer_store_format_xy, 5, "S2")                           \
    X(buffer_store_format_xyz, 6, "S3") X(buffer_store_format_xyzw, 7, "S4")                       \
    X(buffer_load_ubyte, 8, "L1") X(buffer_load_sbyte, 9, "L1") X(buffer_load_ushort, 10, "L1")    \
    X(buffer_load_sshort, 11, "L1") X(buffer_load_dword, 12, "L1")                                 \
    X(buffer_load_dwordx2, 13, "L2") X(buffer_load_dwordx4, 14, "L4")                              \
    X(buffer_load_dwordx3, 15, "L3") X(buffer_store_byte, 24, "S1")                                \
    X(buffer_store_short, 26, "S1") X(buffer_store_dword, 28, "S1")                                \
    X(buffer_store_dwordx2, 29, "S2") X(buffer_store_dwordx4, 30, "S4")                            \
    X(buffer_store_dwordx3, 31, "S3") X(buffer_atomic_swap, 48, "A1")                              \
    X(buffer_atomic_cmpswap, 49, "A2") X(buffer_atomic_add, 50, "A1")                              \
    X(buffer_atomic_sub, 51, "A1") X(buffer_atomic_smin, 53, "A1") X(buffer_atomic_umin, 54, "A1") \
    X(buffer_atomic_smax, 55, "A1") X(buffer_atomic_umax, 56, "A1") X(buffer_atomic_and, 57, "A1") \
    X(buffer_atomic_or, 58, "A1") X(buffer_atomic_xor, 59, "A1") X(buffer_atomic_inc, 60, "A1")    \
    X(buffer_atomic_dec, 61, "A1") X(buffer_atomic_fcmpswap, 62, "A2")                             \
    X(buffer_atomic_fmin, 63, "A1") X(buffer_atomic_fmax, 64, "A1")                                \
    X(buffer_atomic_swap_x2, 80, "A2") X(buffer_atomic_cmpswap_x2, 81, "A4")                       \
    X(buffer_atomic_add_x2, 82, "A2") X(buffer_atomic_sub_x2, 83, "A2")                            \
    X(buffer_atomic_smin_x2, 85, "A2") X(buffer_atomic_umin_x2, 86, "A2")                          \
    X(buffer_atomic_smax_x2, 87, "A2") X(buffer_atomic_umax_x2, 88, "A2")                          \
    X(buffer_atomic_and_x2, 89, "A2") X(buffer_atomic_or_x2, 90, "A2")                             \
    X(buffer_atomic_xor_x2, 91, "A2") X(buffer_atomic_inc_x2, 92, "A2")                            \
    X(buffer_atomic_dec_x2, 93, "A2") X(buffer_atomic_fcmpswap_x2, 94, "A4")                       \
    X(buffer_atomic_fmin_x2, 95, "A2") X(buffer_atomic_fmax_x2, 96, "A2")                          \
    X(buffer_wbinvl1_vol, 112, "N0") X(buffer_wbinvl1, 113, "N0")

#define VPGCN_MTBUF(X) \
    X(tbuffer_load_format_x, 0, "L1") X(tbuffer_load_format_xy, 1, "L2")                           \
    X(tbuffer_load_format_xyz, 2, "L3") X(tbuffer_load_format_xyzw, 3, "L4")                       \
    X(tbuffer_store_format_x, 4, "S1") X(tbuffer_store_format_xy, 5, "S2")                         \
    X(tbuffer_store_format_xyz, 6, "S3") X(tbuffer_store_format_xyzw, 7, "S4")

#define VPGCN_MIMG(X) \
    X(image_load, 0, "L2-") X(image_load_mip, 1, "L3-") X(image_load_pck, 2, "L2-")                \
    X(image_load_pck_sgn, 3, "L2-") X(image_load_mip_pck, 4, "L3-")                                \
    X(image_load_mip_pck_sgn, 5, "L3-") X(image_store, 8, "W2-") X(image_store_mip, 9, "W3-")      \
    X(image_store_pck, 10, "W2-") X(image_store_mip_pck, 11, "W3-")                                \
    X(image_get_resinfo, 14, "R1-") X(image_atomic_swap, 15, "A2-")                                \
    X(image_atomic_cmpswap, 16, "A2-C") X(image_atomic_add, 17, "A2-")                             \
    X(image_atomic_sub, 18, "A2-") X(image_atomic_rsub, 19, "A2-") X(image_atomic_smin, 20, "A2-") \
    X(image_atomic_umin, 21, "A2-") X(image_atomic_smax, 22, "A2-")                                \
    X(image_atomic_umax, 23, "A2-") X(image_atomic_and, 24, "A2-") X(image_atomic_or, 25, "A2-")   \
    X(image_atomic_xor, 26, "A2-") X(image_atomic_inc, 27, "A2-") X(image_atomic_dec, 28, "A2-")   \
    X(image_atomic_fcmpswap, 29, "A2-C") X(image_atomic_fmin, 30, "A2-")                           \
    X(image_atomic_fmax, 31, "A2-") X(image_sample, 32, "S2s") X(image_sample_cl, 33, "S3s")       \
    X(image_sample_d, 34, "S6s") X(image_sample_d_cl, 35, "S7s") X(image_sample_l, 36, "S3s")      \
    X(image_sample_b, 37, "S3s") X(image_sample_b_cl, 38, "S4s") X(image_sample_lz, 39, "S2s")     \
    X(image_sample_c, 40, "S3s") X(image_sample_c_cl, 41, "S4s") X(image_sample_c_d, 42, "S7s")    \
    X(image_sample_c_d_cl, 43, "S8s") X(image_sample_c_l, 44, "S4s")                               \
    X(image_sample_c_b, 45, "S4s") X(image_sample_c_b_cl, 46, "S5s")                               \
    X(image_sample_c_lz, 47, "S3s") X(image_sample_o, 48, "S3s") X(image_sample_cl_o, 49, "S4s")   \
    X(image_sample_d_o, 50, "S7s") X(image_sample_d_cl_o, 51, "S8s")                               \
    X(image_sample_l_o, 52, "S4s") X(image_sample_b_o, 53, "S4s")                                  \
    X(image_sample_b_cl_o, 54, "S5s") X(image_sample_lz_o, 55, "S3s")                              \
    X(image_sample_c_o, 56, "S4s") X(image_sample_c_cl_o, 57, "S5s")                               \
    X(image_sample_c_d_o, 58, "S8s") X(image_sample_c_d_cl_o, 59, "S9s")                           \
    X(image_sample_c_l_o, 60, "S5s") X(image_sample_c_b_o, 61, "S5s")                              \
    X(image_sample_c_b_cl_o, 62, "S6s") X(image_sample_c_lz_o, 63, "S4s")                          \
    X(image_gather4, 64, "G2s") X(image_gather4_cl, 65, "G3s") X(image_gather4_l, 68, "G3s")       \
    X(image_gather4_b, 69, "G3s") X(image_gather4_b_cl, 70, "G4s") X(image_gather4_lz, 71, "G2s")  \
    X(image_gather4_c, 72, "G3s") X(image_gather4_c_cl, 73, "G4s") X(image_gather4_c_l, 76, "G4s") \
    X(image_gather4_c_b, 77, "G4s") X(image_gather4_c_b_cl, 78, "G5s")                             \
    X(image_gather4_c_lz, 79, "G3s") X(image_gather4_o, 80, "G3s")                                 \
    X(image_gather4_cl_o, 81, "G4s") X(image_gather4_l_o, 84, "G4s")                               \
    X(image_gather4_b_o, 85, "G4s") X(image_gather4_b_cl_o, 86, "G5s")                             \
    X(image_gather4_lz_o, 87, "G3s") X(image_gather4_c_o, 88, "G4s")                               \
    X(image_gather4_c_cl_o, 89, "G5s") X(image_gather4_c_l_o, 92, "G5s")                           \
    X(image_gather4_c_b_o, 93, "G5s") X(image_gather4_c_b_cl_o, 94, "G6s")                         \
    X(image_gather4_c_lz_o, 95, "G4s") X(image_get_lod, 96, "O2s") X(image_sample_cd, 104, "S6s")  \
    X(image_sample_cd_cl, 105, "S7s") X(image_sample_c_cd, 106, "S7s")                             \
    X(image_sample_c_cd_cl, 107, "S8s") X(image_sample_cd_o, 108, "S7s")                           \
    X(image_sample_cd_cl_o, 109, "S8s") X(image_sample_c_cd_o, 110, "S8s")                         \
    X(image_sample_c_cd_cl_o, 111, "S9s")

#define VPGCN_FLAT(X) \
    X(flat_load_ubyte, 8, "L1") X(flat_load_sbyte, 9, "L1") X(flat_load_ushort, 10, "L1")          \
    X(flat_load_sshort, 11, "L1") X(flat_load_dword, 12, "L1") X(flat_load_dwordx2, 13, "L2")      \
    X(flat_load_dwordx4, 14, "L4") X(flat_load_dwordx3, 15, "L3") X(flat_store_byte, 24, "S1")     \
    X(flat_store_short, 26, "S1") X(flat_store_dword, 28, "S1") X(flat_store_dwordx2, 29, "S2")    \
    X(flat_store_dwordx4, 30, "S4") X(flat_store_dwordx3, 31, "S3") X(flat_atomic_swap, 48, "A1")  \
    X(flat_atomic_cmpswap, 49, "A2") X(flat_atomic_add, 50, "A1") X(flat_atomic_sub, 51, "A1")     \
    X(flat_atomic_smin, 53, "A1") X(flat_atomic_umin, 54, "A1") X(flat_atomic_smax, 55, "A1")      \
    X(flat_atomic_umax, 56, "A1") X(flat_atomic_and, 57, "A1") X(flat_atomic_or, 58, "A1")         \
    X(flat_atomic_xor, 59, "A1") X(flat_atomic_inc, 60, "A1") X(flat_atomic_dec, 61, "A1")         \
    X(flat_atomic_fcmpswap, 62, "A2") X(flat_atomic_fmin, 63, "A1") X(flat_atomic_fmax, 64, "A1")  \
    X(flat_atomic_swap_x2, 80, "A2") X(flat_atomic_cmpswap_x2, 81, "A4")                           \
    X(flat_atomic_add_x2, 82, "A2") X(flat_atomic_sub_x2, 83, "A2")                                \
    X(flat_atomic_smin_x2, 85, "A2") X(flat_atomic_umin_x2, 86, "A2")                              \
    X(flat_atomic_smax_x2, 87, "A2") X(flat_atomic_umax_x2, 88, "A2")                              \
    X(flat_atomic_and_x2, 89, "A2") X(flat_atomic_or_x2, 90, "A2") X(flat_atomic_xor_x2, 91, "A2") \
    X(flat_atomic_inc_x2, 92, "A2") X(flat_atomic_dec_x2, 93, "A2")                                \
    X(flat_atomic_fcmpswap_x2, 94, "A4") X(flat_atomic_fmin_x2, 95, "A2")                          \
    X(flat_atomic_fmax_x2, 96, "A2")

#define VPGCN_EXP(X) X(exp, 0, "")

// VINTRP: p1/p2 read a VGPR (the barycentric), mov reads a parameter (p10, p20, p0).
#define VPGCN_VINTRP(X) X(v_interp_p1_f32, 0, "v") X(v_interp_p2_f32, 1, "v") X(v_interp_mov_f32, 2, "p")
