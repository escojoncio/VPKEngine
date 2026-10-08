/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * General-register access for the generated code. Without VP_REGCACHE every access goes to the
 * state struct. With it, each function keeps the sixteen registers in locals (VP_DECL) and the
 * generated code writes them back (VP_OUT) before anything that reads the struct and reloads
 * them (VP_IN) after; VP_LOCAL, defined per instruction by the generated code, selects which
 * copy an access uses.
 */
#ifndef VP_REGS_H
#define VP_REGS_H

#include "vp_cpu.h"

#ifndef VP_REGCACHE
#define VP_REGCACHE 0
#endif
#ifndef VP_LOCAL
#define VP_LOCAL VP_REGCACHE
#endif

/* Register names to local names: VP_GR(3) and VP_GR(VP_RBX) are both g3. */
#define VP_GR_0 g0
#define VP_GR_1 g1
#define VP_GR_2 g2
#define VP_GR_3 g3
#define VP_GR_4 g4
#define VP_GR_5 g5
#define VP_GR_6 g6
#define VP_GR_7 g7
#define VP_GR_8 g8
#define VP_GR_9 g9
#define VP_GR_10 g10
#define VP_GR_11 g11
#define VP_GR_12 g12
#define VP_GR_13 g13
#define VP_GR_14 g14
#define VP_GR_15 g15
#define VP_GR_VP_RAX g0
#define VP_GR_VP_RCX g1
#define VP_GR_VP_RDX g2
#define VP_GR_VP_RBX g3
#define VP_GR_VP_RSP g4
#define VP_GR_VP_RBP g5
#define VP_GR_VP_RSI g6
#define VP_GR_VP_RDI g7
#define VP_GR__(i) VP_GR_##i
#define VP_GR(i) VP_GR__(i)

#if VP_REGCACHE
#define VP_DECL() uint64_t g0 = cpu->r[0], g1 = cpu->r[1], g2 = cpu->r[2], g3 = cpu->r[3], g4 = cpu->r[4], g5 = cpu->r[5], \
                  g6 = cpu->r[6], g7 = cpu->r[7], g8 = cpu->r[8], g9 = cpu->r[9], g10 = cpu->r[10], g11 = cpu->r[11], \
                  g12 = cpu->r[12], g13 = cpu->r[13], g14 = cpu->r[14], g15 = cpu->r[15]
#define VP_OUT() do { cpu->r[0] = g0; cpu->r[1] = g1; cpu->r[2] = g2; cpu->r[3] = g3; cpu->r[4] = g4; cpu->r[5] = g5; \
                      cpu->r[6] = g6; cpu->r[7] = g7; cpu->r[8] = g8; cpu->r[9] = g9; cpu->r[10] = g10; cpu->r[11] = g11; \
                      cpu->r[12] = g12; cpu->r[13] = g13; cpu->r[14] = g14; cpu->r[15] = g15; } while (0)
#define VP_IN() do { g0 = cpu->r[0]; g1 = cpu->r[1]; g2 = cpu->r[2]; g3 = cpu->r[3]; g4 = cpu->r[4]; g5 = cpu->r[5]; \
                     g6 = cpu->r[6]; g7 = cpu->r[7]; g8 = cpu->r[8]; g9 = cpu->r[9]; g10 = cpu->r[10]; g11 = cpu->r[11]; \
                     g12 = cpu->r[12]; g13 = cpu->r[13]; g14 = cpu->r[14]; g15 = cpu->r[15]; } while (0)
#define VP_REF(i) (*(VP_LOCAL ? &VP_GR(i) : &cpu->r[(i)]))
#else
#define VP_DECL() ((void)0)
#define VP_OUT() ((void)0)
#define VP_IN() ((void)0)
#define VP_REF(i) (cpu->r[(i)])
#endif

#define VP_R64(i) ((uint64_t)VP_REF(i))
#define VP_R32(i) ((uint32_t)VP_REF(i))
#define VP_R16(i) ((uint16_t)VP_REF(i))
#define VP_R8(i) ((uint8_t)VP_REF(i))
#define VP_R8H(i) ((uint8_t)(VP_REF(i) >> 8))
#define VP_W64(i, v) ((void)(VP_REF(i) = (uint64_t)(v)))
#define VP_W32(i, v) ((void)(VP_REF(i) = (uint32_t)(v)))
#define VP_W16(i, v) ((void)(VP_REF(i) = (VP_REF(i) & ~UINT64_C(0xffff)) | (uint16_t)(v)))
#define VP_W8(i, v) ((void)(VP_REF(i) = (VP_REF(i) & ~UINT64_C(0xff)) | (uint8_t)(v)))
#define VP_W8H(i, v) ((void)(VP_REF(i) = (VP_REF(i) & ~UINT64_C(0xff00)) | ((uint64_t)(uint8_t)(v) << 8)))

#endif
