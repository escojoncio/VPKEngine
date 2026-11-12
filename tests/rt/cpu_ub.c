/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * runtime/vp_cpu.h helpers with the inputs that reach undefined behaviour in C (build with
 * -fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all): shifts past the width of
 * the operand (SHL r8 by 9..31, as vpaot emits it), a conversion of a double that rounds out of
 * the 32-bit range, and the quotient limit of a signed division.
 */
#include "vp_emit.h"
#include <stdio.h>

static int bad;
#define CHECK(cond, what) do { if (!(cond)) { printf("FAIL %s\n", what); bad = 1; } } while (0)

int main(void) {
    VpCpu c;
    memset(&c, 0, sizeof c);

    /* SHL r8 / r16 by a count larger than the width: CF is architecturally undefined; any value, no UB. */
    volatile unsigned n9 = 9, n20 = 20, n31 = 31;
    c.cf = 1;
    vp_flags_shl(&c, 8, 0x81, n9, 0);
    CHECK(c.cf == 0, "shl8 by 9: CF is 0 (every bit has been shifted out)");
    vp_flags_shl(&c, 16, 0x8001, n20, 0);
    vp_flags_shl(&c, 16, 0x8001, n31, 0);
    vp_flags_shl(&c, 8, 0x80, 8, 0); /* a count equal to the width still gives the last bit out */
    CHECK(c.cf == 0, "shl8 by 8 of 0x80: bit 0 is out");
    vp_flags_shl(&c, 8, 0x81, 8, 0);
    CHECK(c.cf == 1, "shl8 by 8 of 0x81: bit 0 is out");

    /* CVTSD2SI: a value that rounds to 2^31 is out of range: the integer indefinite. */
    volatile double up = 2147483647.7, down = -2147483648.7, edge = 2147483647.0;
    CHECK(vp_cvt_f64_i32(up) == INT32_MIN, "cvtsd2si 2147483647.7");
    CHECK(vp_cvt_f64_i32(down) == INT32_MIN, "cvtsd2si -2147483648.7");
    CHECK(vp_cvt_f64_i32(edge) == 2147483647, "cvtsd2si 2147483647.0");
    CHECK(vp_cvt_f64_i32(-2147483648.4) == INT32_MIN, "cvtsd2si -2147483648.4");

    /* IDIV: INT_MIN / -1 is a #DE, in every width. */
    c.r[VP_RAX] = 0; c.r[VP_RDX] = UINT64_C(0x8000000000000000);
    CHECK(vp_div1(&c, 64, ~UINT64_C(0), 1) == 1, "idiv64 INT128_MIN / -1");
    c.r[VP_RAX] = 0x80000000u; c.r[VP_RDX] = 0xffffffffu;
    CHECK(vp_div1(&c, 32, 0xffffffffu, 1) == 1, "idiv32 INT64_MIN / -1");
    c.r[VP_RAX] = 0x8000; c.r[VP_RDX] = 0xffff;
    CHECK(vp_div1(&c, 16, 0xffff, 1) == 1, "idiv16 INT32_MIN / -1");
    c.r[VP_RAX] = 0x8000;
    CHECK(vp_div1(&c, 8, 0xff, 1) == 1, "idiv8 -32768 / -1");
    c.r[VP_RAX] = 7; c.r[VP_RDX] = 0;
    CHECK(vp_div1(&c, 64, 2, 1) == 0 && c.r[VP_RAX] == 3 && c.r[VP_RDX] == 1, "idiv64 7 / 2");

    if (!bad) printf("ok   cpu_ub\n");
    return bad;
}
