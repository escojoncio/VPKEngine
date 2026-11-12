/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * VP_RCP_MODEL: how RCPPS/RCPSS/RSQRTPS/RSQRTSS (vp_rcp_f32, vp_rsqrt_f32) are computed.
 *   VP_RCP_INTEL (default): the 12-bit tables of the Intel CPUs, bit for bit.
 *   VP_RCP_EXACT: 1/x and 1/sqrt(x) in single precision (what the runtime did before the tables).
 * Built twice by run_local.sh: without the macro and with -DVP_RCP_MODEL=VP_RCP_EXACT.
 */
#include "vp_cpu.h"
#include <stdio.h>

static int bad;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL " __VA_ARGS__); printf("\n"); bad = 1; } } while (0)
static float f32(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t b32(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

int main(void) {
#if VP_RCP_MODEL == VP_RCP_EXACT
    static const uint32_t in[] = {0x3F800000u, 0x40000000u, 0x40400000u, 0x40A00000u, 0x3DCCCCCDu, 0x42C80000u, 0x7F000000u, 0x00800000u,
                                  0x80800000u, 0xC0400000u, 0x4B189680u, 0x3F9E3779u, 0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x00000001u};
    for (unsigned i = 0; i < sizeof in / sizeof *in; ++i) {
        const float x = f32(in[i]);
        CHECK(b32(vp_rcp_f32(x)) == b32(1.0f / x), "exact rcp(%08x) = %08x", in[i], b32(vp_rcp_f32(x)));
        CHECK(b32(vp_rsqrt_f32(x)) == b32(1.0f / sqrtf(x)), "exact rsqrt(%08x) = %08x", in[i], b32(vp_rsqrt_f32(x)));
    }
    CHECK(b32(vp_rcp_f32(3.0f)) == 0x3EAAAAABu, "exact rcp(3) is the correctly rounded 1/3");
    /* an Intel table value: 12 bits, 1.0 gives 0x3F7FF000, not 1.0 */
    CHECK(b32(vp_rcp_f32(1.0f)) == 0x3F800000u, "exact rcp(1) = 1");
#else
    CHECK(b32(vp_rcp_f32(f32(0x3F800000u))) == 0x3F7FF000u, "intel rcp(1) = %08x", b32(vp_rcp_f32(1.0f)));
    CHECK(b32(vp_rsqrt_f32(f32(0x3F800000u))) == 0x3F7FF000u, "intel rsqrt(1) = %08x", b32(vp_rsqrt_f32(1.0f)));
    CHECK(b32(vp_rcp_f32(3.0f)) != 0x3EAAAAABu, "intel rcp(3) is an approximation, not 1/3");
    CHECK(b32(vp_rcp_f32(f32(0x00000001u))) == 0x7F800000u, "intel rcp of a denormal is infinity");
#endif
    printf(bad ? "rcp_model FAILED (VP_RCP_MODEL=%d)\n" : "ok   rcp_model (VP_RCP_MODEL=%d)\n", VP_RCP_MODEL);
    return bad;
}
