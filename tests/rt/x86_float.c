/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * x86 floating point behaviour that must not depend on the host (an Intel one, arm64, ...), checked
 * against models written with integers only:
 *  - NaN selection and sign (vp_fadd32 & co.): the first operand if it is a NaN, else the second,
 *    always quieted; an invalid operation without a NaN gives the negative QNaN indefinite.
 *  - RCPPS / RSQRTPS (vp_rcp_f32, vp_rsqrt_f32): values measured on an Intel CPU, special cases.
 *  - MXCSR.DAZ / FTZ applied by vp_apply_mxcsr.
 *  - The MXCSR_MASK that FXSAVE reports: 0x2FFFF is deliberate (see vp_x87_fxsave).
 */
#include "vp_emit.h"
#include <stdio.h>

static int bad;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL " __VA_ARGS__); printf("\n"); bad = 1; } } while (0)

static float f32(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t b32(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static double f64(uint64_t b) { double f; memcpy(&f, &b, 8); return f; }
static uint64_t b64(double f) { uint64_t b; memcpy(&b, &f, 8); return b; }

static int is_nan32(uint32_t b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
static int is_nan64(uint64_t b) { return (b & UINT64_C(0x7FFFFFFFFFFFFFFF)) > UINT64_C(0x7FF0000000000000); }

/* The model: NaN operands only (the result of two non-NaN operands is the host's, IEEE). */
static uint32_t model32(uint32_t a, uint32_t b) { return (is_nan32(a) ? a : b) | 0x400000u; }
static uint64_t model64(uint64_t a, uint64_t b) { return (is_nan64(a) ? a : b) | UINT64_C(0x0008000000000000); }

static const uint32_t v32[] = {0, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x7F800000u, 0xFF800000u, 0x00000001u, 0x7F7FFFFFu,
                               0x7FC00000u, 0xFFC00000u, 0x7FC00123u, 0xFFC00456u, 0x7F800001u, 0xFF800789u, 0x7FA00000u, 0xFFBFFFFFu};
static const uint64_t v64[] = {0, UINT64_C(0x8000000000000000), UINT64_C(0x3FF0000000000000), UINT64_C(0xBFF0000000000000), UINT64_C(0x7FF0000000000000),
                               UINT64_C(0xFFF0000000000000), 1, UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x7FF8000000000000), UINT64_C(0xFFF8000000000000),
                               UINT64_C(0x7FF8000000000123), UINT64_C(0xFFF8000000000456), UINT64_C(0x7FF0000000000001), UINT64_C(0xFFF0000000000789),
                               UINT64_C(0x7FF4000000000000), UINT64_C(0xFFF7FFFFFFFFFFFF)};

int main(void) {
    typedef float (*op32)(float, float);
    typedef double (*op64)(double, double);
    const op32 o32[4] = {vp_fadd32, vp_fsub32, vp_fmul32, vp_fdiv32};
    const op64 o64[4] = {vp_fadd64, vp_fsub64, vp_fmul64, vp_fdiv64};
    for (int op = 0; op < 4; ++op) {
        for (unsigned i = 0; i < sizeof v32 / 4; ++i)
            for (unsigned j = 0; j < sizeof v32 / 4; ++j) {
                const uint32_t a = v32[i], b = v32[j], r = b32(o32[op](f32(a), f32(b)));
                if (is_nan32(a) || is_nan32(b)) CHECK(r == model32(a, b), "f32 op%d %08x %08x -> %08x, want %08x", op, a, b, r, model32(a, b));
                else if (is_nan32(r)) CHECK(r == 0xFFC00000u, "f32 op%d %08x %08x invalid -> %08x, want the indefinite", op, a, b, r);
            }
        for (unsigned i = 0; i < sizeof v64 / 8; ++i)
            for (unsigned j = 0; j < sizeof v64 / 8; ++j) {
                const uint64_t a = v64[i], b = v64[j], r = b64(o64[op](f64(a), f64(b)));
                if (is_nan64(a) || is_nan64(b)) CHECK(r == model64(a, b), "f64 op%d %016llx %016llx -> %016llx", op, (unsigned long long)a, (unsigned long long)b, (unsigned long long)r);
                else if (is_nan64(r)) CHECK(r == UINT64_C(0xFFF8000000000000), "f64 op%d %016llx %016llx invalid -> %016llx", op, (unsigned long long)a, (unsigned long long)b, (unsigned long long)r);
            }
    }
    /* sqrt: a NaN is quieted, a negative number gives the indefinite, -0 stays. */
    CHECK(b32(vp_fsqrt32(f32(0x7F800001u))) == 0x7FC00001u, "sqrtss snan");
    CHECK(b32(vp_fsqrt32(f32(0xFF800789u))) == 0xFFC00789u, "sqrtss -snan");
    CHECK(b32(vp_fsqrt32(f32(0xBF800000u))) == 0xFFC00000u, "sqrtss -1");
    CHECK(b32(vp_fsqrt32(f32(0x80000000u))) == 0x80000000u, "sqrtss -0");
    CHECK(b64(vp_fsqrt64(f64(UINT64_C(0xBFF0000000000000)))) == UINT64_C(0xFFF8000000000000), "sqrtsd -1");
    CHECK(b64(vp_fsqrt64(f64(UINT64_C(0x7FF0000000000001)))) == UINT64_C(0x7FF8000000000001), "sqrtsd snan");
    CHECK(b32(vp_fquiet32(f32(0x7F800001u))) == 0x7FC00001u && b32(vp_fquiet32(f32(0x3F800000u))) == 0x3F800000u, "quiet32");
    CHECK(b64(vp_fquiet64(f64(UINT64_C(0xFFF0000000000001)))) == UINT64_C(0xFFF8000000000001), "quiet64");

    /* RCPPS / RSQRTPS, Intel: values measured with the instructions. */
    CHECK(b32(vp_rcp_f32(f32(0x3F800000u))) == 0x3F7FF000u, "rcp 1.0 = %08x", b32(vp_rcp_f32(f32(0x3F800000u))));
    CHECK(b32(vp_rcp_f32(f32(0x40000000u))) == 0x3EFFF000u, "rcp 2.0 = %08x", b32(vp_rcp_f32(f32(0x40000000u))));
    CHECK(b32(vp_rsqrt_f32(f32(0x3F800000u))) == 0x3F7FF000u, "rsqrt 1.0 = %08x", b32(vp_rsqrt_f32(f32(0x3F800000u))));
    CHECK(b32(vp_rcp_f32(f32(0))) == 0x7F800000u && b32(vp_rcp_f32(f32(0x80000000u))) == 0xFF800000u, "rcp +-0");
    CHECK(b32(vp_rcp_f32(f32(1))) == 0x7F800000u, "rcp of a denormal is infinity");
    CHECK(b32(vp_rcp_f32(f32(0x7F800000u))) == 0 && b32(vp_rcp_f32(f32(0xFF800000u))) == 0x80000000u, "rcp +-inf");
    CHECK(b32(vp_rcp_f32(f32(0x7F000000u))) == 0, "rcp of a huge number: the denormal result is 0, got %08x", b32(vp_rcp_f32(f32(0x7F000000u))));
    CHECK(b32(vp_rsqrt_f32(f32(0xBF800000u))) == 0xFFC00000u && b32(vp_rsqrt_f32(f32(0xFF800000u))) == 0xFFC00000u, "rsqrt of a negative number");
    CHECK(b32(vp_rsqrt_f32(f32(0x80000000u))) == 0xFF800000u && b32(vp_rsqrt_f32(f32(0))) == 0x7F800000u, "rsqrt +-0");
    CHECK(b32(vp_rsqrt_f32(f32(0x7F800000u))) == 0, "rsqrt +inf");
    CHECK(b32(vp_rcp_f32(f32(0x7F800001u))) == 0x7FC00001u, "rcp snan");

    /* MXCSR.DAZ / FTZ reach the host: 1.0e-38 * 0.5 is denormal. */
    VpCpu c;
    memset(&c, 0, sizeof c);
    volatile float tiny = f32(0x00800000u), half = 0.5f;
    c.mxcsr = 0x1F80;
    vp_apply_mxcsr(&c);
    CHECK(b32(tiny * half) == 0x00400000u, "denormal result without FTZ");
    c.mxcsr = 0x1F80 | 0x8000;
    vp_apply_mxcsr(&c);
    CHECK(b32(tiny * half) == 0, "FTZ flushes a denormal result, got %08x", b32(tiny * half));
    c.mxcsr = 0x1F80 | 0x40 | 0x8000;
    vp_apply_mxcsr(&c);
    volatile float den = f32(0x00000001u), one = 1.0f;
    CHECK(b32(den + one) == 0x3F800000u && b32(den * 2.0f) == 0, "DAZ+FTZ: a denormal operand is zero, got %08x", b32(den * 2.0f));
    c.mxcsr = 0x1F80;
    vp_apply_mxcsr(&c);
    CHECK(b32(tiny * half) == 0x00400000u, "denormal result again without FTZ");

    /* FXSAVE's MXCSR_MASK is 0x2FFFF (Jaguar: MM, the misaligned SSE bit), not Intel's 0xFFFF: a decision of the project. */
    uint8_t area[512] __attribute__((aligned(16)));
    memset(&c, 0, sizeof c);
    vp_x87_fxsave(&c, (uint64_t)(uintptr_t)area);
    uint32_t mask;
    memcpy(&mask, area + 28, 4);
    CHECK(mask == 0x2FFFFu, "MXCSR_MASK %#x", mask);

    printf(bad ? "x86_float: FAIL\n" : "x86_float: ok\n");
    return bad;
}
