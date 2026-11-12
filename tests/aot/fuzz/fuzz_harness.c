/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Randomised differential harness (x86-64 host only). Like tests/aot/harness.c, but one code image holds
 * many one-instruction "slots" (the forms tools/isa_audit lists, each followed by a ret) and every slot
 * runs from many random start states, natively and translated, and the end states are compared.
 *
 *   fuzz_harness CODE.bin SLOTS.txt NSEEDS SEED0 [PROFILE_MASK [NAME_FILTER]]
 *
 * SLOTS.txt: one slot per line, "<offset from the base> <undefined-flags hex> <vector compare kind> <text>"
 * (kind: 0 exact, 32 / 64 NaN-insensitive lanes, -32 approximate single lanes; see vec_equal). The flags in
 * the mask (RFLAGS bit positions: cf 0, pf 2, af 4, zf 6, sf 7, of 11) are not compared.
 * The start state: general registers edge-biased, flags random, vector registers made of float / double /
 * small-integer edge values, MXCSR rounding random, DAZ / FTZ with profile bit 8 (all exceptions masked), x87 stack random.
 * Exit status 1 when any slot differs; each differing slot is reported once, with the first seed.
 */
#include "vp_emit.h"

#include <fenv.h>
#include <inttypes.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define VP_CODE_BASE  UINT64_C(0x400000)
#define VP_CODE_SIZE  UINT64_C(0x80000)
#define VP_STACK_BASE UINT64_C(0x7f000000)
#define VP_STACK_SIZE UINT64_C(0x10000)
#define VP_DATA_BASE  UINT64_C(0x600000)
#define VP_DATA_SIZE  UINT64_C(0x10000)
#define WINDOW        0x1000u /* the data bytes initialised and compared */
#define STACKLO       0x200u /* stack bytes below the initial rsp that are initialised and compared */
#define STACKWIN      0x300u

extern void vp_native_run(VpCpu* cpu, uint64_t code);
extern uint64_t vp_native_exit_address(void);

static void* map_fixed(uint64_t at, uint64_t size, int prot) {
    void* p = mmap((void*)(uintptr_t)at, size, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED || (uintptr_t)p != at) { fprintf(stderr, "cannot map %#" PRIx64 "\n", at); exit(3); }
    return p;
}

/* ---- random numbers -------------------------------------------------------------------------- */
static uint64_t rng_state;
static uint64_t rnd(void) {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static unsigned rn(unsigned n) { return (unsigned)(rnd() % n); }

static uint64_t edge64(void) {
    uint64_t v;
    switch (rn(20)) {
    case 0: v = 0; break;
    case 1: v = 1; break;
    case 2: v = ~UINT64_C(0); break;
    case 3: v = 0x7f; break;
    case 4: v = 0x80; break;
    case 5: v = 0xff; break;
    case 6: v = 0x7fff; break;
    case 7: v = 0x8000; break;
    case 8: v = 0xffff; break;
    case 9: v = 0x7fffffff; break;
    case 10: v = 0x80000000u; break;
    case 11: v = 0xffffffffu; break;
    case 12: v = 0x7fffffffffffffffull; break;
    case 13: v = 0x8000000000000000ull; break;
    case 14: v = UINT64_C(1) << rn(64); break;
    case 15: v = (UINT64_C(1) << rn(64)) - 1; break;
    case 16: { const unsigned w = 1 + rn(64); v = rnd() & (w == 64 ? ~UINT64_C(0) : (UINT64_C(1) << w) - 1); break; } /* random width */
    case 17: v = rn(40); break;
    case 18: v = 0 - (uint64_t)rn(40); break;
    default: v = rnd(); break;
    }
    if (rn(8) == 0) v = ~v;
    return v;
}

static const uint32_t f32_edges[] = {
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x40000000, 0x40400000, 0x3fc00000, 0x3f800001,
    0x7f7fffff, 0xff7fffff, 0x00800000, 0x007fffff, 0x00000001, 0x80000001, 0x807fffff, 0x7f800000, 0xff800000,
    0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001, 0x7fffffff, 0x7fc00001, 0x477fe000, 0x477ff000, 0x477fefff,
    0x33800000, 0x33800001, 0x387fc000, 0x38800000, 0x38000000, 0x4f000000, 0x4effffff, 0xcf000000, 0xcf000001,
    0x4f800000, 0x5f000000, 0x5effffff, 0xdf000000, 0xdf000001, 0x5f800000, 0x4b000000, 0x4b000001, 0x40200000,
    0x40600000, 0xbf000000, 0xbfc00000, 0x3efffffe, 0x3f7fffff, 0x477fe001, 0x47800000, 0x38ffe000, 0x33000000,
    0x00400000, 0x34000000, 0x7e967699, 0x1e3ce508, 0x0000ffff, 0x00010000, 0x7f000000, 0x00ffffff,
};
static const uint64_t f64_edges[] = {
    0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull, 0x3fe0000000000000ull,
    0x4000000000000000ull, 0x4008000000000000ull, 0x3ff8000000000000ull, 0x3ff0000000000001ull, 0x7fefffffffffffffull,
    0xffefffffffffffffull, 0x0010000000000000ull, 0x000fffffffffffffull, 0x0000000000000001ull, 0x8000000000000001ull,
    0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff0000000000001ull,
    0x7fffffffffffffffull, 0x41e0000000000000ull, 0x41dfffffffc00000ull, 0xc1e0000000000000ull, 0xc1e0000000200000ull,
    0x41dfffffffdfffffull, 0x43e0000000000000ull, 0x43dfffffffffffffull, 0xc3e0000000000000ull, 0xc3e0000000000001ull,
    0x4330000000000000ull, 0x4330000000000001ull, 0x4004000000000000ull, 0x400c000000000000ull, 0xbfe0000000000000ull,
    0x3fdfffffffffffffull, 0x3feffffffffffffull, 0x40effc0000000000ull, 0x40effe0000000000ull, 0x3e70000000000000ull,
    0x47efffffe0000000ull, 0x47effffff0000000ull, 0x36a0000000000000ull, 0x3810000000000000ull, 0x7ff00000ffffffffull,
};
#define NELEM(a) (sizeof(a) / sizeof((a)[0]))

static uint32_t f32_any(void) {
    uint32_t v;
    switch (rn(8)) {
    case 0: v = (uint32_t)rnd(); break;
    case 1: { /* a float in a plausible range */ float f = (float)((int32_t)rnd() >> rn(32)) / (float)(1 + rn(1000)); memcpy(&v, &f, 4); break; }
    case 2: { float f = (float)((int)rn(2001) - 1000) / 8.0f; memcpy(&v, &f, 4); break; }
    default: v = f32_edges[rn(NELEM(f32_edges))]; break;
    }
    if (rn(6) == 0) v ^= 0x80000000u;
    return v;
}
static uint64_t f64_any(void) {
    uint64_t v;
    switch (rn(8)) {
    case 0: v = rnd(); break;
    case 1: { double d = (double)((int64_t)rnd() >> rn(64)) / (double)(1 + rn(1000)); memcpy(&v, &d, 8); break; }
    case 2: { double d = (double)((int)rn(2001) - 1000) / 8.0; memcpy(&v, &d, 8); break; }
    default: v = f64_edges[rn(NELEM(f64_edges))]; break;
    }
    if (rn(6) == 0) v ^= 0x8000000000000000ull;
    return v;
}
static uint8_t byte_any(void) {
    static const uint8_t e[] = {0, 1, 0x7f, 0x80, 0xff, 0x81, 0xfe, 2, 0x40, 0xc0};
    return rn(3) ? e[rn(sizeof e)] : (uint8_t)rnd();
}
static uint16_t word_any(void) {
    static const uint16_t e[] = {0, 1, 0x7fff, 0x8000, 0xffff, 0x8001, 0xfffe, 0x00ff, 0xff00, 0x4000, 0xc000, 0x7f80, 0x0100};
    return rn(3) ? e[rn(sizeof e / 2)] : (uint16_t)rnd();
}
static uint32_t dword_any(void) {
    static const uint32_t e[] = {0, 1, 0x7fffffff, 0x80000000u, 0xffffffffu, 0x80000001u, 0xfffffffeu, 0xffff, 0x10000, 0xffff0000u, 0x7fff8000u};
    return rn(3) ? e[rn(sizeof e / 4)] : (uint32_t)rnd();
}

/* Fills 16 bytes with one of several lane styles. */
static VpXmm vec_any(void) {
    VpXmm v;
    switch (rn(7)) {
    case 0: v.u64[0] = rnd(); v.u64[1] = rnd(); break;
    case 1: for (int i = 0; i < 4; ++i) v.u32[i] = f32_any(); break;
    case 2: for (int i = 0; i < 2; ++i) v.u64[i] = f64_any(); break;
    case 3: for (int i = 0; i < 16; ++i) v.u8[i] = byte_any(); break;
    case 4: for (int i = 0; i < 8; ++i) v.u16[i] = word_any(); break;
    case 5: for (int i = 0; i < 4; ++i) v.u32[i] = dword_any(); break;
    default: v.u32[0] = f32_any(); v.u32[1] = f32_any(); v.u64[1] = f64_any(); break; /* scalar-ish */
    }
    return v;
}

/* x87 value (80-bit) from a list of interesting encodings. */
static void f80_any(uint64_t* m, uint16_t* se) {
    const unsigned k = rn(26);
    const uint16_t sgn = rn(2) ? 0x8000 : 0;
    switch (k) {
    case 0: *m = 0; *se = sgn; break;                                    /* zero */
    case 1: *m = 0x8000000000000000ull; *se = sgn | 0x3fff; break;       /* 1.0 */
    case 2: *m = 0x8000000000000000ull; *se = sgn | 0x7fff; break;       /* inf */
    case 3: *m = 0xc000000000000000ull; *se = sgn | 0x7fff; break;       /* qnan (indefinite) */
    case 4: *m = 0xa000000000000000ull; *se = sgn | 0x7fff; break;       /* snan */
    case 5: *m = rnd() >> 1; *se = sgn; break;                           /* denormal */
    case 6: *m = 1; *se = sgn; break;                                    /* smallest denormal */
    case 7: *m = rnd() | 0x8000000000000000ull; *se = sgn | 0x0001; break; /* smallest exponent */
    case 8: *m = rnd() | 0x8000000000000000ull; *se = sgn | 0x7ffe; break; /* largest exponent */
    case 9: *m = 0xffffffffffffffffull; *se = sgn | 0x3fff; break;
    case 10: *m = 0x8000000000000001ull; *se = sgn | 0x3fff; break;
    case 11: *m = 0x8000000000000000ull; *se = sgn | (uint16_t)(0x3fff + rn(80) - 40); break; /* powers of two */
    case 12: *m = ((uint64_t)(rn(1000) + 1) << 53) | 0x8000000000000000ull; *se = sgn | (uint16_t)(0x3fff + rn(70) - 10); break;
    case 13: *m = 0xc000000000000000ull; *se = sgn | 0x3fff; break;     /* 1.5 */
    case 14: *m = 0xa000000000000000ull; *se = sgn | 0x4000; break;     /* 2.5 */
    case 15: *m = 0xb504f333f9de6484ull; *se = sgn | 0x3fff; break;     /* sqrt 2 */
    case 16: *m = 0x8000000000000000ull; *se = sgn | 0x403d; break;     /* 2^62 */
    case 17: *m = 0xffffffffffffffffull; *se = sgn | 0x403d; break;
    case 18: *m = 0x8000000000000000ull; *se = sgn | 0x401e; break;     /* 2^31 */
    case 19: *m = 0xfffffffe00000000ull; *se = sgn | 0x401d; break;
    case 20: *m = rnd() & 0x7fffffffffffffffull; *se = sgn | 0x3fff; break;     /* unnormal (no integer bit) */
    case 21: *m = rnd() | 0x8000000000000000ull; *se = sgn | 0x3fff; break;
    case 22: *m = 0x8000000000000000ull; *se = sgn | (uint16_t)(0x3fff + rn(16) - 8); break;
    case 23: *m = rnd() | 0x8000000000000000ull; *se = sgn | (uint16_t)(0x3fff + rn(130) - 65); break;
    case 24: *m = (rnd() | 0x8000000000000000ull) & ~0x7ffull; *se = sgn | (uint16_t)(0x3fff + rn(70) - 5); break; /* double-exact */
    default: *m = (rnd() | 0x8000000000000000ull) & ~0xffffffffffull; *se = sgn | (uint16_t)(0x3fff + rn(40) - 10); break;
    }
}

#define P_FLOAT 1u   /* vector registers get the float-edge values (else plain random bytes) */
#define P_X87   2u   /* the x87 stack is populated, with random control word */
#define P_DF    4u   /* DF may be set */
#define P_DAZ   8u   /* MXCSR.DAZ / FTZ may be set (the translation does not model them) */
#define P_ALL   7u

static void init_state(VpCpu* c, unsigned profile) {
    memset(c, 0, sizeof *c);
    for (int i = 0; i < 16; ++i) c->r[i] = rn(3) ? edge64() : rnd();
    c->r[VP_RSP] = VP_STACK_BASE + VP_STACK_SIZE - 0x100;
    c->r[VP_RBP] = VP_DATA_BASE + 0x400;
    c->r[VP_RDI] = VP_DATA_BASE;
    c->r[VP_RSI] = VP_DATA_BASE + 0x800;
    c->cf = rn(2); c->pf = rn(2); c->af = rn(2); c->zf = rn(2); c->sf = rn(2); c->of = rn(2);
    c->df = (profile & P_DF) ? (rn(4) == 0) : 0;
    for (int i = 0; i < 16; ++i) {
        c->xmm[i] = (profile & P_FLOAT) ? vec_any() : (VpXmm){{0}};
        c->ymmh[i] = (profile & P_FLOAT) ? vec_any() : (VpXmm){{0}};
        if (!(profile & P_FLOAT)) { c->xmm[i].u64[0] = rnd(); c->xmm[i].u64[1] = rnd(); c->ymmh[i].u64[0] = rnd(); c->ymmh[i].u64[1] = rnd(); }
    }
    /* MXCSR: all exceptions masked; rounding, DAZ and FTZ random (FTZ / DAZ about a third of the time). */
    c->mxcsr = 0x1f80u | (rn(4) << 13);
    if (profile & P_DAZ) c->mxcsr |= (rn(3) == 0 ? 0x40u : 0) | (rn(3) == 0 ? 0x8000u : 0);
    c->fcw = 0x037f;
    c->fsw = 0;
    c->ftop = 0;
    c->ftag = 0;
    if (profile & P_X87) {
        /* every exception masked (bits 0-5; the reserved bits 6-7 read as 1); precision 24 (rare), 53 or 64; random rounding */
        c->fcw = (uint16_t)(0x007fu | ((rn(4) == 1 ? 0u : (rn(2) ? 3u : 2u)) << 8) | (rn(4) << 10));
        c->ftop = (uint8_t)rn(8);
        const unsigned depth = rn(3) ? 2 + rn(7) : rn(9);
        for (unsigned i = 0; i < depth && i < 8; ++i) {
            const int p = (c->ftop + i) & 7; /* ST(i) = R[(ftop + i) & 7] */
            f80_any(&c->st[p].m, &c->st[p].se);
            c->ftag |= (uint8_t)(1u << p);
        }
        for (int p = 0; p < 8; ++p) if (!(c->ftag & (1u << p))) { c->st[p].m = 0; c->st[p].se = 0; }
        c->fsw = (uint16_t)(rn(2) ? 0 : ((rn(2) << 8) | (rn(2) << 9) | (rn(2) << 10) | (rn(2) << 14))); /* C0..C3 */
    }
    uint8_t* d = (uint8_t*)(uintptr_t)VP_DATA_BASE;
    for (unsigned i = 0; i < WINDOW; i += 16) {
        VpXmm v = vec_any();
        if (!(profile & P_FLOAT)) { v.u64[0] = rnd(); v.u64[1] = rnd(); }
        memcpy(d + i, &v, 16);
    }
    memset((void*)(uintptr_t)VP_STACK_BASE, 0, VP_STACK_SIZE);
    uint8_t* s = (uint8_t*)(uintptr_t)(VP_STACK_BASE + VP_STACK_SIZE - 0x100 - STACKLO);
    for (unsigned i = 0; i < STACKWIN; i += 8) { const uint64_t v = edge64(); memcpy(s + i, &v, 8); }
}

/* ---- running --------------------------------------------------------------------------------- */
static sigjmp_buf fault_jmp;
static volatile int in_native;
static const char* cur_text; static uint64_t cur_seed;
static void on_fault(int sig) {
    if (in_native) siglongjmp(fault_jmp, 1);
    fprintf(stderr, "signal %d in the translated run (or the harness): slot \"%s\" seed=%" PRIu64 "\n", sig, cur_text ? cur_text : "?", cur_seed);
    _exit(99);
}

typedef struct Snapshot { uint8_t data[WINDOW]; uint8_t stack[STACKWIN]; } Snapshot;
static void snap(Snapshot* s) {
    memcpy(s->data, (const void*)(uintptr_t)VP_DATA_BASE, WINDOW);
    memcpy(s->stack, (const void*)(uintptr_t)(VP_STACK_BASE + VP_STACK_SIZE - 0x100 - STACKLO), STACKWIN);
    /* the pushed return address differs by design */
    memset(s->stack + STACKLO - 8, 0, 8);
}

/* How a vector register result is compared: 0 bit-exact; 32 / 64: lanes of that width, any NaN equals any NaN
 * (which NaN an operation returns depends on how the host compiler ordered the operands, and on ARM on its
 * own rules); -32: single-precision lanes within 2^-11 of each other (rcpps / rsqrtps: the hardware's
 * table-based estimate is not specified to the bit, and differs between vendors). */
static int nan32(uint32_t v) { return (v & 0x7f800000u) == 0x7f800000u && (v & 0x7fffffu); }
static int nan64(uint64_t v) { return (v & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (v & 0xfffffffffffffull); }
static int vec_equal(const VpXmm* a, const VpXmm* b, int kind) {
    if (!memcmp(a, b, 16)) return 1;
    if (kind == 64) {
        for (int i = 0; i < 2; ++i) if (a->u64[i] != b->u64[i] && !(nan64(a->u64[i]) && nan64(b->u64[i]))) return 0;
        return 1;
    }
    if (kind == 32 || kind == -32) {
        for (int i = 0; i < 4; ++i) {
            if (a->u32[i] == b->u32[i] || (nan32(a->u32[i]) && nan32(b->u32[i]))) continue;
            if (kind == 32) return 0;
            const double x = a->f32[i], y = b->f32[i];
            const double ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
            /* the hardware flushes a tiny result to zero and treats a denormal operand as zero: not modelled */
            if (ax < 1e-30 || ay < 1e-30 || ax > 1e30 || ay > 1e30) continue;
            if (!(x - y <= (x < 0 ? -x : x) * 0.0005 && y - x <= (x < 0 ? -x : x) * 0.0005)) return 0;
        }
        return 1;
    }
    return 0;
}

static const char* field_diff(const VpCpu* t, const VpCpu* n, const Snapshot* ts, const Snapshot* ns, unsigned undef, int kind, int x87, char* extra, size_t xl) {
    extra[0] = 0;
    for (int i = 0; i < 16; ++i) if (t->r[i] != n->r[i]) { snprintf(extra, xl, "r%d native=%016" PRIx64 " translated=%016" PRIx64, i, n->r[i], t->r[i]); return "gpr"; }
    if (!(undef & 0x001) && t->cf != n->cf) { snprintf(extra, xl, "native=%d translated=%d", n->cf, t->cf); return "cf"; }
    if (!(undef & 0x004) && t->pf != n->pf) { snprintf(extra, xl, "native=%d translated=%d", n->pf, t->pf); return "pf"; }
    if (!(undef & 0x010) && t->af != n->af) { snprintf(extra, xl, "native=%d translated=%d", n->af, t->af); return "af"; }
    if (!(undef & 0x040) && t->zf != n->zf) { snprintf(extra, xl, "native=%d translated=%d", n->zf, t->zf); return "zf"; }
    if (!(undef & 0x080) && t->sf != n->sf) { snprintf(extra, xl, "native=%d translated=%d", n->sf, t->sf); return "sf"; }
    if (!(undef & 0x800) && t->of != n->of) { snprintf(extra, xl, "native=%d translated=%d", n->of, t->of); return "of"; }
    if (t->df != n->df) return "df";
    for (int i = 0; i < 16; ++i) {
        if (!vec_equal(&t->xmm[i], &n->xmm[i], kind)) { snprintf(extra, xl, "xmm%d native=%016" PRIx64 "%016" PRIx64 " translated=%016" PRIx64 "%016" PRIx64, i, n->xmm[i].u64[1], n->xmm[i].u64[0], t->xmm[i].u64[1], t->xmm[i].u64[0]); return "xmm"; }
        if (!vec_equal(&t->ymmh[i], &n->ymmh[i], kind)) { snprintf(extra, xl, "ymmh%d native=%016" PRIx64 "%016" PRIx64 " translated=%016" PRIx64 "%016" PRIx64, i, n->ymmh[i].u64[1], n->ymmh[i].u64[0], t->ymmh[i].u64[1], t->ymmh[i].u64[0]); return "ymmh"; }
    }
    if ((t->mxcsr & 0xffc0u) != (n->mxcsr & 0xffc0u)) { snprintf(extra, xl, "native=%08x translated=%08x", n->mxcsr, t->mxcsr); return "mxcsr"; }
    if (x87) {
        if (t->fcw != n->fcw) { snprintf(extra, xl, "native=%04x translated=%04x", n->fcw, t->fcw); return "fcw"; }
        if (t->ftop != n->ftop) { snprintf(extra, xl, "native=%d translated=%d", n->ftop, t->ftop); return "ftop"; }
        if (t->ftag != n->ftag) { snprintf(extra, xl, "native=%02x translated=%02x", n->ftag, t->ftag); return "ftag"; }
        if (t->fsw != n->fsw) { snprintf(extra, xl, "native=%04x translated=%04x", n->fsw, t->fsw); return "fsw"; }
        for (int i = 0; i < 8; ++i) {
            if (!(n->ftag & (1u << i))) continue; /* an empty register's contents are not architectural */
            if (t->st[i].m != n->st[i].m || t->st[i].se != n->st[i].se) { snprintf(extra, xl, "R%d native=%04x:%016" PRIx64 " translated=%04x:%016" PRIx64, i, n->st[i].se, n->st[i].m, t->st[i].se, t->st[i].m); return "st"; }
        }
    }
    if (memcmp(ts->data, ns->data, WINDOW)) {
        unsigned o = 0; while (ts->data[o] == ns->data[o]) ++o;
        snprintf(extra, xl, "data+%#x native=%02x translated=%02x", o, ns->data[o], ts->data[o]); return "memory";
    }
    if (memcmp(ts->stack, ns->stack, STACKWIN)) { snprintf(extra, xl, "stack differs"); return "stack"; }
    return NULL;
}

static void dump_start(const VpCpu* c) {
    for (int i = 0; i < 16; ++i) fprintf(stderr, "    r%-2d=%016" PRIx64, i, c->r[i]), fprintf(stderr, i % 4 == 3 ? "\n" : "");
    fprintf(stderr, "    flags c%d p%d a%d z%d s%d o%d d%d mxcsr=%08x\n", c->cf, c->pf, c->af, c->zf, c->sf, c->of, c->df, c->mxcsr);
    for (int i = 0; i < 4; ++i) fprintf(stderr, "    xmm%d=%016" PRIx64 "%016" PRIx64 "  ymmh%d=%016" PRIx64 "%016" PRIx64 "\n", i, c->xmm[i].u64[1], c->xmm[i].u64[0], i, c->ymmh[i].u64[1], c->ymmh[i].u64[0]);
    fprintf(stderr, "    x87 fcw=%04x fsw=%04x top=%d tag=%02x st:", c->fcw, c->fsw, c->ftop, c->ftag);
    for (int i = 0; i < 8; ++i) fprintf(stderr, " %04x:%016" PRIx64, c->st[i].se, c->st[i].m);
    fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: fuzz_harness CODE.bin SLOTS.txt NSEEDS SEED0 [PROFILE_MASK [FILTER]]\n"); return 2; }
    const unsigned nseeds = (unsigned)strtoul(argv[3], NULL, 0);
    const uint64_t seed0 = strtoull(argv[4], NULL, 0);
    const unsigned profile_mask = argc > 5 ? (unsigned)strtoul(argv[5], NULL, 0) : P_ALL;
    const char* filter = argc > 6 && argv[6][0] ? argv[6] : NULL;
    const int all = getenv("FUZZ_ALL") != NULL; /* report every differing field of every slot, not just the first */
    { VpModule* m = vp_first_module(); if (m && !m->attached) vp_module_set_base(m, m->link_base); }
    map_fixed(VP_CODE_BASE, VP_CODE_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC);
    map_fixed(VP_STACK_BASE, VP_STACK_SIZE, PROT_READ | PROT_WRITE);
    map_fixed(VP_DATA_BASE, VP_DATA_SIZE, PROT_READ | PROT_WRITE);
    FILE* cf = fopen(argv[1], "rb");
    if (!cf) { perror(argv[1]); return 2; }
    fread((void*)(uintptr_t)VP_CODE_BASE, 1, VP_CODE_SIZE, cf);
    fclose(cf);
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = on_fault; sa.sa_flags = SA_NODEFER;
    sigaction(SIGSEGV, &sa, NULL); sigaction(SIGFPE, &sa, NULL); sigaction(SIGILL, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
    FILE* sf = fopen(argv[2], "r");
    if (!sf) { perror(argv[2]); return 2; }
    char line[512];
    unsigned slots = 0, bad = 0, runs = 0, nativefaults = 0, tfaults = 0;
    static Snapshot ts, ns;
    while (fgets(line, sizeof line, sf)) {
        char* end;
        const uint64_t off = strtoull(line, &end, 0);
        const unsigned undef = (unsigned)strtoul(end, &end, 16);
        const int kind = (int)strtol(end, &end, 10);
        while (*end == ' ' || *end == '\t') ++end;
        end[strcspn(end, "\n")] = 0;
        if (filter && !strstr(end, filter)) continue;
        ++slots;
        const uint64_t addr = VP_CODE_BASE + off;
        for (unsigned s = 0; s < nseeds; ++s) {
            unsigned profile = 0;
            rng_state = (seed0 + s) * 0x2545f4914f6cdd1dull + off * 0x9e3779b97f4a7c15ull;
            rnd();
            if (profile_mask & P_FLOAT) profile |= rn(4) ? P_FLOAT : 0;
            if (profile_mask & P_X87) profile |= rn(10) ? P_X87 : 0;
            if (profile_mask & P_DF) profile |= P_DF;
            if (profile_mask & P_DAZ) profile |= P_DAZ;
            cur_text = end; cur_seed = seed0 + s;
            VpCpu start, t, n;
            const uint64_t rng_start = rng_state;
            init_state(&start, profile);
            t = start; n = start;
            t.r[VP_RSP] -= 8;
            vp_st64(t.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
            __builtin_ia32_ldmxcsr(0x1f80); /* the host's own MXCSR: translated code never flushes denormals */
            vp_apply_mxcsr(&t);
            const int fault = vp_run(&t, addr) != 0;
            __builtin_ia32_ldmxcsr(0x1f80);
            if (fault && !tfaults++) fprintf(stderr, "translated fault at slot %#" PRIx64 " (%s): %s\n", off, end, t.fault_what ? t.fault_what : "?");
            if (!fault) snap(&ts);
            /* the same start for the native run (memory is initialised again from the same seed) */
            rng_state = rng_start;
            { VpCpu again; init_state(&again, profile); (void)again; }
            n.r[VP_RSP] -= 8;
            vp_st64(n.r[VP_RSP], vp_native_exit_address());
            in_native = 1;
            int nfault = 0;
            if (sigsetjmp(fault_jmp, 1)) { in_native = 0; ++nativefaults; nfault = 1; }
            else { vp_native_run(&n, addr); in_native = 0; }
            __builtin_ia32_ldmxcsr(0x1f80);
            __asm__ volatile("fninit");
            if (fault || nfault) { /* both must fault (#DE, ...) or neither */
                if (fault != nfault) {
                    ++bad;
                    fprintf(stderr, "MISMATCH slot=%#" PRIx64 " \"%s\" seed=%" PRIu64 " profile=%u field=fault native=%d translated=%d\n", off, end, seed0 + s, profile, nfault, fault);
                    if (!all) break;
                }
                continue;
            }
            snap(&ns);
            ++runs;
            char extra[600];
            const char* d = field_diff(&t, &n, &ts, &ns, undef, kind, 1, extra, sizeof extra);
            if (d) {
                if (all) { /* keep going: one report per slot and differing field */
                    static struct { uint64_t off; char field[8]; } seen[8192];
                    static unsigned nseen;
                    int dup = 0;
                    for (unsigned q = 0; q < nseen; ++q) if (seen[q].off == off && !strncmp(seen[q].field, d, 7)) dup = 1;
                    if (dup) continue;
                    if (nseen < 8192) { seen[nseen].off = off; strncpy(seen[nseen].field, d, 7); seen[nseen].field[7] = 0; ++nseen; }
                }
                ++bad;
                fprintf(stderr, "MISMATCH slot=%#" PRIx64 " \"%s\" seed=%" PRIu64 " profile=%u field=%s %s\n", off, end, seed0 + s, profile, d, extra);
                if (getenv("FUZZ_VERBOSE")) {
                    rng_state = rng_start;
                    VpCpu st; init_state(&st, profile); dump_start(&st);
                }
                if (!all) break;
            }
        }
    }
    fprintf(stderr, "fuzz: %u slots, %u runs, %u mismatching slots, %u native faults, %u translated faults\n", slots, runs, bad, nativefaults, tfaults);
    return bad ? 1 : 0;
}
