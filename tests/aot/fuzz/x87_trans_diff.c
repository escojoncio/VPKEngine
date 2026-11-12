/* x86 only: compares the x87 transcendental instructions of the runtime (vp_x87_transcendental) with the
 * host's own, argument regime by argument regime: how many results are bit-exact, the largest
 * difference in units of the last place, and status word differences.
 *   cc -O2 -std=gnu11 -frounding-math -I../../../runtime x87_trans_diff.c ../../../runtime/vp_host.c -o diff -lm -latomic -lpthread
 *   [X87_DUMP=N] ./diff [count] [seed]   (X87_DUMP: see x87_trans_ref.py, which says which of the two is rounded correctly)
 */
#include "vp_emit.h"
#include <stdio.h>
#include <stdlib.h>

enum { SIN, COS, SINCOS, PTAN, PATAN, F2XM1, YL2X, YL2XP1 };
extern void vp_x87_transcendental(VpCpu* c, int op);

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double ur(void) { return (double)(rnd() >> 11) / 9007199254740992.0; }

typedef struct { uint64_t m; uint16_t se; } F80;
static F80 make(double v, int ex) { /* v in [1,2) times 2^ex, random 64-bit significand */
    (void)v;
    F80 r;
    r.m = rnd() | (UINT64_C(1) << 63);
    r.se = (uint16_t)(0x3FFF + ex) | (rnd() & 1 ? 0x8000 : 0);
    return r;
}

/* host results: st0, st1 (after the instruction), fsw */
static void host(int op, F80 a, F80 b, F80* o0, F80* o1, uint16_t* fsw) {
    uint8_t buf[2][16] __attribute__((aligned(16)));
    memset(buf, 0, sizeof buf);
    uint16_t sw;
    if (op == PATAN || op == YL2X || op == YL2XP1) __asm__ volatile("fninit; fldt %[b]; fldt %[a]" : : [a] "m"(a), [b] "m"(b));
    else __asm__ volatile("fninit; fldt %[a]" : : [a] "m"(a));
    switch (op) {
    case SIN: __asm__ volatile("fsin"); break;
    case COS: __asm__ volatile("fcos"); break;
    case SINCOS: __asm__ volatile("fsincos"); break;
    case PTAN: __asm__ volatile("fptan"); break;
    case PATAN: __asm__ volatile("fpatan"); break;
    case F2XM1: __asm__ volatile("f2xm1"); break;
    case YL2X: __asm__ volatile("fyl2x"); break;
    default: __asm__ volatile("fyl2xp1"); break;
    }
    __asm__ volatile("fnstsw %[sw]; fstpt %[o0]" : [sw] "=m"(sw), [o0] "=m"(buf[0]));
    if (op == SINCOS || op == PTAN) __asm__ volatile("fstpt %[o1]" : [o1] "=m"(buf[1]));
    __asm__ volatile("fninit");
    memcpy(&o0->m, buf[0], 8); memcpy(&o0->se, buf[0] + 8, 2);
    memcpy(&o1->m, buf[1], 8); memcpy(&o1->se, buf[1] + 8, 2);
    *fsw = sw;
}

static void mine(int op, F80 a, F80 b, F80* o0, F80* o1, uint16_t* fsw) {
    VpCpu c;
    memset(&c, 0, sizeof c);
    c.fcw = 0x037F;
    int n = op == PATAN || op == YL2X || op == YL2XP1 ? 2 : 1;
    /* push b then a */
    c.ftop = 0;
    c.ftag = 0;
    if (n == 2) { c.ftop = 6; c.st[7].m = b.m; c.st[7].se = b.se; c.st[6].m = a.m; c.st[6].se = a.se; c.ftag = 0xC0; }
    else { c.ftop = 7; c.st[7].m = a.m; c.st[7].se = a.se; c.ftag = 0x80; }
    vp_x87_transcendental(&c, op);
    const int t = c.ftop;
    o0->m = c.st[t & 7].m; o0->se = c.st[t & 7].se;
    o1->m = c.st[(t + 1) & 7].m; o1->se = c.st[(t + 1) & 7].se;
    *fsw = (uint16_t)((c.fsw & ~0x3800) | ((c.ftop & 7) << 11));
}

static long ulps(F80 x, F80 y) {
    if (x.se != y.se) return 1000000;
    return (long)((int64_t)(x.m - y.m));
}

int main(int argc, char** argv) {
    const int count = argc > 1 ? atoi(argv[1]) : 20000;
    if (argc > 2) rs = strtoull(argv[2], 0, 0) | 1;
    const char* dm = getenv("X87_DUMP"); /* up to N differing results per instruction go to stderr */
    const int dump_max = dm ? atoi(dm) : 0;
    int dumped[8] = {0};
    static const char* names[] = {"fsin", "fcos", "fsincos", "fptan", "fpatan", "f2xm1", "fyl2x", "fyl2xp1"};
    struct { const char* name; int lo, hi; } reg[] = {{"|x| in 2^-30..2^-1", -30, -2}, {"|x| in 0.5..4", -1, 1}, {"|x| in 4..2^20", 2, 20}, {"|x| in 2^20..2^62", 20, 62}};
    for (int op = 0; op < 8; ++op) {
        for (unsigned g = 0; g < sizeof reg / sizeof *reg; ++g) {
            if ((op == F2XM1 && g > 1) || (op == YL2XP1 && g > 1)) continue;
            long exact = 0, bad = 0, maxu = 0, swd = 0, n = 0;
            uint16_t firstsw = 0, firstsw_h = 0;
            for (int i = 0; i < count; ++i) {
                F80 a = make(0, reg[g].lo + (int)(rnd() % (unsigned)(reg[g].hi - reg[g].lo + 1))), b = make(0, (int)(rnd() % 4) - 1);
                if (op == YL2X) a.se &= 0x7FFF;
                if (op == YL2XP1) { a.se = (uint16_t)((a.se & 0x8000) | (0x3FFF + (int)(rnd() % 3) - 3)); }
                if (op == F2XM1) { a.se = (uint16_t)((a.se & 0x8000) | (0x3FFF - 1 - (int)(rnd() % 6))); }
                F80 h0, h1, m0, m1; uint16_t hs, ms;
                host(op, a, b, &h0, &h1, &hs);
                mine(op, a, b, &m0, &m1, &ms);
                ++n;
                const int two = op == SINCOS || op == PTAN;
                const long u0 = ulps(h0, m0), u1 = two ? ulps(h1, m1) : 0;
                if (dump_max && !two && (u0 != 0) && dumped[op] < dump_max) { /* for x87_trans_ref.py */
                    ++dumped[op];
                    fprintf(stderr, "DUMP %s %016llx %04x %016llx %04x %016llx %04x %016llx %04x\n", names[op], (unsigned long long)a.m, a.se, (unsigned long long)b.m, b.se,
                            (unsigned long long)h0.m, h0.se, (unsigned long long)m0.m, m0.se);
                }
                if (u0 == 0 && u1 == 0) ++exact; else { ++bad; long u = labs(u0) > labs(u1) ? labs(u0) : labs(u1); if (u > maxu) maxu = u; }
                if ((hs & 0x7F27) != (ms & 0x7F27)) { if (!swd) { firstsw = ms; firstsw_h = hs; } ++swd; }
            }
            printf("%-8s %-20s n=%ld exact=%.2f%% maxulp=%ld fsw-differences=%ld", names[op], reg[g].name, n, 100.0 * exact / n, maxu, swd);
            if (swd) printf(" (e.g. host %04x ours %04x)", firstsw_h, firstsw);
            printf("\n");
        }
    }
    return 0;
}
