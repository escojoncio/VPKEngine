// Differential tests of the soft-float f64 (vp4ir/f64.h) against the host's IEEE double arithmetic, in
// all four rounding modes, with random operands from every class (normal, denormal, huge, tiny, ties)
// and the special values; plus the GCN-specific functions against their documented semantics.
#include <bit>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "vp4ir/f64.h"

namespace f = vp4ir::f64;

namespace {

int failures = 0;
uint64_t checks = 0;

uint64_t bits(double d) { return std::bit_cast<uint64_t>(d); }
double dbl(uint64_t b) { return std::bit_cast<double>(b); }
bool isNan(uint64_t b) { return (b & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; }

uint64_t rngState = 0x9e3779b97f4a7c15ull;
uint64_t rnd() {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return rngState * 0x2545f4914f6cdd1dull;
}

// A random double from a mix of distributions.
uint64_t randomValue() {
    uint64_t r = rnd();
    switch (r % 12) {
        case 0: return rnd();  // any bit pattern
        case 1: return rnd() & 0x800fffffffffffffull;  // denormal / zero
        case 2: return (rnd() & 0x800fffffffffffffull) | (uint64_t(1 + rnd() % 3) << 52);  // tiny normal
        case 3: return (rnd() & 0x800fffffffffffffull) | (uint64_t(0x7fe - rnd() % 3) << 52);  // huge
        case 4: return (rnd() & 0x800fffffffffffffull) | (uint64_t(1023 - 4 + rnd() % 9) << 52);  // near 1
        case 5: return (rnd() & 0x8000000000000000ull) | (uint64_t(1023 - 60 + rnd() % 121) << 52) | (rnd() & 0x7ull);  // few bits
        case 6: return (rnd() & 0x8000000000000000ull) | (uint64_t(1023 - 60 + rnd() % 121) << 52) | (rnd() & 0xfffffffffffffull & ~0x7ffull);
        case 7: return (rnd() & 0x8000000000000000ull) | (uint64_t(1023 - 60 + rnd() % 121) << 52) | (0xfffffffffffffull - (rnd() & 7));  // all ones
        case 8: {
            static const uint64_t special[] = {0, 0x8000000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
                                               0x7ff8000000000000ull, 0xfff8000000000001ull, 0x7ff0000000000001ull,
                                               0x3ff0000000000000ull, 0xbff0000000000000ull, 0x0000000000000001ull,
                                               0x000fffffffffffffull, 0x0010000000000000ull, 0x7fefffffffffffffull,
                                               0x3fe0000000000000ull, 0x4000000000000000ull, 0x4330000000000000ull,
                                               0x41dfffffffc00000ull, 0xc1e0000000000000ull};
            return special[rnd() % (sizeof special / sizeof *special)];
        }
        default: return (rnd() & 0x800fffffffffffffull) | (uint64_t(1023 - 80 + rnd() % 161) << 52);  // moderate
    }
}

const int kRoundModes[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};

uint32_t modeWord(int rm, bool denormIn = true, bool denormOut = true) {
    return uint32_t(rm) << 2 | (denormIn ? 0x40u : 0u) | (denormOut ? 0x80u : 0u);
}

void fail(const char* what, uint64_t a, uint64_t b, uint64_t c, uint64_t got, uint64_t want, uint32_t mode) {
    if (++failures <= 40)
        std::printf("FAIL %s a=%016llx b=%016llx c=%016llx mode=%03x: got %016llx want %016llx\n", what,
                    (unsigned long long)a, (unsigned long long)b, (unsigned long long)c, mode, (unsigned long long)got,
                    (unsigned long long)want);
}

// Compare a result with the host's: NaNs only need to be NaNs (payload rules are tested apart).
void expectSame(const char* what, uint64_t a, uint64_t b, uint64_t c, uint64_t got, uint64_t want, uint32_t mode) {
    ++checks;
    if (isNan(want)) {
        if (!isNan(got) || !(got & 0x0008000000000000ull)) fail(what, a, b, c, got, want, mode);
        return;
    }
    if (got != want) fail(what, a, b, c, got, want, mode);
}

// Host result with the denormal handling of `mode` applied around it (inputs flushed by the caller).
uint64_t flushOut(uint64_t r, uint32_t mode) {
    if ((mode & 0x80u) == 0 && (r & 0x7ff0000000000000ull) == 0) return r & 0x8000000000000000ull;
    return r;
}
uint64_t flushIn(uint64_t a, uint32_t mode) {
    if ((mode & 0x40u) == 0 && (a & 0x7ff0000000000000ull) == 0) return a & 0x8000000000000000ull;
    return a;
}

void arithmetic() {
    for (int rmIdx = 0; rmIdx < 4; ++rmIdx) {
        for (int denorm = 0; denorm < 3; ++denorm) {
            uint32_t mode = modeWord(rmIdx, denorm != 1, denorm == 0);
            if (denorm == 2) mode = modeWord(rmIdx, true, false);
            std::fesetround(kRoundModes[rmIdx]);
            for (int i = 0; i < 60000; ++i) {
                uint64_t a = randomValue(), b = randomValue(), c = randomValue();
                volatile double va = dbl(flushIn(a, mode)), vb = dbl(flushIn(b, mode)), vc = dbl(flushIn(c, mode));
                // Flushing denormal results is a post-step on the host's (already rounded) result: it is
                // exact only when the result is not rounded across the denormal boundary, which the
                // hardware's rounding-before-flush matches for the modes tested here.
                uint64_t want;
                want = bits(va + vb);
                expectSame("add", a, b, 0, f::add(a, b, mode), flushOut(want, mode), mode);
                want = bits(va * vb);
                expectSame("mul", a, b, 0, f::mul(a, b, mode), flushOut(want, mode), mode);
                want = bits(std::fma(va, vb, vc));
                expectSame("fma", a, b, c, f::fma(a, b, c, mode), flushOut(want, mode), mode);
                want = bits(va / vb);
                expectSame("div", a, b, 0, f::div(a, b, mode), flushOut(want, mode), mode);
                want = bits(std::sqrt(va));
                expectSame("sqrt", a, 0, 0, f::sqrt(a, mode), flushOut(want, mode), mode);
                want = bits(1.0 / va);
                expectSame("rcp", a, 0, 0, f::rcp(a, mode), flushOut(want, mode), mode);
            }
            // sub is add with a flipped sign
            for (int i = 0; i < 20000; ++i) {
                uint64_t a = randomValue(), b = randomValue();
                volatile double va = dbl(flushIn(a, mode)), vb = dbl(flushIn(b, mode));
                expectSame("sub", a, b, 0, f::add(a, b ^ 0x8000000000000000ull, mode), flushOut(bits(va - vb), mode), mode);
            }
        }
    }
    std::fesetround(FE_TONEAREST);
}

// Cancellation, tiny addends, large exponent gaps: the corners of add / fma.
void arithmeticCorners() {
    for (int rmIdx = 0; rmIdx < 4; ++rmIdx) {
        uint32_t mode = modeWord(rmIdx);
        std::fesetround(kRoundModes[rmIdx]);
        for (int i = 0; i < 100000; ++i) {
            uint64_t a = randomValue();
            // b = -a * (1 + small), c = tiny relative: near-total cancellation
            uint64_t b = a ^ 0x8000000000000000ull;
            b += (rnd() % 5) - 2;
            uint64_t c = (a & 0x8000000000000000ull) | (((a >> 52) & 0x7ff) - 50 - rnd() % 20) << 52 | (rnd() & 0xfffffffffffffull);
            volatile double va = dbl(a), vb = dbl(b), vc = dbl(c);
            if ((a & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) continue;
            expectSame("add-cancel", a, b, 0, f::add(a, b, mode), bits(va + vb), mode);
            expectSame("fma-cancel", a, 0x3ff0000000000000ull, b, f::fma(a, 0x3ff0000000000000ull, b, mode),
                       bits(std::fma(va, 1.0, vb)), mode);
            expectSame("fma-tiny-c", a, b, c, f::fma(a, b, c, mode), bits(std::fma(va, vb, vc)), mode);
            // product that cancels the addend except for the rounding error of the product
            volatile double p = va * vb;
            volatile double np = -p;
            expectSame("fma-residual", a, b, bits(np), f::fma(a, b, bits(np), mode), bits(std::fma(va, vb, np)), mode);
        }
    }
    std::fesetround(FE_TONEAREST);
}

void ldexpTests() {
    for (int i = 0; i < 100000; ++i) {
        uint64_t a = randomValue();
        int32_t k = int32_t(rnd() % 400) - 200;
        if (i % 7 == 0) k = int32_t(rnd() % 2300) - 1150;
        double want = std::ldexp(dbl(a), k);
        expectSame("ldexp", a, uint64_t(uint32_t(k)), 0, f::ldexp(a, k), bits(want), f::kDefaultMode);
    }
}

void integral() {
    for (int i = 0; i < 200000; ++i) {
        uint64_t a = randomValue();
        double d = dbl(a);
        expectSame("trunc", a, 0, 0, f::trunc(a), bits(std::trunc(d)), f::kDefaultMode);
        expectSame("floor", a, 0, 0, f::floor(a), bits(std::floor(d)), f::kDefaultMode);
        expectSame("ceil", a, 0, 0, f::ceil(a), bits(std::ceil(d)), f::kDefaultMode);
        expectSame("rndne", a, 0, 0, f::rndne(a), bits(std::nearbyint(d)), f::kDefaultMode);
        // fract = x - floor(x) (RNE) clamped just below 1.0; inf gives the default NaN
        double fl = std::floor(d);
        double fr = d - fl;
        if (!std::isnan(fr)) fr = std::min(fr, std::nextafter(1.0, 0.0));
        expectSame("fract", a, 0, 0, f::fract(a), bits(fr), f::kDefaultMode);
        if (std::isfinite(d)) {
            int e;
            double m = std::frexp(d, &e);
            if (d == 0) e = 0;
            expectSame("frexp_mant", a, 0, 0, f::frexpMant(a), bits(m), f::kDefaultMode);
            ++checks;
            if (f::frexpExp(a) != e) fail("frexp_exp", a, 0, 0, uint64_t(uint32_t(f::frexpExp(a))), uint64_t(uint32_t(e)), 0);
        } else {
            ++checks;
            if (f::frexpExp(a) != 0) fail("frexp_exp special", a, 0, 0, 0, 0, 0);
            if (!isNan(a)) expectSame("frexp_mant special", a, 0, 0, f::frexpMant(a), a, f::kDefaultMode);
        }
    }
}

void conversions() {
    for (int rmIdx = 0; rmIdx < 4; ++rmIdx) {
        std::fesetround(kRoundModes[rmIdx]);
        uint32_t mode = modeWord(rmIdx) | rmIdx | 0x20u;  // single destination: same round mode, denormals kept
        for (int i = 0; i < 100000; ++i) {
            uint64_t a = randomValue();
            volatile double va = dbl(a);
            volatile float vf = float(va);
            uint32_t want = std::bit_cast<uint32_t>(vf);
            ++checks;
            uint32_t got = f::toF32(a, mode);
            bool wn = std::isnan(vf);
            if (wn ? !((got & 0x7f800000u) == 0x7f800000u && (got & 0x7fffffu) && (got & 0x400000u)) : got != want)
                fail("to_f32", a, 0, 0, got, want, mode);
        }
    }
    std::fesetround(FE_TONEAREST);
    for (int i = 0; i < 100000; ++i) {
        uint64_t a = randomValue();
        double d = dbl(a);
        int32_t wi = std::isnan(d) ? 0 : d >= 2147483648.0 ? INT32_MAX : d <= -2147483648.0 ? INT32_MIN : int32_t(d);
        uint32_t wu = std::isnan(d) ? 0 : d >= 4294967296.0 ? UINT32_MAX : d <= 0 ? 0 : uint32_t(d);
        ++checks;
        if (f::toI32(a) != wi) fail("to_i32", a, 0, 0, uint32_t(f::toI32(a)), uint32_t(wi), 0);
        if (f::toU32(a) != wu) fail("to_u32", a, 0, 0, f::toU32(a), wu, 0);
        uint32_t v = uint32_t(rnd());
        if (i % 3 == 0) v >>= rnd() % 32;
        expectSame("from_i32", v, 0, 0, f::fromI32(int32_t(v)), bits(double(int32_t(v))), 0);
        expectSame("from_u32", v, 0, 0, f::fromU32(v), bits(double(v)), 0);
        uint32_t fb = uint32_t(rnd());
        if (i % 4 == 0) fb &= 0x807fffffu;  // denormal f32
        float ff = std::bit_cast<float>(fb);
        expectSame("from_f32", fb, 0, 0, f::fromF32(fb), bits(double(ff)), 0);
    }
}

bool hostCmp(uint32_t cond, double a, double b) {
    bool nan = std::isnan(a) || std::isnan(b);
    switch (cond) {
        case 0: return false;
        case 1: return a < b;
        case 2: return a == b;
        case 3: return a <= b;
        case 4: return a > b;
        case 5: return a < b || a > b;
        case 6: return a >= b;
        case 7: return !nan;
        case 8: return nan;
        case 9: return !(a >= b);
        case 10: return !(a < b || a > b);
        case 11: return !(a > b);
        case 12: return !(a <= b);
        case 13: return !(a == b);
        case 14: return !(a < b);
        default: return true;
    }
}

void compares() {
    for (int i = 0; i < 200000; ++i) {
        uint64_t a = randomValue(), b = randomValue();
        if (i % 5 == 0) b = a;
        if (i % 11 == 0) b = a ^ 0x8000000000000000ull;
        for (uint32_t cond = 0; cond < 16; ++cond) {
            ++checks;
            bool got = f::cmp(a, b, cond);
            bool want = hostCmp(cond, dbl(a), dbl(b));
            if (got != want) fail("cmp", a, b, cond, got, want, 0);
        }
        // class
        uint32_t mask = uint32_t(rnd()) & 0x3ff;
        uint32_t cls;
        double d = dbl(a);
        bool neg = std::signbit(d);
        if (std::isnan(d)) cls = (a & 0x0008000000000000ull) ? 1 : 0;
        else if (std::isinf(d)) cls = neg ? 2 : 9;
        else if (d == 0) cls = neg ? 5 : 6;
        else if (std::fpclassify(d) == FP_SUBNORMAL) cls = neg ? 4 : 7;
        else cls = neg ? 3 : 8;
        ++checks;
        if (f::classify(a, mask) != bool((mask >> cls) & 1)) fail("class", a, mask, 0, 0, 0, 0);
    }
}

void minMax() {
    for (int i = 0; i < 200000; ++i) {
        uint64_t a = randomValue(), b = randomValue();
        for (uint32_t mode : {f::kDefaultMode, f::kDefaultMode | 0x200u}) {
            bool ieee = mode & 0x200u;
            auto snan = [](uint64_t x) { return isNan(x) && !(x & 0x0008000000000000ull); };
            auto ref = [&](bool isMax) -> uint64_t {
                if (ieee && snan(a)) return a | 0x0008000000000000ull;
                if (ieee && snan(b)) return b | 0x0008000000000000ull;
                if (isNan(a)) return b;
                if (isNan(b)) return a;
                bool lt = dbl(a) < dbl(b);
                return isMax ? (!lt ? a : b) : (lt ? a : b);
            };
            ++checks;
            if (f::min(a, b, mode) != ref(false)) fail("min", a, b, 0, f::min(a, b, mode), ref(false), mode);
            if (f::max(a, b, mode) != ref(true)) fail("max", a, b, 0, f::max(a, b, mode), ref(true), mode);
        }
    }
}

// Directed NaN / invalid rules.
void nanRules() {
    const uint64_t qn = 0x7ff8000000000123ull, sn = 0x7ff0000000000045ull, inf = 0x7ff0000000000000ull;
    const uint64_t one = 0x3ff0000000000000ull;
    auto check = [&](const char* w, uint64_t got, uint64_t want) {
        ++checks;
        if (got != want) fail(w, 0, 0, 0, got, want, 0);
    };
    check("add qnan first", f::add(qn, one), qn);
    check("add snan quieted", f::add(sn, one), sn | 0x0008000000000000ull);
    check("add second nan", f::add(one, qn), qn);
    check("add inf-inf", f::add(inf, inf ^ 0x8000000000000000ull), f::kDefaultNan);
    check("mul inf*0", f::mul(inf, 0), f::kDefaultNan);
    check("fma inf*0+1", f::fma(inf, 0, one), f::kDefaultNan);
    check("div 0/0", f::div(0, 0), f::kDefaultNan);
    check("div inf/inf", f::div(inf, inf), f::kDefaultNan);
    check("div 1/0", f::div(one, 0), inf);
    check("div -1/0", f::div(one | 0x8000000000000000ull, 0), inf | 0x8000000000000000ull);
    check("sqrt -1", f::sqrt(one | 0x8000000000000000ull), f::kDefaultNan);
    check("sqrt -0", f::sqrt(0x8000000000000000ull), 0x8000000000000000ull);
    check("rsq 0", f::rsq(0), inf);
    check("rsq -0", f::rsq(0x8000000000000000ull), inf | 0x8000000000000000ull);
    check("rsq inf", f::rsq(inf), 0);
    check("rcp inf", f::rcp(inf), 0);
    check("rcp -0", f::rcp(0x8000000000000000ull), inf | 0x8000000000000000ull);
    check("x - x = +0", f::add(one, one | 0x8000000000000000ull), 0);
    check("x - x = -0 toward -inf", f::add(one, one | 0x8000000000000000ull, 0x8u | 0x1c0u), 0x8000000000000000ull);
    check("-0 + -0", f::add(0x8000000000000000ull, 0x8000000000000000ull), 0x8000000000000000ull);
    check("overflow rtz", f::mul(0x7fe0000000000000ull, 0x4010000000000000ull, 0xcu | 0x1c0u), 0x7fefffffffffffffull);
    check("overflow rne", f::mul(0x7fe0000000000000ull, 0x4010000000000000ull), inf);
    check("sat nan dx10", f::saturate(qn, 0x1c0u), 0);
    check("sat nan", f::saturate(qn, 0xc0u), qn);
    check("sat 2", f::saturate(0x4000000000000000ull), one);
    check("sat -1", f::saturate(one | 0x8000000000000000ull), 0);
    // denormal flush
    check("flush in", f::add(1, 0, 0x80u), 0);
    check("flush out", f::mul(0x0010000000000000ull, 0x3fe0000000000000ull, 0x40u), 0);
    check("denormal kept", f::mul(0x0010000000000000ull, 0x3fe0000000000000ull), 0x0008000000000000ull);
}

// The division sequence compilers emit for a / b: div_scale x2, rcp, Newton fma steps, div_fmas, div_fixup.
uint64_t gcnDivide(uint64_t n, uint64_t d) {
    const uint64_t one = 0x3ff0000000000000ull, neg = 0x8000000000000000ull;
    bool v0, v1;
    uint64_t s0 = f::divScale(d, d, n, v0);
    uint64_t s1 = f::divScale(n, d, n, v1);
    uint64_t r = f::rcp(s0);
    uint64_t f0 = f::fma(s0 ^ neg, r, one);
    uint64_t f1 = f::fma(r, f0, r);
    uint64_t f2 = f::fma(s0 ^ neg, f1, one);
    uint64_t f3 = f::fma(f1, f2, f1);
    uint64_t m = f::mul(s1, f3);
    uint64_t f4 = f::fma(s0 ^ neg, m, s1);
    uint64_t f5 = f::divFmas(f4, f3, m, v1);
    return f::divFixup(f5, d, n);
}

void divisionSequence() {
    for (int i = 0; i < 300000; ++i) {
        uint64_t n = randomValue(), d = randomValue();
        uint64_t want = bits(dbl(n) / dbl(d));
        uint64_t got = gcnDivide(n, d);
        ++checks;
        if (isNan(want) ? !isNan(got) : got != want) {
            // Results the algorithm can only approximate at the extremes of the range are reported
            // but do not stop the run: count them separately.
            fail("div sequence", n, d, 0, got, want, 0);
        }
    }
}

void rsqAccuracy() {
    for (int i = 0; i < 200000; ++i) {
        uint64_t a = randomValue() & 0x7fffffffffffffffull;
        if (isNan(a) || (a & 0x7ff0000000000000ull) == 0x7ff0000000000000ull || a == 0) continue;
        long double ref = 1.0L / std::sqrt((long double)dbl(a));
        double got = dbl(f::rsq(a));
        ++checks;
        if (std::isinf(ref) || ref > 1e308L || ref < 1e-307L) continue;
        long double ulp = std::ldexp(1.0L, std::ilogb((double)ref) - 52);
        // Where long double is only a double (Apple arm64) the reference itself carries up to ~1 ulp of error
        // (division and square root each round once), so the bound is widened to the reference's own accuracy.
        const long double bound = LDBL_MANT_DIG >= 64 ? 0.502L : 1.52L;
        if (std::fabs((long double)got - ref) > bound * ulp) fail("rsq accuracy", a, 0, 0, bits(got), bits((double)ref), 0);
    }
}

void trigPreop() {
    // The segments of 2/pi sum to 2/pi (to the long double's precision) and decrease in magnitude.
    long double sum = 0;
    for (uint32_t i = 0; i < 3; ++i) sum += dbl(f::trigPreop(0x3ff0000000000000ull, i));
    ++checks;
    if (std::fabs(sum - 0.636619772367581343075535053490057448L) > 1e-18L) fail("trig_preop sum", 0, 0, 0, bits((double)sum), 0, 0);
    ++checks;
    if (dbl(f::trigPreop(0x3ff0000000000000ull, 0)) != 0.6366197723675813) fail("trig_preop 0", 0, 0, 0, f::trigPreop(0x3ff0000000000000ull, 0), bits(0.6366197723675813), 0);
}

}  // namespace

int main() {
    arithmetic();
    arithmeticCorners();
    ldexpTests();
    integral();
    conversions();
    compares();
    minMax();
    nanRules();
    divisionSequence();
    rsqAccuracy();
    trigPreop();
    std::printf("f64: %llu checks, %d failures\n", (unsigned long long)checks, failures);
    return failures ? 1 : 0;
}
