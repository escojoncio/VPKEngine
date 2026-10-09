/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The part of <math.h> the translated C uses, for building a game pack without any platform SDK
 * (see string.h here). Every function maps to the compiler builtin, which is a single arm64
 * instruction with -fno-math-errno (fsqrt, frintx, frinti, frintm, frintp, frintz, frintn): the
 * pack needs no libm.
 */
#ifndef VP_FREESTANDING_MATH_H
#define VP_FREESTANDING_MATH_H

#define INFINITY __builtin_inff()
#define NAN __builtin_nanf("")
#define HUGE_VAL __builtin_huge_val()
#define HUGE_VALF __builtin_huge_valf()

#define isnan(x) __builtin_isnan(x)
#define isinf(x) __builtin_isinf(x)
#define isfinite(x) __builtin_isfinite(x)
#define signbit(x) __builtin_signbit(x)
#define isunordered(a, b) __builtin_isunordered((a), (b))
#define isgreater(a, b) __builtin_isgreater((a), (b))
#define isgreaterequal(a, b) __builtin_isgreaterequal((a), (b))
#define isless(a, b) __builtin_isless((a), (b))
#define islessequal(a, b) __builtin_islessequal((a), (b))
#define FP_NAN 0
#define FP_INFINITE 1
#define FP_NORMAL 4
#define FP_SUBNORMAL 3
#define FP_ZERO 2
#define fpclassify(x) __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, (x))

#define sqrt(x) __builtin_sqrt(x)
#define sqrtf(x) __builtin_sqrtf(x)
#define rint(x) __builtin_rint(x)
#define rintf(x) __builtin_rintf(x)
#define nearbyint(x) __builtin_nearbyint(x)
#define nearbyintf(x) __builtin_nearbyintf(x)
#define floor(x) __builtin_floor(x)
#define floorf(x) __builtin_floorf(x)
#define ceil(x) __builtin_ceil(x)
#define ceilf(x) __builtin_ceilf(x)
#define trunc(x) __builtin_trunc(x)
#define truncf(x) __builtin_truncf(x)
#define fabs(x) __builtin_fabs(x)
#define fabsf(x) __builtin_fabsf(x)
#define copysign(a, b) __builtin_copysign((a), (b))
#define copysignf(a, b) __builtin_copysignf((a), (b))
#endif
