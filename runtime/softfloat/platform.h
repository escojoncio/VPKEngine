/* SoftFloat's platform header for VPEngine (little-endian, clang/gcc). */
#ifndef VP_SOFTFLOAT_PLATFORM_H
#define VP_SOFTFLOAT_PLATFORM_H
#define LITTLEENDIAN 1
#define INLINE static inline
/* no SOFTFLOAT_BUILTIN_CLZ / SOFTFLOAT_INTRINSIC_INT128: their header inlines would bypass the renaming */
#ifndef THREAD_LOCAL
#define THREAD_LOCAL _Thread_local
#endif
#include "opts-GCC.h"
#endif
