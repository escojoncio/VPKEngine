// Soft-float f64 library (vp4ir f64 op). Generated into every shader that uses it.
typedef uint u32;
typedef ulong u64;
typedef int i32;
#define VP_CLZ64(x) clz(u64(x))
#define VP_TABLE constant
#define VP_NOUNROLL _Pragma("clang loop unroll(disable)")
