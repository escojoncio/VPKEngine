// A guest library: exported function that calls back into the main module through a pointer.
typedef unsigned long u64;
extern u64 vp_ext_counter(void);
u64 lib_mix(u64 x, u64 (*cb)(u64)) {
    u64 acc = x;
    for (int i = 0; i < 6; ++i) acc = cb(acc ^ (u64)i) * 31 + vp_ext_counter();
    return acc;
}
