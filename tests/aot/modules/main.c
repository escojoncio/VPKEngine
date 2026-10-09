// The main module: imports lib_mix from the guest library and passes it its own function.
typedef unsigned long u64;
extern u64 lib_mix(u64 x, u64 (*cb)(u64));
static u64 square_plus(u64 v) { return v * v + 3; }
u64 vp_main(u64* mem) {
    u64 acc = lib_mix(5, square_plus);
    mem[0] = acc;
    return lib_mix(acc, square_plus) ^ mem[0];
}
