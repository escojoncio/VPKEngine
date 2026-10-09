// A second translated module (a "prx"): reached only through HLE, loaded at its own address and
// linked into the host ahead of the game's translation, so that attaching must search every module.
typedef unsigned long u64;
u64 lib_fn(u64 x) {
    double d = (double)(x & 0xffff) + 0.5;
    for (int i = 0; i < 20; ++i) { x = (x << 7 | x >> 57) ^ (x * 0x9e3779b97f4a7c15ul); d = d * 0.75 + (double)(x & 15); }
    return x + (u64)(long)(d * 64.0);
}
