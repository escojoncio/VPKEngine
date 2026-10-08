// A program with imports: `vp_ext_double` and `vp_ext_counter` come from the host.
typedef unsigned long u64;
extern u64 vp_ext_double(u64 x);
extern u64 vp_ext_counter(void);
static u64 (*const indirect)(u64) = vp_ext_double; // an import through a data pointer too
u64 vp_main(u64* mem) {
    u64 acc = 7;
    for (int i = 0; i < 10; ++i) { acc = vp_ext_double(acc + (u64)i) ^ vp_ext_counter(); mem[i] = acc; }
    return indirect(acc) + mem[3];
}
