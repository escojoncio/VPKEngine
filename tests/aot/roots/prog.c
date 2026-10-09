// A function reached only through a pointer built at run time from a distance the assembler
// stores as a plain number: static analysis cannot find it, so the first translation lacks it.
// The runtime logs it as a missing entry and `vpaot --roots` takes the log back.
typedef unsigned long u64;
typedef u64 (*fn_t)(u64);
__attribute__((noinline)) u64 anchor(u64 x) { return x * 3 + 1; }
__attribute__((noinline, used)) u64 hidden(u64 x) { u64 a = x; for (int i = 0; i < 10; ++i) a = a * 2654435761u + (a >> 7); return a; }
#ifndef vp_main
__asm__(".data\n.balign 8\nhidden_distance: .quad hidden - anchor\n.text\n");
extern const long hidden_distance;
#define DISTANCE hidden_distance
#else
#define DISTANCE ((char*)hidden - (char*)anchor)
#endif
u64 vp_main(u64* mem) {
    fn_t f = (fn_t)((char*)anchor + DISTANCE);
    u64 acc = anchor(5);
    for (int i = 0; i < 4; ++i) { acc += f(acc + i); mem[i] = acc; }
    return acc;
}
