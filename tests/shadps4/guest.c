// The guest module of the engine test: calls HLE functions (through veneers), is called back
// from HLE (CallGuest), starts guest threads through HLE (CreateThread + Run), reads its TLS
// through fs (as PS4 code does), and uses SSE. Built natively too (-DNATIVE) for the expected
// result: then the HLE functions are plain C and TLS is a thread_local.
typedef unsigned long u64;
extern u64 hle_mix(u64 a, u64 b);
extern u64 hle_callback(u64 (*fn)(u64), u64 x);
extern u64 hle_thread(u64 (*fn)(u64), u64 x);
#ifdef NATIVE
extern __thread u64 native_tls;
static u64 tls_read(void) { return native_tls; }
#else
static u64 tls_read(void) { u64 v; __asm__ volatile("mov %%fs:0x10, %0" : "=r"(v)); return v; }
#endif
static u64 work(u64 x) {
    u64 acc = x ^ tls_read();
    double d = 1.0;
    for (int i = 0; i < 50; ++i) { acc = acc * 6364136223846793005ul + 1442695040888963407ul; d = d * 1.0001 + (double)(acc >> 60); }
    return acc ^ (u64)(long)(d * 1000.0);
}
static u64 nested(u64 x) { return hle_mix(work(x), 3) + hle_callback(work, x + 1); }
u64 guest_main(u64 seed) {
    u64 acc = hle_mix(seed, 7);
    acc += hle_callback(work, acc);          // HLE calls back into the guest
    acc ^= hle_callback(nested, acc);        // ... which calls HLE, which calls back again
    for (int i = 0; i < 4; ++i) acc += hle_thread(work, acc + (u64)i); // guest threads
    return acc ^ tls_read();
}
