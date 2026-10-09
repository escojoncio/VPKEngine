// The guest module of the engine test: calls HLE functions (through veneers), is called back
// from HLE (CallGuest), starts guest threads through HLE (CreateThread + Run), reads its TLS
// through fs (as PS4 code does), and uses SSE. Built natively too (-DNATIVE) for the expected
// result: then the HLE functions are plain C and TLS is a thread_local.
typedef unsigned long u64;
extern u64 hle_mix(u64 a, u64 b);
extern u64 hle_callback(u64 (*fn)(u64), u64 x);
extern u64 hle_thread(u64 (*fn)(u64), u64 x);
extern u64 hle_lib(u64 x);                         // calls into the second module
extern double hle_fmul(double a, double b);        // float arguments and result; a signal lands in it
extern u64 hle_fail(u64 (*fn)(u64), u64 x);        // calls back, then fails with EIO
extern void hle_fiber_switch(u64* from, u64* to);  // sceFiberSwitch-like: swaps rbx rsp rbp r12-r15
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
// The Orbis signal handler: float work and flag changes that must not leak into the interrupted code.
volatile u64 signals_seen;
volatile double signal_sink;
void on_signal(int sig, void* ctx) {
    (void)ctx;
    double d = 3.25;
    for (int i = 0; i < 8; ++i) d = d * 1.5 - (double)sig;
    signal_sink = d;
    signals_seen += (u64)sig;
}
volatile u64 fail_out;
void fail_test(u64 x) { fail_out = hle_fail(work, x) + 1000; }
// Fibers switched by the host as AstroVisionPro's fiber_fex.cpp does: the HLE call swaps the
// callee-saved registers and the stack pointer in the frame, and the veneer's `ret` continues in
// the other context (in the translation: a resume point with no host frames).
#ifndef NATIVE
static u64 fib_stack[4096] __attribute__((aligned(16)));
static u64 main_ctx[7], fib_ctx[7];
static volatile u64 fib_acc;
static void fiber_body(void) {
    for (u64 k = 1;; ++k) { fib_acc = fib_acc * 5 + k; hle_fiber_switch(fib_ctx, main_ctx); }
}
u64 fiber_test(u64 rounds) {
    fib_ctx[1] = (u64)&fib_stack[4090]; // rsp: the veneer's ret pops fiber_body
    fib_stack[4090] = (u64)fiber_body;
    u64 acc = 0;
    for (u64 i = 0; i < rounds; ++i) { hle_fiber_switch(main_ctx, fib_ctx); acc = acc * 31 + fib_acc + i; }
    return acc;
}
#endif
// HLE call cost: n calls through a veneer (timed by the host when VP_HLE_BENCH is set).
u64 hle_bench(u64 n) { u64 acc = 1; for (u64 i = 0; i < n; ++i) acc = hle_mix(acc, i); return acc; }
u64 guest_main(u64 seed) {
    u64 acc = hle_mix(seed, 7);
    acc += hle_callback(work, acc);          // HLE calls back into the guest
    acc ^= hle_callback(nested, acc);        // ... which calls HLE, which calls back again
    for (int i = 0; i < 4; ++i) acc += hle_thread(work, acc + (u64)i); // guest threads
    acc += hle_lib(acc);                    // into another translated module
    double f = (double)(acc & 0xfff) + 0.25;
    f = hle_fmul(f, 1.5) + hle_fmul(f, -0.125);
    acc += (u64)(long)(f * 16.0) + signals_seen;
    return acc ^ tls_read();
}
