// C++ exceptions through the game's own runtime (libsupc++ + libgcc_eh, statically linked, as a PS4
// title ships libc++/libunwind): throw across frames, catch by type, rethrow, destructors run
// during unwinding, nested try. vp_main(mem) returns a value built from what happened.
typedef unsigned long u64;
extern "C" u64 vp_main(u64* mem);
static u64* g_mem;
struct Guard { int id; ~Guard() { g_mem[0] = g_mem[0] * 31 + id; } };
struct Err { int code; };
struct Other { u64 v; };
__attribute__((noinline)) static int deep(int n) {
    Guard g{n};
    if (n == 0) throw Err{42};
    return deep(n - 1) + 1;
}
__attribute__((noinline)) static u64 rethrower(int k) {
    try { deep(k); } catch (Err& e) { e.code += k; throw; }
    return 0;
}
__attribute__((noinline)) static u64 other(u64 x) { if (x & 1) throw Other{x * 3}; return x; }
extern "C" u64 vp_main(u64* mem) {
    g_mem = mem;
    mem[0] = 1;
    u64 acc = 0;
    for (int i = 0; i < 5; ++i) {
        try { acc += rethrower(i); }
        catch (const Err& e) { acc = acc * 7 + (u64)e.code; }
    }
    for (u64 x = 0; x < 6; ++x) {
        try { acc += other(x); }
        catch (Other& o) { acc ^= o.v << x; }
        catch (...) { acc += 1000; }
    }
    try { try { throw 5; } catch (int v) { acc += (u64)v; throw Err{v}; } }
    catch (Err& e) { acc += (u64)e.code * 100; }
    return acc ^ mem[0];
}
// The default terminate handler (it would pull in the demangler and stdio): never reached here.
namespace __gnu_cxx { void __verbose_terminate_handler() { __builtin_trap(); } }
