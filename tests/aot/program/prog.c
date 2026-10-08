// A whole freestanding program: globals, a function-pointer table in .data, switch, recursion,
// float math, string copies. Built as a static x86-64 ELF for the translator and, with the same
// source, as a host function for the expected result.  vp_main(mem) returns a checksum.
typedef unsigned long u64;
static u64 counter;
static const char text[] = "vpengine whole program test";
static u64 op_add(u64 a, u64 b) { return a + b; }
static u64 op_mul(u64 a, u64 b) { return a * b | 1; }
static u64 op_rot(u64 a, u64 b) { return (a << (b & 63)) | (a >> ((64 - b) & 63)); }
static u64 (*const table[3])(u64, u64) = { op_add, op_mul, op_rot };
static u64 fib(u64 n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
static void copy(char* d, const char* s, u64 n) { for (u64 i = 0; i < n; ++i) d[i] = s[i]; }
static u64 strlen_(const char* s) { u64 n = 0; while (s[n]) ++n; return n; }
u64 vp_main(u64* mem) {
    u64 acc = 0x9e3779b97f4a7c15ul; float f = 1.0f; double d = 0.5;
    char* buf = (char*)(mem + 512);
    copy(buf, text, strlen_(text) + 1);
    for (int i = 0; i < 64; ++i) {
        acc = table[acc % 3](acc, (u64)i + 3);
        switch (i & 7) {
        case 0: acc ^= fib((acc & 7) + 6); break;
        case 1: f = f * 1.5f - (float)(acc & 0xff); break;
        case 2: d += (double)(long)(acc >> 40) / 7.0; break;
        case 3: counter += acc & 0xff; break;
        case 4: mem[i] = acc; break;
        case 5: acc += (u64)buf[i % 27]; break;
        case 6: acc = (u64)(-(long)acc) / 3; break;
        default: acc -= mem[(i * 5) & 63]; break;
        }
        acc = (acc << 7) | (acc >> 57);
    }
    return acc ^ counter ^ (u64)(long)f ^ (u64)(long)d ^ strlen_(buf);
}
