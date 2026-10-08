/* Runs the benchmark image natively (x86-64) and translated and prints both times. */
#include "vp_emit.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#define CODE 0x400000ull
#define STACK 0x7f000000ull
#define DATA 0x10000000ull
#define DATA_SIZE (16ull << 20)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void* fixed(uint64_t at, uint64_t size, int prot) {
    void* p = mmap((void*)(uintptr_t)at, size, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(2); } return p;
}
static void setup(VpCpu* c, uint64_t ret) {
    memset(c, 0, sizeof *c); c->mxcsr = 0x1f80;
    c->r[VP_RSP] = STACK + 0x100000 - 0x100 - 8; vp_st64(c->r[VP_RSP], ret);
    memset((void*)(uintptr_t)DATA, 0, DATA_SIZE);
}
#if defined(__x86_64__)
extern void vp_native_run(VpCpu* cpu, uint64_t code);
extern uint64_t vp_native_exit_address(void);
#endif
int main(int argc, char** argv) {
    fixed(CODE, 0x10000, PROT_READ | PROT_WRITE | PROT_EXEC); fixed(STACK, 0x100000, PROT_READ | PROT_WRITE); fixed(DATA, DATA_SIZE, PROT_READ | PROT_WRITE);
    FILE* f = fopen(argv[1], "rb"); if (!f) return 2; fread((void*)(uintptr_t)CODE, 1, 0x10000, f); fclose(f);
    VpCpu c; double t0, tt, tn = 0;
    setup(&c, VP_HOST_EXIT_ADDRESS); t0 = now(); if (vp_run(&c, CODE)) { fprintf(stderr, "translated fault %s at %#" PRIx64 "\n", c.fault_what, c.fault_rip); return 1; } tt = now() - t0;
    const uint64_t rt = c.r[VP_RAX];
#if defined(__x86_64__)
    setup(&c, vp_native_exit_address()); t0 = now(); vp_native_run(&c, CODE); tn = now() - t0;
    printf("native:     %.3f s  (rax=%016" PRIx64 ")\n", tn, c.r[VP_RAX]);
    if (c.r[VP_RAX] != rt) { printf("MISMATCH\n"); return 1; }
#endif
    printf("translated: %.3f s  (rax=%016" PRIx64 ")\n", tt, rt);
    if (tn > 0) printf("translated / native: %.2fx slower\n", tt / tn);
    return 0;
}
