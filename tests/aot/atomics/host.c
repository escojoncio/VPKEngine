/* Locked operations under contention: the translated contend() on 8 threads at once over one
 * shared block; every total must be exact (lost updates would show as a smaller count). */
#include "vp_emit.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#define THREADS 8
#define ITERS 200000
static uint8_t* block;
static void* worker(void* arg) {
    (void)arg;
    uint8_t* stack = malloc(1 << 16);
    VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
    c.r[VP_RSP] = (((uint64_t)(uintptr_t)stack + (1 << 16)) & ~15ull) - 8;
    vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    c.r[VP_RDI] = (uint64_t)(uintptr_t)block;
    c.r[VP_RSI] = ITERS;
    if (vp_run(&c, 0x400000)) { fprintf(stderr, "fault: %s at %#llx\n", c.fault_what, (unsigned long long)c.fault_rip); exit(1); }
    free(stack);
    return NULL;
}
int main(void) {
    VpModule* m = vp_first_module(); if (m && !m->attached) vp_module_set_base(m, m->link_base);
    block = aligned_alloc(64, 256); memset(block, 0, 256);
    pthread_t t[THREADS];
    for (int i = 0; i < THREADS; ++i) pthread_create(&t[i], NULL, worker, NULL);
    for (int i = 0; i < THREADS; ++i) pthread_join(t[i], NULL);
    uint64_t q0, q20, q40, q48, q58, q60, q68; uint32_t d13, d2d;
    memcpy(&q0, block, 8); memcpy(&d13, block + 0x13, 4); memcpy(&q20, block + 0x20, 8); memcpy(&d2d, block + 0x2d, 4);
    memcpy(&q40, block + 0x40, 8); memcpy(&q48, block + 0x48, 8); memcpy(&q58, block + 0x58, 8); memcpy(&q60, block + 0x60, 8); memcpy(&q68, block + 0x68, 8);
    const uint64_t n = (uint64_t)THREADS * ITERS;
    int ok = q0 == n && d13 == 2 * n && q20 == 3 * n && d2d == n && q40 == 0 && q58 == n && q60 == 1 && q68 == 0;
    printf("locked ops, %d threads x %d: inc %llu, split add %u, xadd %llu, split inc %u, cmpxchg %llu, neg %llu, xor %llu, bt %llu: %s\n",
           THREADS, ITERS, (unsigned long long)q0, d13, (unsigned long long)q20, d2d, (unsigned long long)q58,
           (unsigned long long)q40, (unsigned long long)q68, (unsigned long long)q48, ok ? "OK" : "LOST UPDATES");
    return ok ? 0 : 1;
}
