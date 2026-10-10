/* Message-passing litmus test on the translated mp.s: a writer and a reader thread; with x86's memory
 * order (VP_TSO on a weakly ordered host) the reader never sees the flag ahead of the data, and its spin
 * loop on the flag ends (a plain load could be hoisted out of it). Prints the count; exit 0 when it is 0. */
#include "vp_emit.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ROUNDS 20000000ull
static uint8_t* block;
static uint64_t result[2];
static void* worker(void* arg) {
    const uint64_t role = (uint64_t)(uintptr_t)arg;
    uint8_t* stack = malloc(1 << 16);
    VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
    c.r[VP_RSP] = (((uint64_t)(uintptr_t)stack + (1 << 16)) & ~15ull) - 8;
    vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    c.r[VP_RDI] = (uint64_t)(uintptr_t)block;
    c.r[VP_RSI] = ROUNDS;
    c.r[VP_RDX] = role;
    if (vp_run(&c, 0x400000)) { fprintf(stderr, "fault: %s at %#llx\n", c.fault_what, (unsigned long long)c.fault_rip); exit(1); }
    result[role] = c.r[VP_RAX];
    free(stack);
    return NULL;
}
int main(void) {
    VpModule* m = vp_first_module(); if (m && !m->attached) vp_module_set_base(m, m->link_base);
    block = aligned_alloc(128, 128); memset(block, 0, 128);
    pthread_t r, w;
    pthread_create(&r, NULL, worker, (void*)1);
    pthread_create(&w, NULL, worker, (void*)0);
    pthread_join(w, NULL);
    pthread_join(r, NULL);
    printf("message passing, %llu rounds (VP_TSO=%d): flag seen ahead of data %llu times: %s\n", ROUNDS, VP_TSO,
           (unsigned long long)result[1], result[1] ? "REORDERED" : "OK");
    return result[1] != 0;
}
