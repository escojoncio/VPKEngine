/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * runtime/vp_host.c with other threads at work (build with -fsanitize=thread to see the races):
 *  - natives registered while another thread dispatches to natives already there;
 *  - the fake TSC read from two threads;
 *  - the missing-entries log, which must stop growing once its table of seen addresses is full.
 *   host_races DIR
 */
#include "vp_emit.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long calls;
static void counted(VpCpu* c) { __atomic_fetch_add(&calls, 1, __ATOMIC_RELAXED); c->rip = VP_HOST_EXIT_ADDRESS; }

#define FIXED_LO UINT64_C(0x7ffe100000)
#define FIXED_N 64
#define NEW_N 3000
#define ROUNDS 20000

static void* registrar(void* arg) {
    (void)arg;
    /* An order that makes the table shift: addresses interleaved with the fixed ones. */
    for (unsigned i = 0; i < NEW_N; ++i) {
        const unsigned k = (i * 7919u) % NEW_N;
        vp_register_native(FIXED_LO + 0x1000 + k * 16 + 8, counted);
        if (i % 16 == 0) (void)vp_rdtsc(NULL);
    }
    return NULL;
}

static void* dispatcher(void* arg) {
    (void)arg;
    VpCpu c;
    memset(&c, 0, sizeof c);
    for (unsigned i = 0; i < ROUNDS; ++i) {
        vp_call_native(&c, FIXED_LO + (i % FIXED_N) * 16);
        if (i % 16 == 0) (void)vp_rdtsc(&c);
    }
    return NULL;
}

static int natives_race(void) {
    for (unsigned i = 0; i < FIXED_N; ++i) vp_register_native(FIXED_LO + i * 16, counted);
    pthread_t a, b;
    pthread_create(&a, NULL, registrar, NULL);
    pthread_create(&b, NULL, dispatcher, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    if (calls != ROUNDS) { printf("FAIL natives_race: %lu calls of %u\n", calls, ROUNDS); return 1; }
    VpCpu c;
    memset(&c, 0, sizeof c);
    calls = 0;
    for (unsigned k = 0; k < NEW_N; ++k) vp_call_native(&c, FIXED_LO + 0x1000 + k * 16 + 8);
    if (calls != NEW_N) { printf("FAIL natives_race: %lu of %u new natives found\n", calls, NEW_N); return 1; }
    printf("ok   natives_race\n");
    return 0;
}

/* A module of 1 MB with no entries: every address in it is "missing". */
static VpModule empty_module;
#define MOD_BASE UINT64_C(0x600000000)
static uint64_t next_missing;
static void walk(VpCpu* c) { c->rip = next_missing; }

static int missing_log(const char* dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/missing.txt", dir);
    remove(path);
    vp_set_missing_log(path);
    empty_module.name = "empty";
    empty_module.link_base = MOD_BASE;
    empty_module.size = 1 << 20;
    empty_module.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(&empty_module);
    vp_module_set_base(&empty_module, MOD_BASE);
    const uint64_t entry = UINT64_C(0x7ffe200000);
    vp_register_native(entry, walk);
    VpCpu c;
    memset(&c, 0, sizeof c);
    /* 5000 different missing addresses, then one of them 100 times more. */
    for (unsigned i = 0; i < 5000; ++i) { next_missing = MOD_BASE + 8 * (i + 1); vp_run(&c, entry); }
    for (unsigned i = 0; i < 100; ++i) { next_missing = MOD_BASE + 8 * 5000; vp_run(&c, entry); }
    FILE* f = fopen(path, "r");
    unsigned lines = 0;
    for (int ch; f && (ch = fgetc(f)) != EOF;) lines += ch == '\n';
    if (f) fclose(f);
    if (lines > 4096) { printf("FAIL missing_log: %u lines, the table of seen addresses holds 4096\n", lines); return 1; }
    printf("ok   missing_log (%u lines)\n", lines);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    int bad = natives_race();
    bad += missing_log(argv[1]);
    return bad ? 1 : 0;
}
