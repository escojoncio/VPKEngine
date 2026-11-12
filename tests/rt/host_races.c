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

static unsigned long calls, wrong, lookups;
static volatile int registered_all;
static void counted(VpCpu* c) { __atomic_fetch_add(&calls, 1, __ATOMIC_RELAXED); c->rip = VP_HOST_EXIT_ADDRESS; }
/* The fixed natives answer with their own number, so that a lookup that read the table while it
 * was being shifted (and found the neighbour's entry) is seen. */
static void answer0(VpCpu* c) { c->r[VP_RAX] = 100; c->rip = VP_HOST_EXIT_ADDRESS; }
static void answer1(VpCpu* c) { c->r[VP_RAX] = 101; c->rip = VP_HOST_EXIT_ADDRESS; }

#define FIXED_LO UINT64_C(0x7ffe100000)
#define FIXED_N 64
#define NEW_N 12000
#define NEW_LO (FIXED_LO - 0x100000)

static void* registrar(void* arg) {
    (void)arg;
    /* Below the fixed ones, in an order that makes the table shift: every insertion moves all of
     * them one place up while the dispatcher looks one of them up. */
    for (unsigned i = 0; i < NEW_N; ++i) {
        const unsigned k = (i * 7919u) % NEW_N;
        vp_register_native(NEW_LO + k * 16 + 8, counted);
        if (i % 16 == 0) (void)vp_rdtsc(NULL);
    }
    registered_all = 1;
    return NULL;
}

static void* dispatcher(void* arg) {
    (void)arg;
    VpCpu c;
    memset(&c, 0, sizeof c);
    for (unsigned i = 0; !registered_all || i < FIXED_N; ++i) { /* for as long as the registrations go on */
        c.r[VP_RAX] = 0;
        vp_call_native(&c, FIXED_LO + (i % FIXED_N) * 16);
        if (c.r[VP_RAX] != 100 + (i % FIXED_N) % 2) wrong++;
        lookups++;
        if (i % 16 == 0) (void)vp_rdtsc(&c);
    }
    return NULL;
}

static int natives_race(void) {
    for (unsigned i = 0; i < FIXED_N; ++i) vp_register_native(FIXED_LO + i * 16, i % 2 ? answer1 : answer0);
    pthread_t a, b;
    pthread_create(&a, NULL, registrar, NULL);
    pthread_create(&b, NULL, dispatcher, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    if (wrong) { printf("FAIL natives_race: %lu of %lu lookups answered by the wrong native\n", wrong, lookups); return 1; }
    if (calls != 0) { printf("FAIL natives_race: %lu lookups reached a native that is not the one asked for\n", calls); return 1; }
    VpCpu c;
    memset(&c, 0, sizeof c);
    calls = 0;
    for (unsigned k = 0; k < NEW_N; ++k) vp_call_native(&c, NEW_LO + k * 16 + 8);
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
