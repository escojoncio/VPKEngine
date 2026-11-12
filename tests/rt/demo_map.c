/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The demo app's host (apps/Demo/App/Bridge/demo.c) maps the guest at fixed addresses: it must not
 * replace memory that is there, and must give back what it mapped on every way out, because the
 * button can be pressed again (a second run must work, also after a failed one).
 *   demo_map  (UBSan; writes its file in $TMPDIR or /tmp)
 */
#include "../../apps/Demo/App/Bridge/demo.h"
#include "elfgen.h"
#include "vp_emit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static int fault_next;
int vp_run(VpCpu* cpu, uint64_t entry) {
    (void)entry;
    if (fault_next) { cpu->fault_what = "test"; cpu->fault_rip = 1; return 1; }
    return 0;
}
unsigned long vp_main_native(unsigned long* mem) { (void)mem; return 0; }

#define GUEST UINT64_C(0x500000000)
#define STACK UINT64_C(0x2f0000000)
#define DATA UINT64_C(0x300000000)
static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL "); printf(__VA_ARGS__); printf("\n"); ++failures; } } while (0)

static int free_at(uint64_t at, uint64_t size) { /* nothing mapped there: a no-replace mapping works, then goes */
    void* p = mmap((void*)(uintptr_t)at, size, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) return 0;
    munmap(p, size);
    return 1;
}

int main(void) {
    const char* tmp = getenv("TMPDIR");
    char path[512];
    snprintf(path, sizeof path, "%s/demo_map.bin", tmp ? tmp : "/tmp");
    if (elfgen(path, GUEST, NULL, 0, 0, NULL, 0)) return 2;

    for (int i = 0; i < 3; ++i) {
        const vp_demo_result r = vp_demo_run(path);
        CHECK(r.ok, "run %d of the same program: %s", i + 1, r.message);
    }
    CHECK(free_at(GUEST, 0x4000) && free_at(STACK, 0x100000) && free_at(DATA, 1 << 20), "a finished run left its mappings");

    fault_next = 1;
    const vp_demo_result f = vp_demo_run(path);
    CHECK(!f.ok, "a faulting run reported ok");
    CHECK(free_at(GUEST, 0x4000) && free_at(STACK, 0x100000) && free_at(DATA, 1 << 20), "a faulted run left its mappings");
    fault_next = 0;

    /* something already at the guest's address */
    void* s = mmap((void*)(uintptr_t)GUEST, 0x4000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    memset(s, 0x5a, 0x4000);
    const vp_demo_result o = vp_demo_run(path);
    CHECK(!o.ok, "the guest was mapped over existing memory");
    CHECK(((unsigned char*)s)[0] == 0x5a && ((unsigned char*)s)[0x3fff] == 0x5a, "the existing memory was replaced");
    munmap(s, 0x4000);

    /* something at the data address: the guest and the stack mapped before must go */
    s = mmap((void*)(uintptr_t)DATA, 0x4000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    const vp_demo_result d = vp_demo_run(path);
    CHECK(!d.ok, "the data was mapped over existing memory");
    CHECK(free_at(GUEST, 0x4000) && free_at(STACK, 0x100000), "a run that failed half way left its mappings");
    munmap(s, 0x4000);
    CHECK(vp_demo_run(path).ok, "the run after the failures");

    remove(path);
    printf(failures ? "demo_map: %d FAILED\n" : "demo_map: ok\n", failures);
    return failures != 0;
}
