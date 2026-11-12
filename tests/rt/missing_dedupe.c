/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The missing-entries log (vpaot --roots reads it) gets each "module+0xOFFSET" once: an entry an
 * earlier session already wrote, or that the app added after translating it on demand, is not
 * written again when the runtime misses it in this one.
 *   missing_dedupe DIR
 */
#include "vp_emit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static VpModule module;
#define MOD_BASE UINT64_C(0x600000000)
static uint64_t next_missing;
static void walk(VpCpu* c) { c->rip = next_missing; }

/* How many lines of the file are exactly `line` (with its newline). */
static unsigned count(const char* path, const char* line) {
    FILE* f = fopen(path, "r");
    char buf[256];
    unsigned n = 0;
    while (f && fgets(buf, sizeof buf, f)) n += !strcmp(buf, line);
    if (f) fclose(f);
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    char path[512];
    snprintf(path, sizeof path, "%s/missing_dedupe.txt", argv[1]);
    FILE* f = fopen(path, "w"); /* what earlier sessions left (the second line is another module's) */
    if (!f) return 2;
    fputs("dedupe+0x40\nother+0x40\ndedupe+0x80\n", f);
    fclose(f);
    vp_set_missing_log(path);
    module.name = "dedupe";
    module.link_base = MOD_BASE;
    module.size = 1 << 20;
    module.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(&module);
    vp_module_set_base(&module, MOD_BASE);
    const uint64_t entry = UINT64_C(0x7ffe200000);
    vp_register_native(entry, walk);
    VpCpu c;
    memset(&c, 0, sizeof c);
    const uint64_t offsets[] = {0x40, 0x80, 0x100, 0x100, 0x40};
    for (unsigned i = 0; i < sizeof offsets / sizeof offsets[0]; ++i) { next_missing = MOD_BASE + offsets[i]; vp_run(&c, entry); }
    int bad = 0;
    if (count(path, "dedupe+0x40\n") != 1) { printf("FAIL: dedupe+0x40 is %u times in the log\n", count(path, "dedupe+0x40\n")); bad = 1; }
    if (count(path, "dedupe+0x80\n") != 1) { printf("FAIL: dedupe+0x80 is %u times\n", count(path, "dedupe+0x80\n")); bad = 1; }
    if (count(path, "dedupe+0x100\n") != 1) { printf("FAIL: dedupe+0x100 is %u times in the log\n", count(path, "dedupe+0x100\n")); bad = 1; }
    if (!bad) printf("ok   missing_dedupe\n");
    remove(path);
    return bad;
}
