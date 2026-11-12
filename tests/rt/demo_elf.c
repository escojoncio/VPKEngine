/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The demo app's host (apps/Demo/App/Bridge/demo.c) with malformed ELF files: it must refuse them
 * (no read outside the file, no leak). Build with -fsanitize=address,undefined; the run is the test.
 * Usage: demo_elf  (writes its files in $TMPDIR or /tmp)
 */
#include "../../apps/Demo/App/Bridge/demo.h"
#include "elfgen.h"
#include "vp_emit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Stubs: the translated program and the native build are not under test. */
int vp_run(VpCpu* cpu, uint64_t entry) { (void)cpu; (void)entry; return 0; }
unsigned long vp_main_native(unsigned long* mem) { (void)mem; return 0; }

static int failures;
static void expect_refused(const char* name, const char* path) {
    const vp_demo_result r = vp_demo_run(path);
    if (r.ok || r.translated_ms != 0) { printf("FAIL %s: accepted (%s)\n", name, r.message); ++failures; }
    else printf("ok   %s: %s\n", name, r.message);
}

int main(void) {
    const char* tmp = getenv("TMPDIR");
    char path[512];
    snprintf(path, sizeof path, "%s/demo_elf.bin", tmp ? tmp : "/tmp");

    /* shorter than an ELF header */
    FILE* f = fopen(path, "wb"); fputs("\x7f" "ELF", f); fclose(f);
    expect_refused("short file", path);

    /* program headers beyond the file */
    if (elfgen(path, 0x500000000ull, NULL, 0, 0, NULL, 0)) return 2;
    f = fopen(path, "r+b"); uint64_t phoff = 0x10000; fseek(f, 32, SEEK_SET); fwrite(&phoff, 8, 1, f); fclose(f);
    expect_refused("phoff beyond the file", path);

    /* a segment whose file bytes are past the end of the file */
    TestPhdr big = {1, 7, 0xfff, 0x600000000ull, 0, 0x2000, 0x2000, 0x1000};
    if (elfgen(path, 0x500000000ull, NULL, 0, 0, &big, 1)) return 2;
    expect_refused("filesz beyond the file", path);

    /* a segment whose address plus size wraps */
    TestPhdr wrap = {1, 7, 0, UINT64_MAX - 0x10, 0, 0x10, 0x100, 0x1000};
    if (elfgen(path, 0x500000000ull, NULL, 0, 0, &wrap, 1)) return 2;
    expect_refused("vaddr+memsz wraps", path);

    /* a mapping that cannot be made (the kernel refuses an address above the user space) */
    TestPhdr high = {1, 7, 0, 0xffffffffff000000ull, 0, 0, 0x1000, 0x1000};
    if (elfgen(path, 0x500000000ull, NULL, 0, 0, &high, 1)) return 2;
    expect_refused("unmappable address", path);

    remove(path);
    printf(failures ? "demo_elf: %d FAILED\n" : "demo_elf: ok\n", failures);
    return failures != 0;
}
