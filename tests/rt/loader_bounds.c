/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * vp_loader.c with images whose 64-bit fields wrap around: each case runs in a child process and
 * must end with the loader's answer (rejected, or loaded with the bad record skipped), not with a
 * crash from a write or read outside the image.
 *   loader_bounds DIR
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define BASE UINT64_C(0x500000000)

static VpModule make_module(const char* name) {
    VpModule m;
    memset(&m, 0, sizeof m);
    m.name = name;
    m.link_base = BASE;
    m.size = TEST_IMG;
    m.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    return m;
}

/* expect_ok: 1 the load must succeed, 0 it must be refused. Returns 0 on the right outcome. */
static int run_case(const char* dir, const char* name, int expect_ok, const TestRela* rela, unsigned nrela, uint64_t rela_size,
                    const TestPhdr* extra, unsigned nextra) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.elf", dir, name);
    if (elfgen(path, BASE, rela, nrela, rela_size, extra, nextra)) return 2;
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        VpModule m = make_module(name);
        vp_register_module(&m);
        VpLoadedImage out;
        const int r = vp_load_module(path, &m, 0, NULL, NULL, &out);
        _exit((r == 0) == (expect_ok != 0) ? 0 : 3);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) { printf("FAIL %s: crashed (signal %d)\n", name, WTERMSIG(status)); return 1; }
    if (WEXITSTATUS(status)) { printf("FAIL %s: wrong answer (%d)\n", name, WEXITSTATUS(status)); return 1; }
    printf("ok   %s\n", name);
    return 0;
}

/* A load that fails after the image was mapped (the code is not the translated one) must give the
 * mapping back: it holds the whole image, up to 2 GB. */
static int run_unmaps(const char* dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/fingerprint.elf", dir);
    if (elfgen(path, BASE, NULL, 0, 0, NULL, 0)) return 2;
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        VpModule m = make_module("fingerprint");
        VpRange code = {0x100, 0x10};
        m.code = &code;
        m.code_count = 1;
        m.fingerprint = 12345; /* not the hash of the code in the file */
        vp_register_module(&m);
        VpLoadedImage out;
        if (vp_load_module(path, &m, 0, NULL, NULL, &out) == 0) _exit(3);
        _exit(msync((void*)(uintptr_t)BASE, 4096, MS_ASYNC) != 0 ? 0 : 4); /* ENOMEM: not mapped */
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status) || WEXITSTATUS(status)) { printf("FAIL fingerprint_unmaps: %d\n", WIFSIGNALED(status) ? -WTERMSIG(status) : WEXITSTATUS(status)); return 1; }
    printf("ok   fingerprint_unmaps\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    int bad = 0;
    /* A segment whose vaddr + memsz wraps: the image range stays small, the copy would go to the
     * end of the address space. */
    const TestPhdr wrap = {1, 7, 0, UINT64_C(0xffffffffffffff00), 0, 0x100, 0x200, 0x1000};
    bad += run_case(argv[1], "segment_wraps", 0, NULL, 0, 0, &wrap, 1);
    /* A relocation whose target + 8 wraps. */
    const TestRela far = {UINT64_C(0xfffffffffffffffc), 8, 0};
    bad += run_case(argv[1], "reloc_target_wraps", 1, &far, 1, sizeof far, NULL, 0);
    /* A relocation table size that wraps when added to its address. */
    bad += run_case(argv[1], "reloc_size_wraps", 1, NULL, 0, UINT64_C(0xfffffffffffffff0), NULL, 0);
    /* A sane one still loads. */
    const TestRela ok = {BASE + 0x600, 8, 0x10};
    bad += run_case(argv[1], "plain", 1, &ok, 1, sizeof ok, NULL, 0);
    bad += run_unmaps(argv[1]);
    return bad ? 1 : 0;
}
