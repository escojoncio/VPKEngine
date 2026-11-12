/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * A load that fails (the code is not the translated one, or too many imports) must leave no native
 * registered for the stubs it handed out: they are not reachable from any image, and the table of
 * natives is not unlimited. A load of the same image that works afterwards gets working stubs.
 * Each case runs in a child process: the first stub of a process is VP_IMPORT_STUB_BASE.
 *   loader_rollback DIR
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define BASE UINT64_C(0x500000000)
#define IMG 0x200000u
#define SLOTS 0x80000u
#define MANY 4100u

static void native_a(VpCpu* c) { c->r[VP_RAX] = 0xa; c->rip = 0; }
static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)name; (void)data; (void)user;
    if (function) *native = native_a;
    return 1;
}

/* Whether a call of the stub reaches a native (in a grandchild: an unregistered one aborts). */
static int registered(uint64_t stub) {
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(2);
        VpCpu c;
        memset(&c, 0, sizeof c);
        vp_call_native(&c, stub);
        _exit(c.r[VP_RAX] == 0xa ? 0 : 3);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static int write_many(const char* path) {
    uint8_t* f = calloc(1, IMG);
    const uint16_t type = 3, machine = 62, phs = 56, phn = 2;
    const uint64_t phoff = 64, entry = BASE + 0x100;
    memcpy(f, "\x7f" "ELF\x02\x01\x01", 7);
    memcpy(f + 16, &type, 2); memcpy(f + 18, &machine, 2); memcpy(f + 24, &entry, 8); memcpy(f + 32, &phoff, 8);
    memcpy(f + 54, &phs, 2); memcpy(f + 56, &phn, 2);
    const TestPhdr ph[2] = {{1, 7, 0, BASE, BASE, IMG, IMG, 0x1000}, {2, 6, 0x400, BASE + 0x400, BASE + 0x400, 0x40, 0x40, 8}};
    memcpy(f + 64, ph, sizeof ph);
    const uint64_t dyn[6] = {7, BASE + 0x1000, 8, (uint64_t)MANY * 24, 0, 0};
    memcpy(f + 0x400, dyn, sizeof dyn);
    for (unsigned i = 0; i < MANY; ++i) {
        const uint64_t r[3] = {BASE + SLOTS + 8 * i, ((uint64_t)1 << 32) | 6, 0};
        memcpy(f + 0x1000 + 24 * i, r, 24);
    }
    FILE* o = fopen(path, "wb");
    const int ok = o && fwrite(f, 1, IMG, o) == IMG;
    if (o) fclose(o);
    free(f);
    return ok ? 0 : -1;
}

static int case_fingerprint(const char* dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/rollback.elf", dir);
    const TestRela r = {BASE + 0x600, ((uint64_t)1 << 32) | 6, 0};
    if (elfgen(path, BASE, &r, 1, sizeof r, NULL, 0)) return 2;
    VpImport im = {"imported", BASE + 0x600, 1};
    VpRange code = {0x100, 0x10};
    VpModule m;
    memset(&m, 0, sizeof m);
    m.name = "rollback";
    m.link_base = BASE;
    m.size = TEST_IMG;
    m.imports = &im;
    m.import_count = 1;
    m.code = &code;
    m.code_count = 1;
    m.fingerprint = 12345; /* not the hash of the code in the file */
    vp_register_module(&m);
    VpLoadedImage out;
    if (vp_load_module(path, &m, 0, resolve, NULL, &out) == 0) return 3;
    if (registered(VP_IMPORT_STUB_BASE)) return 4; /* the failed load's native is still there */
    static const unsigned char zeros[0x200]; /* the image's code bytes are all zero */
    m.fingerprint = vp_fingerprint((uint64_t)(uintptr_t)zeros, &code, 1, NULL, 0); /* now the right one */
    if (vp_load_module(path, &m, 0, resolve, NULL, &out)) return 5;
    uint64_t stub;
    memcpy(&stub, (void*)(uintptr_t)(BASE + 0x600), 8);
    if (!registered(stub)) return 6; /* the retry's stub does not reach the native */
    return 0;
}

static int case_too_many(const char* dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/rollback_many.elf", dir);
    if (write_many(path)) return 2;
    static VpImport im[MANY];
    static char names[MANY][16];
    for (unsigned i = 0; i < MANY; ++i) {
        snprintf(names[i], 16, "sym_%u", i);
        im[i] = (VpImport){names[i], SLOTS + 8 * i, 1};
    }
    VpModule m;
    memset(&m, 0, sizeof m);
    m.name = "rollback_many";
    m.link_base = BASE; m.size = IMG; m.imports = im; m.import_count = MANY; m.relative = 1;
    m.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(&m);
    VpLoadedImage out;
    if (vp_load_module(path, &m, 0, resolve, NULL, &out) == 0) return 3;
    if (registered(VP_IMPORT_STUB_BASE) || registered(VP_IMPORT_STUB_BASE + 4095 * VP_IMPORT_STUB_STRIDE)) return 4;
    return 0;
}

static int run(const char* dir, const char* name, int (*fn)(const char*)) {
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) _exit(fn(dir));
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) { printf("FAIL %s: crashed (signal %d)\n", name, WTERMSIG(status)); return 1; }
    if (WEXITSTATUS(status)) { printf("FAIL %s: step %d (2 setup, 3 load accepted, 4 stub left registered, 5/6 retry)\n", name, WEXITSTATUS(status)); return 1; }
    printf("ok   %s\n", name);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    int bad = run(argv[1], "rollback_fingerprint", case_fingerprint);
    bad += run(argv[1], "rollback_too_many", case_too_many);
    return bad != 0;
}
