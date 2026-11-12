/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * vp_load_module on a thread with a small stack (a game's loader thread has 512 KB, and part of
 * that is its thread-local state): its working tables belong on the heap. The thread here has
 * 128 KB, which cannot hold them (they were ~118 KB); a crash is the failure.
 *   loader_stack DIR
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define BASE UINT64_C(0x500000000)
static void native_a(VpCpu* c) { c->r[VP_RAX] = 0xa; c->rip = 0; }
static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)name; (void)function; (void)data; (void)user;
    *native = native_a;
    return 1;
}

static const char* dir;
static int result = 1;
static void* loader(void* arg) {
    (void)arg;
    char path[512];
    snprintf(path, sizeof path, "%s/stack.elf", dir);
    const TestRela r = {BASE + 0x600, ((uint64_t)1 << 32) | 6, 0};
    if (elfgen(path, BASE, &r, 1, sizeof r, NULL, 0)) { result = 2; return NULL; }
    VpImport im = {"sym", BASE + 0x600, 1};
    static VpModule m;
    m.name = "stack";
    m.link_base = BASE;
    m.size = TEST_IMG;
    m.imports = &im;
    m.import_count = 1;
    m.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(&m);
    VpLoadedImage out;
    result = vp_load_module(path, &m, 0, resolve, NULL, &out);
    if (result) printf("load: %s\n", out.error);
    remove(path);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    dir = argv[1];
    pthread_attr_t attr;
    pthread_t t;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 128 * 1024);
    if (pthread_create(&t, &attr, loader, NULL)) return 2;
    pthread_join(t, NULL);
    printf(result ? "loader_stack: FAILED\n" : "loader_stack: ok\n");
    return result != 0;
}
