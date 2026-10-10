/* Translation on demand: the program's translation lacks a function (tests/aot/roots/prog.c); when
 * the run reaches it, the miss hook translates just that function (vpaot --fragment, with the
 * entries the module already has as --known), compiles it into a shared library, loads it (its
 * module registers itself) and the run goes on. Result compared with the native program. */
#include "vp_emit.h"
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
typedef unsigned long u64;
u64 vp_main_native(u64* mem);
#define STACK 0x7f000000ull
#define DATA 0x20000000ull
static const char *vpaot, *elf_path, *build, *runtime;
static int fragments;
static void* fixed(uint64_t at, uint64_t size, int prot) {
    void* p = mmap((void*)(uintptr_t)(at & ~0xfffull), size + (at & 0xfff), prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(2); } return p;
}
static int on_demand(VpCpu* cpu, uint64_t target) {
    VpModule* m = vp_module_at(target);
    if (!m) return 0;
    const uint64_t off = target - m->base;
    char known[512], cmd[4096], name[64], so[512];
    snprintf(known, sizeof known, "%s/known.txt", build);
    FILE* k = fopen(known, "w");
    for (VpModule* x = vp_module_list(); x; x = x->next)
        for (size_t i = 0; i < x->entry_count; ++i) fprintf(k, "%" PRIx64 "\n", x->entries[i].guest - x->link_base);
    fclose(k);
    snprintf(name, sizeof name, "%s_ondemand_%" PRIx64, m->name, off);
    snprintf(so, sizeof so, "%s/%s.so", build, name);
    snprintf(cmd, sizeof cmd,
             "%s --elf %s --module %s --fragment --known %s --entry %#" PRIx64 " --out %s/%s.c && "
             "%s --registry %s/%s_reg.c --pack ondemand %s && "
             "cc -O2 -frounding-math -fPIC -shared -I %s -o %s %s/%s.c %s/%s_reg.c",
             vpaot, elf_path, name, known, target, build, name, vpaot, build, name, name, runtime, so, build, name, build, name);
    fprintf(stderr, "on demand: %s+%#" PRIx64 "\n", m->name, off);
    if (system(cmd) != 0) return 0;
    if (!dlopen(so, RTLD_NOW | RTLD_LOCAL)) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 0; }
    ++fragments;
    vp_dispatch(cpu, target);
    return 1;
}
int main(int argc, char** argv) {
    if (argc < 5) return 2;
    elf_path = argv[1]; vpaot = argv[2]; build = argv[3]; runtime = argv[4];
    FILE* f = fopen(elf_path, "rb"); if (!f) { perror(elf_path); return 2; }
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* elf = malloc(size); if (fread(elf, 1, size, f) != (size_t)size) return 2; fclose(f);
    uint64_t entry, phoff; uint16_t phnum; memcpy(&entry, elf + 24, 8); memcpy(&phoff, elf + 32, 8); memcpy(&phnum, elf + 56, 2);
    for (uint16_t i = 0; i < phnum; ++i) {
        unsigned char* ph = elf + phoff + i * 56; uint32_t type; uint64_t off, vaddr, filesz, memsz;
        memcpy(&type, ph, 4); memcpy(&off, ph + 8, 8); memcpy(&vaddr, ph + 16, 8); memcpy(&filesz, ph + 32, 8); memcpy(&memsz, ph + 40, 8);
        if (type != 1 || !memsz) continue;
        fixed(vaddr, memsz, PROT_READ | PROT_WRITE);
        memcpy((void*)(uintptr_t)vaddr, elf + off, filesz);
    }
    static const VpEmbedderHooks hooks = {on_demand, NULL, NULL};
    vp_set_embedder_hooks(&hooks);
    fixed(STACK, 0x100000, PROT_READ | PROT_WRITE);
    u64* mem = fixed(DATA, 1 << 20, PROT_READ | PROT_WRITE);
    VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
    c.r[VP_RSP] = STACK + 0x100000 - 0x100 - 8; vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    c.r[VP_RDI] = DATA;
    const int fault = vp_run(&c, entry);
    if (fault) { fprintf(stderr, "translated: %s at %#" PRIx64 "\n", c.fault_what, c.fault_rip); return 1; }
    u64* mem2 = malloc(1 << 20); memset(mem2, 0, 1 << 20);
    const u64 expected = vp_main_native(mem2);
    const int ok = c.r[VP_RAX] == expected && fragments == 1;
    printf("on demand: %d fragment(s), translated rax=%016" PRIx64 " native=%016" PRIx64 " %s\n", fragments, c.r[VP_RAX], expected, ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}
