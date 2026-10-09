/* Runs the translated C++ program (exceptions through its own static libsupc++/libgcc_eh) from its
 * entry point and compares rax with the same C++ compiled natively. */
#include "vp_emit.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
typedef unsigned long u64;
u64 vp_main_native(u64* mem);
#define STACK 0x7f000000ull
#define DATA 0x20000000ull
static void* fixed(uint64_t at, uint64_t size, int prot) {
    void* p = mmap((void*)(uintptr_t)(at & ~0xfffull), size + (at & 0xfff), prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(2); } return p;
}
int main(int argc, char** argv) {
    if (argc < 2) return 2;
    FILE* f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 2; }
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
    fixed(STACK, 0x100000, PROT_READ | PROT_WRITE);
    u64* mem = fixed(DATA, 1 << 20, PROT_READ | PROT_WRITE);
    VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
    c.r[VP_RSP] = STACK + 0x100000 - 0x100 - 8; vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    c.r[VP_RDI] = DATA;
    /* x86-64 TLS (variant II): the thread pointer points at itself, the static TLS block lies
     * below it, the stack-protector canary is at fs:0x28. */
    static uint64_t tls_area[1024];
    uint64_t* tp = &tls_area[768];
    tp[0] = (uint64_t)(uintptr_t)tp;
    tp[5] = 0x5eed5eed5eed5e00ull;
    c.fs_base = (uint64_t)(uintptr_t)tp;
    const int fault = vp_run(&c, entry);
    if (fault) { fprintf(stderr, "translated: %s at %#" PRIx64 "\n", c.fault_what, c.fault_rip); return 1; }
    u64* mem2 = malloc(1 << 20); memset(mem2, 0, 1 << 20);
    const u64 expected = vp_main_native(mem2);
    printf("translated rax=%016" PRIx64 " native=%016" PRIx64 " %s\n", c.r[VP_RAX], expected, c.r[VP_RAX] == expected ? "OK" : "MISMATCH");
    return c.r[VP_RAX] == expected ? 0 : 1;
}
