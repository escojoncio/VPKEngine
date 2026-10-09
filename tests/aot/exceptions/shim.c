/* The few C library functions libsupc++/libgcc_eh need, for a freestanding static image. */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <link.h>
#include <dlfcn.h>
static unsigned char heap[1 << 20];
static size_t heap_used;
void* malloc(size_t n) { n = (n + 15) & ~(size_t)15; if (heap_used + n > sizeof heap) return 0; void* p = heap + heap_used; heap_used += n; return p; }
void free(void* p) { (void)p; }
void* calloc(size_t a, size_t b) { unsigned char* p = malloc(a * b); for (size_t i = 0; p && i < a * b; ++i) p[i] = 0; return p; }
void* realloc(void* p, size_t n) { unsigned char* q = malloc(n); if (p && q) for (size_t i = 0; i < n; ++i) q[i] = ((unsigned char*)p)[i]; return q; }
void abort(void) { for (;;) __builtin_trap(); }
void* memcpy(void* d, const void* s, size_t n) { unsigned char* a = d; const unsigned char* b = s; while (n--) *a++ = *b++; return d; }
void* memset(void* d, int c, size_t n) { unsigned char* a = d; while (n--) *a++ = (unsigned char)c; return d; }
int memcmp(const void* x, const void* y, size_t n) { const unsigned char* a = x; const unsigned char* b = y; for (; n--; ++a, ++b) if (*a != *b) return *a - *b; return 0; }
void* memmove(void* d, const void* s, size_t n) { unsigned char* a = d; const unsigned char* b = s; if (a < b) while (n--) *a++ = *b++; else { a += n; b += n; while (n--) *--a = *--b; } return d; }
size_t strlen(const char* s) { size_t n = 0; while (s[n]) ++n; return n; }
int strcmp(const char* a, const char* b) { while (*a && *a == *b) ++a, ++b; return (unsigned char)*a - (unsigned char)*b; }
/* The unwinder finds .eh_frame through the program headers of the (only) image. */
extern const ElfW(Ehdr) __ehdr_start;
int dl_iterate_phdr(int (*cb)(struct dl_phdr_info*, size_t, void*), void* data) {
    struct dl_phdr_info info = {0};
    info.dlpi_addr = 0;
    info.dlpi_name = "";
    info.dlpi_phdr = (const ElfW(Phdr)*)((const char*)&__ehdr_start + __ehdr_start.e_phoff);
    info.dlpi_phnum = __ehdr_start.e_phnum;
    return cb(&info, sizeof info, data);
}
/* Single-threaded: the pthread hooks libgcc probes for are absent (weak). */
int __libc_single_threaded = 1;
/* Hooks libsupc++ references (single-threaded, no environment). */
int pthread_mutex_lock(void* m) { (void)m; return 0; }
int pthread_mutex_unlock(void* m) { (void)m; return 0; }
char* secure_getenv(const char* n) { (void)n; return 0; }
char* strchr(const char* s, int c) { for (;; ++s) { if (*s == (char)c) return (char*)s; if (!*s) return 0; } }
unsigned long __isoc23_strtoul(const char* s, char** e, int b) { (void)b; if (e) *e = (char*)s; return 0; }
void __stack_chk_fail(void) { abort(); }
void* stderr;
size_t fwrite(const void* p, size_t a, size_t b, void* f) { (void)p; (void)f; return a * b; }
int fputs(const char* s, void* f) { (void)s; (void)f; return 0; }
int fputc(int c, void* f) { (void)f; return c; }
/* glibc 2.35+'s fast path of the unwinder: the .eh_frame_hdr of the image holding `pc`. */
int _dl_find_object(void* pc, struct dl_find_object* r) {
    const ElfW(Phdr)* ph = (const ElfW(Phdr)*)((const char*)&__ehdr_start + __ehdr_start.e_phoff);
    uintptr_t lo = ~(uintptr_t)0, hi = 0;
    void* hdr = 0;
    for (int i = 0; i < __ehdr_start.e_phnum; ++i) {
        if (ph[i].p_type == PT_LOAD) {
            if (ph[i].p_vaddr < lo) lo = ph[i].p_vaddr;
            if (ph[i].p_vaddr + ph[i].p_memsz > hi) hi = ph[i].p_vaddr + ph[i].p_memsz;
        }
        if (ph[i].p_type == PT_GNU_EH_FRAME) hdr = (void*)ph[i].p_vaddr;
    }
    if ((uintptr_t)pc < lo || (uintptr_t)pc >= hi || !hdr) return -1;
    memset(r, 0, sizeof *r);
    r->dlfo_map_start = (void*)lo;
    r->dlfo_map_end = (void*)hi;
    r->dlfo_eh_frame = hdr;
    return 0;
}
