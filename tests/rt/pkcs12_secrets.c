/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * runtime/vp_pkcs12.c (built with -DVP_PKCS12_OPENSSL):
 *  - wipe: the password (as UTF-16, and the copies the key derivation makes) and the private key
 *    are gone from every heap block when it is freed. Link with -Wl,--wrap=free and build with
 *    -O2 and no sanitizer (a plain memset before free is dropped by the optimizer; ASan forbids
 *    reading a block past what was asked for).
 *  - threads: the first reads on several threads at once load the OpenSSL providers (legacy and
 *    default) once, not once per thread (and with TSan, with no race).
 *   pkcs12_secrets wipe|threads FILE.p12 PASSWORD
 */
#include "vp_pkcs12.h"
#include <malloc.h>
#include <openssl/provider.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void __real_free(void* p);
OSSL_PROVIDER* __real_OSSL_PROVIDER_load(OSSL_LIB_CTX* ctx, const char* name);
static int provider_loads;
OSSL_PROVIDER* __wrap_OSSL_PROVIDER_load(OSSL_LIB_CTX* ctx, const char* name) {
    __atomic_fetch_add(&provider_loads, 1, __ATOMIC_RELAXED);
    return __real_OSSL_PROVIDER_load(ctx, name);
}
static uint8_t pw_pattern[64], key_pattern[32];
static size_t pw_pattern_len;
static int scanning, pw_hits, key_hits;

static int contains(const uint8_t* p, size_t n, const uint8_t* pat, size_t m) {
    for (size_t i = 0; i + m <= n; ++i) if (!memcmp(p + i, pat, m)) return 1;
    return 0;
}

void __wrap_free(void* p) {
    if (p && scanning) {
        const size_t n = malloc_usable_size(p);
        if (contains(p, n, pw_pattern, pw_pattern_len)) ++pw_hits;
        if (contains(p, n, key_pattern, sizeof key_pattern)) ++key_hits;
    }
    __real_free(p);
}

/* UTF-8 of one or two bytes per character, as UTF-16BE. */
static size_t bmp(const char* s, uint8_t* out) {
    size_t at = 0;
    for (; *s; ++s) {
        unsigned cp = (uint8_t)*s;
        if (cp >= 0xc0) { cp = (cp & 0x1f) << 6 | ((uint8_t)*++s & 0x3f); }
        out[at++] = (uint8_t)(cp >> 8); out[at++] = (uint8_t)cp;
    }
    return at;
}

static uint8_t* slurp(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* d = malloc(*n);
    if (!d || fread(d, 1, *n, f) != *n) { fclose(f); return NULL; }
    fclose(f);
    return d;
}

static const uint8_t* g_der; static size_t g_len; static const char* g_pw;
static pthread_barrier_t start;
static void* reader(void* arg) {
    (void)arg;
    pthread_barrier_wait(&start);
    for (int i = 0; i < 2; ++i) {
        VpPkcs12 p; char err[128];
        if (vp_pkcs12_read(g_der, g_len, g_pw, &p, err, sizeof err)) { printf("read failed: %s\n", err); exit(2); }
        vp_pkcs12_free(&p);
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 4) return 2;
    if (!(g_der = slurp(argv[2], &g_len))) return 2;
    g_pw = argv[3];
    if (!strcmp(argv[1], "threads")) {
        pthread_t t[8];
        pthread_barrier_init(&start, NULL, 8);
        for (int i = 0; i < 8; ++i) pthread_create(&t[i], NULL, reader, NULL);
        for (int i = 0; i < 8; ++i) pthread_join(t[i], NULL);
        if (provider_loads != 2) { printf("FAIL pkcs12_threads: %d provider loads, 2 expected (legacy, default)\n", provider_loads); return 1; }
        printf("ok   pkcs12_threads\n");
        return 0;
    }
    VpPkcs12 p; char err[128];
    if (vp_pkcs12_read(g_der, g_len, g_pw, &p, err, sizeof err) || p.key_len < 200) { printf("read failed: %s\n", err); return 2; }
    memcpy(key_pattern, p.key + p.key_len / 2, sizeof key_pattern);
    pw_pattern_len = bmp(g_pw, pw_pattern);
    vp_pkcs12_free(&p);
    {  /* Control: a block that still holds both secrets must be seen when freed, or the scan below proves nothing (the wrap
        * not applied, free resolved elsewhere). */
        void (*volatile release)(void*) = free;
        uint8_t* c = malloc(256);
        memset(c, 0, 256);
        memcpy(c + 10, pw_pattern, pw_pattern_len);
        memcpy(c + 100, key_pattern, sizeof key_pattern);
        scanning = 1;
        release(c);
        scanning = 0;
        if (pw_hits != 1 || key_hits != 1) { printf("FAIL pkcs12_wipe control: the scan saw %d/%d of 1/1 planted secrets\n", pw_hits, key_hits); return 2; }
        pw_hits = key_hits = 0;
    }
    scanning = 1;
    if (vp_pkcs12_read(g_der, g_len, g_pw, &p, err, sizeof err)) { printf("read failed: %s\n", err); return 2; }
    vp_pkcs12_free(&p);
    scanning = 0;
    if (pw_hits || key_hits) { printf("FAIL pkcs12_wipe (%s): %d blocks freed with the password, %d with the key\n", argv[2], pw_hits, key_hits); return 1; }
    printf("ok   pkcs12_wipe (%s)\n", argv[2]);
    return 0;
}
