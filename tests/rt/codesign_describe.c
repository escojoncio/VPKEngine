/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * runtime/vp_codesign.c reading signatures it did not write (vp_codesign_describe,
 * vp_codesign_file_team, vp_codesign_file_requirements): a hand-made arm64 Mach-O whose
 * SuperBlob has a CMS slot shorter than its own 8-byte header, then thousands of copies of a
 * plausible signature with bytes changed. Build with -fsanitize=address,undefined; the run is the
 * test.
 *   codesign_describe DIR
 */
#include "vp_codesign.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void be32(uint8_t* p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static void le32(uint8_t* p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

#define SIG_AT 64u /* after the header (32) and LC_CODE_SIGNATURE (16), padded to 16 */

/* A Mach-O of 64 bytes + `sig_size` bytes of signature. */
static size_t macho(uint8_t* f, const uint8_t* sig, uint32_t sig_size) {
    memset(f, 0, SIG_AT);
    le32(f, 0xfeedfacfu); le32(f + 4, 0x0100000cu); le32(f + 16, 1); le32(f + 20, 16);
    le32(f + 32, 0x1d); le32(f + 36, 16); le32(f + 40, SIG_AT); le32(f + 44, sig_size);
    memcpy(f + SIG_AT, sig, sig_size);
    return SIG_AT + sig_size;
}

/* A SuperBlob with one slot of `type` at the very end of the blob, holding `len` bytes after `magic`. */
static uint32_t blob(uint8_t* b, uint32_t type, uint32_t magic, uint32_t len, uint32_t body) {
    be32(b, 0xfade0cc0u); be32(b + 8, 1); be32(b + 12, type); be32(b + 16, 20);
    be32(b + 20, magic); be32(b + 24, len);
    memset(b + 28, 0, body);
    be32(b + 4, 28 + body);
    return 28 + body;
}

static int write_file(const char* path, const uint8_t* d, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    const int ok = fwrite(d, 1, n, f) == n;
    return fclose(f) == 0 && ok ? 0 : -1;
}

static void try_all(const char* path) {
    static char out[8192];
    uint8_t req[4096];
    vp_codesign_describe(path, out, sizeof out);
    (void)vp_codesign_file_team(path, out, sizeof out);
    (void)vp_codesign_file_requirements(path, req, sizeof req);
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    char path[512];
    snprintf(path, sizeof path, "%s/cs.macho", argv[1]);
    static uint8_t sig[1 << 14], file[1 << 15];

    /* The CMS slot (0x10000) says it is 4 bytes long: less than its header. */
    uint32_t n = blob(sig, 0x10000, 0xfade0b01u, 4, 0);
    write_file(path, file, macho(file, sig, n));
    try_all(path);
    printf("ok   cms_shorter_than_header\n");

    /* A CodeDirectory-shaped slot, then a few thousand variants with bytes changed. */
    n = blob(sig, 0, 0xfade0c02u, 120, 120 - 8);
    be32(sig + 28 + 0, 0x20400); be32(sig + 28 + 4, 0); be32(sig + 28 + 8, 80); be32(sig + 28 + 12, 52); /* version.. */
    uint64_t s = 88172645463325252ull;
    uint8_t m[sizeof sig];
    for (unsigned i = 0; i < 4000; ++i) {
        memcpy(m, sig, n);
        for (int k = 0; k < 1 + (int)(i % 4); ++k) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            const size_t at = 4 + s % (n - 4); /* keep the magic of the SuperBlob */
            m[at] = (uint8_t)(s >> 20);
            if (s & 0x100000) m[at] = 0xff;
        }
        write_file(path, file, macho(file, m, n));
        try_all(path);
    }
    printf("ok   mutated_signatures\n");
    remove(path);
    return 0;
}
