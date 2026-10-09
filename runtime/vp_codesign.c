/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Code signing of a thin arm64 Mach-O with a developer certificate (see vp_codesign.h): an
 * embedded signature (SuperBlob) with a SHA-256 CodeDirectory, an empty requirement set, and a
 * detached CMS SignedData over the CodeDirectory carrying the certificates and Apple's cdhash
 * attributes. Written from the published formats (xnu's cs_blobs.h, RFC 5652), not from another
 * signer.
 */
#include "vp_codesign.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ---- SHA-256 ------------------------------------------------------------------------------ */

typedef struct {
    uint32_t h[8];
    uint64_t bytes;
    uint8_t buf[64];
    size_t fill;
} VpSha256;

static const uint32_t vp_sha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define VP_ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void vp_sha_block(VpSha256* s, const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = VP_ROR(w[i - 15], 7) ^ VP_ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = VP_ROR(w[i - 2], 17) ^ VP_ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t t1 = h + (VP_ROR(e, 6) ^ VP_ROR(e, 11) ^ VP_ROR(e, 25)) + ((e & f) ^ (~e & g)) + vp_sha_k[i] + w[i];
        const uint32_t t2 = (VP_ROR(a, 2) ^ VP_ROR(a, 13) ^ VP_ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void vp_sha_init(VpSha256* s) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, iv, sizeof iv);
    s->bytes = 0;
    s->fill = 0;
}

static void vp_sha_update(VpSha256* s, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    s->bytes += len;
    if (s->fill) {
        const size_t take = len < 64 - s->fill ? len : 64 - s->fill;
        memcpy(s->buf + s->fill, p, take);
        s->fill += take; p += take; len -= take;
        if (s->fill == 64) { vp_sha_block(s, s->buf); s->fill = 0; }
    }
    while (len >= 64) { vp_sha_block(s, p); p += 64; len -= 64; }
    if (len) { memcpy(s->buf, p, len); s->fill = len; }
}

static void vp_sha_final(VpSha256* s, uint8_t out[32]) {
    const uint64_t bits = s->bytes * 8;
    uint8_t pad = 0x80;
    vp_sha_update(s, &pad, 1);
    pad = 0;
    while (s->fill != 56) vp_sha_update(s, &pad, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = (uint8_t)(bits >> (56 - 8 * i));
    vp_sha_update(s, len, 8);
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = (uint8_t)(s->h[i] >> 24); out[4 * i + 1] = (uint8_t)(s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(s->h[i] >> 8); out[4 * i + 3] = (uint8_t)s->h[i];
    }
}

void vp_sha256(const void* data, size_t len, uint8_t out[32]) {
    VpSha256 s;
    vp_sha_init(&s);
    vp_sha_update(&s, data, len);
    vp_sha_final(&s, out);
}

/* ---- byte buffers and DER ----------------------------------------------------------------- */

typedef struct {
    uint8_t* p;
    size_t n, cap;
    int oom;
} VpBuf;

static void vb_put(VpBuf* b, const void* data, size_t len) {
    if (b->oom) return;
    if (b->n + len > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n + len) cap *= 2;
        uint8_t* q = (uint8_t*)realloc(b->p, cap);
        if (!q) { b->oom = 1; return; }
        b->p = q;
        b->cap = cap;
    }
    if (len) memcpy(b->p + b->n, data, len);
    b->n += len;
}
static void vb_byte(VpBuf* b, uint8_t v) { vb_put(b, &v, 1); }
static void vb_be32(VpBuf* b, uint32_t v) { uint8_t x[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v}; vb_put(b, x, 4); }
static void vb_be64(VpBuf* b, uint64_t v) { vb_be32(b, (uint32_t)(v >> 32)); vb_be32(b, (uint32_t)v); }
static void vb_free(VpBuf* b) { free(b->p); b->p = NULL; b->n = b->cap = 0; }

/* A complete TLV: tag, DER length, content. */
static void der_tlv(VpBuf* out, uint8_t tag, const uint8_t* content, size_t len) {
    vb_byte(out, tag);
    if (len < 0x80) {
        vb_byte(out, (uint8_t)len);
    } else {
        uint8_t l[8];
        int k = 0;
        for (size_t v = len; v; v >>= 8) l[k++] = (uint8_t)v;
        vb_byte(out, (uint8_t)(0x80 | k));
        while (k) vb_byte(out, l[--k]);
    }
    vb_put(out, content, len);
}
/* Wraps `inner` (consumed) as one TLV appended to `out`. */
static void der_wrap(VpBuf* out, uint8_t tag, VpBuf* inner) {
    der_tlv(out, tag, inner->p, inner->n);
    if (inner->oom) out->oom = 1;
    vb_free(inner);
}
static void der_oid(VpBuf* out, const char* dotted) {
    uint8_t enc[64];
    size_t n = 0;
    unsigned long arcs[32];
    int count = 0;
    for (const char* c = dotted; *c && count < 32;) {
        arcs[count++] = strtoul(c, (char**)&c, 10);
        if (*c == '.') ++c;
    }
    unsigned long first = arcs[0] * 40 + arcs[1];
    for (int i = 1; i < count; ++i) {
        unsigned long v = i == 1 ? first : arcs[i];
        uint8_t tmp[10];
        int k = 0;
        do { tmp[k++] = (uint8_t)(v & 0x7f); v >>= 7; } while (v);
        while (k) { --k; enc[n++] = (uint8_t)(tmp[k] | (k ? 0x80 : 0)); }
    }
    der_tlv(out, 0x06, enc, n);
}
static void der_null(VpBuf* out) { der_tlv(out, 0x05, NULL, 0); }
static void der_int_small(VpBuf* out, uint8_t v) { der_tlv(out, 0x02, &v, 1); }
/* AlgorithmIdentifier { oid, NULL? } */
static void der_alg(VpBuf* out, const char* oid, int with_null) {
    VpBuf a = {0};
    der_oid(&a, oid);
    if (with_null) der_null(&a);
    der_wrap(out, 0x30, &a);
}

/* DER reader: the element at p (bounded by end). 0 on success. */
typedef struct {
    uint8_t tag;
    const uint8_t* start;   /* the tag byte */
    const uint8_t* content;
    size_t len;             /* content length */
    size_t total;           /* tag + length + content */
} DerElem;

static int der_read(const uint8_t* p, const uint8_t* end, DerElem* e) {
    if (p >= end || end - p < 2) return -1;
    e->start = p;
    e->tag = p[0];
    size_t len = p[1];
    const uint8_t* q = p + 2;
    if (len & 0x80) {
        const int k = (int)(len & 0x7f);
        if (k == 0 || k > 4 || end - q < k) return -1;
        len = 0;
        for (int i = 0; i < k; ++i) len = (len << 8) | *q++;
    }
    if ((size_t)(end - q) < len) return -1;
    e->content = q;
    e->len = len;
    e->total = (size_t)(q - p) + len;
    return 0;
}

/* The pieces of a certificate a signer needs: issuer Name and serial INTEGER (raw DER, as the
 * SignerInfo repeats them) and the subject's OU (the team identifier). */
typedef struct {
    const uint8_t* issuer; size_t issuer_len;
    const uint8_t* serial; size_t serial_len;
    char ou[64];
} CertInfo;

static int cert_parse(const uint8_t* der, size_t len, CertInfo* ci) {
    memset(ci, 0, sizeof *ci);
    const uint8_t* end = der + len;
    DerElem cert, tbs, e;
    if (der_read(der, end, &cert) || cert.tag != 0x30) return -1;
    if (der_read(cert.content, cert.content + cert.len, &tbs) || tbs.tag != 0x30) return -1;
    const uint8_t* p = tbs.content;
    const uint8_t* te = tbs.content + tbs.len;
    if (der_read(p, te, &e)) return -1;
    if (e.tag == 0xa0) { p += e.total; if (der_read(p, te, &e)) return -1; } /* version */
    if (e.tag != 0x02) return -1;
    ci->serial = e.start; ci->serial_len = e.total; p += e.total;
    if (der_read(p, te, &e) || e.tag != 0x30) return -1;
    p += e.total; /* signature algorithm */
    if (der_read(p, te, &e) || e.tag != 0x30) return -1;                              /* issuer */
    ci->issuer = e.start; ci->issuer_len = e.total; p += e.total;
    if (der_read(p, te, &e) || e.tag != 0x30) return -1;
    p += e.total; /* validity */
    if (der_read(p, te, &e) || e.tag != 0x30) return -1;                              /* subject */
    /* Name: SEQUENCE OF SET OF SEQUENCE { OID, value } */
    const uint8_t* np = e.content;
    const uint8_t* ne = e.content + e.len;
    static const uint8_t ou_oid[] = {0x06, 0x03, 0x55, 0x04, 0x0b}; /* 2.5.4.11 */
    while (np < ne) {
        DerElem set, atv, oid, val;
        if (der_read(np, ne, &set) || set.tag != 0x31) return -1;
        const uint8_t* sp = set.content;
        const uint8_t* se = set.content + set.len;
        while (sp < se) {
            if (der_read(sp, se, &atv) || atv.tag != 0x30) return -1;
            if (der_read(atv.content, atv.content + atv.len, &oid)) return -1;
            if (der_read(oid.start + oid.total, atv.content + atv.len, &val)) return -1;
            if (oid.total == sizeof ou_oid && !memcmp(oid.start, ou_oid, sizeof ou_oid) && val.len < sizeof ci->ou) {
                memcpy(ci->ou, val.content, val.len);
                ci->ou[val.len] = 0;
            }
            sp += atv.total;
        }
        np += set.total;
    }
    return 0;
}

/* ---- Mach-O ------------------------------------------------------------------------------- */

#define MH_MAGIC_64 0xfeedfacfu
#define LC_SEGMENT_64 0x19u
#define LC_CODE_SIGNATURE 0x1du
#define CS_PAGE_SHIFT 12
#define CS_PAGE (1u << CS_PAGE_SHIFT)

static uint32_t rd32(const uint8_t* p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t rd64(const uint8_t* p) { return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32; }
static void wr32(uint8_t* p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static void wr64(uint8_t* p, uint64_t v) { wr32(p, (uint32_t)v); wr32(p + 4, (uint32_t)(v >> 32)); }
static uint32_t be32(const uint8_t* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

struct VpCodesign {
    int fd;
    uint32_t dataoff, datasize;     /* the new signature's place in the file */
    VpBuf cd;                       /* CodeDirectory blob */
    VpBuf req;                      /* requirements blob */
    VpBuf attrs;                    /* signed attributes, SET form (what is signed) */
    VpBuf certs;                    /* leaf + chain, concatenated DER */
    CertInfo leaf;
    uint8_t* leaf_copy;
    size_t leaf_len;
};

static void set_err(char* err, size_t n, const char* f, ...) {
    if (!err || !n) return;
    va_list ap;
    va_start(ap, f);
    vsnprintf(err, n, f, ap);
    va_end(ap);
}

static int pread_all(int fd, void* buf, size_t len, off_t at) {
    uint8_t* p = (uint8_t*)buf;
    while (len) {
        const ssize_t r = pread(fd, p, len, at);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        p += r; len -= (size_t)r; at += r;
    }
    return 0;
}
static int pwrite_all(int fd, const void* buf, size_t len, off_t at) {
    const uint8_t* p = (const uint8_t*)buf;
    while (len) {
        const ssize_t r = pwrite(fd, p, len, at);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        p += r; len -= (size_t)r; at += r;
    }
    return 0;
}

/* Signed attributes: DER SET OF, sorted by encoding (X.690 11.6). */
static int cmp_der(const void* a, const void* b) {
    const VpBuf* x = (const VpBuf*)a;
    const VpBuf* y = (const VpBuf*)b;
    const size_t n = x->n < y->n ? x->n : y->n;
    const int c = memcmp(x->p, y->p, n);
    return c ? c : (x->n < y->n ? -1 : x->n > y->n);
}

static void attr(VpBuf* out, const char* oid, VpBuf* value_set_content) {
    VpBuf a = {0};
    der_oid(&a, oid);
    der_wrap(&a, 0x31, value_set_content);
    der_wrap(out, 0x30, &a);
}

static void base64(VpBuf* out, const uint8_t* p, size_t n) {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        char c[4] = {t[v >> 18], t[(v >> 12) & 63], i + 1 < n ? t[(v >> 6) & 63] : '=', i + 2 < n ? t[v & 63] : '='};
        vb_put(out, c, 4);
    }
}

static void build_attrs(VpCodesign* s) {
    uint8_t cd_hash[32];
    vp_sha256(s->cd.p, s->cd.n, cd_hash);
    VpBuf items[5] = {{0}};
    VpBuf v = {0};
    /* contentType = data */
    der_oid(&v, "1.2.840.113549.1.7.1");
    attr(&items[0], "1.2.840.113549.1.9.3", &v);
    /* signingTime */
    char t[32];
    const time_t now = time(NULL);
    struct tm g;
    gmtime_r(&now, &g);
    snprintf(t, sizeof t, "%02d%02d%02d%02d%02d%02dZ", g.tm_year % 100, g.tm_mon + 1, g.tm_mday, g.tm_hour, g.tm_min, g.tm_sec);
    der_tlv(&v, 0x17, (const uint8_t*)t, 13);
    attr(&items[1], "1.2.840.113549.1.9.5", &v);
    /* messageDigest = SHA-256 of the CodeDirectory */
    der_tlv(&v, 0x04, cd_hash, 32);
    attr(&items[2], "1.2.840.113549.1.9.4", &v);
    /* Apple: the CodeDirectory hashes as a property list (truncated to 20 bytes)... */
    VpBuf plist = {0};
    static const char head[] = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                               "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
                               "<plist version=\"1.0\">\n<dict>\n\t<key>cdhashes</key>\n\t<array>\n\t\t<data>\n\t\t";
    static const char tail[] = "\n\t\t</data>\n\t</array>\n</dict>\n</plist>\n";
    vb_put(&plist, head, sizeof head - 1);
    base64(&plist, cd_hash, 20);
    vb_put(&plist, tail, sizeof tail - 1);
    der_tlv(&v, 0x04, plist.p, plist.n);
    vb_free(&plist);
    attr(&items[3], "1.2.840.113635.100.9.1", &v);
    /* ... and as (hash algorithm, full hash) */
    VpBuf h = {0};
    der_oid(&h, "2.16.840.1.101.3.4.2.1");
    der_tlv(&h, 0x04, cd_hash, 32);
    der_wrap(&v, 0x30, &h);
    attr(&items[4], "1.2.840.113635.100.9.2", &v);
    qsort(items, 5, sizeof items[0], cmp_der);
    VpBuf set = {0};
    for (int i = 0; i < 5; ++i) { vb_put(&set, items[i].p, items[i].n); if (items[i].oom) set.oom = 1; vb_free(&items[i]); }
    der_wrap(&s->attrs, 0x31, &set);
}

/* The whole CMS ContentInfo (SignedData, detached, one signer). */
static void build_cms(VpCodesign* s, const uint8_t* sig, size_t sig_len, int is_ec, VpBuf* out) {
    VpBuf si = {0};
    der_int_small(&si, 1);
    VpBuf ias = {0};
    vb_put(&ias, s->leaf.issuer, s->leaf.issuer_len);
    vb_put(&ias, s->leaf.serial, s->leaf.serial_len);
    der_wrap(&si, 0x30, &ias);
    der_alg(&si, "2.16.840.1.101.3.4.2.1", 1);
    /* signedAttrs [0] IMPLICIT: the same content as the SET that was signed */
    DerElem e;
    der_read(s->attrs.p, s->attrs.p + s->attrs.n, &e);
    der_tlv(&si, 0xa0, e.content, e.len);
    if (is_ec) der_alg(&si, "1.2.840.10045.4.3.2", 0); else der_alg(&si, "1.2.840.113549.1.1.1", 1);
    der_tlv(&si, 0x04, sig, sig_len);
    VpBuf sis = {0};
    der_wrap(&sis, 0x30, &si);

    VpBuf sd = {0};
    der_int_small(&sd, 1);
    VpBuf algs = {0};
    der_alg(&algs, "2.16.840.1.101.3.4.2.1", 1);
    der_wrap(&sd, 0x31, &algs);
    VpBuf eci = {0};
    der_oid(&eci, "1.2.840.113549.1.7.1");
    der_wrap(&sd, 0x30, &eci);
    der_tlv(&sd, 0xa0, s->certs.p, s->certs.n);
    der_wrap(&sd, 0x31, &sis);
    VpBuf sdseq = {0};
    der_wrap(&sdseq, 0x30, &sd);

    VpBuf ci = {0};
    der_oid(&ci, "1.2.840.113549.1.7.2");
    der_wrap(&ci, 0xa0, &sdseq);
    der_wrap(out, 0x30, &ci);
}

/* Room for the CMS: certificates + names + attributes + a signature of up to 4096-bit RSA. */
static size_t cms_room(const VpCodesign* s) {
    return s->certs.n + 2 * s->leaf.issuer_len + s->leaf.serial_len + s->attrs.n + 512 + 512;
}

void vp_codesign_abort(VpCodesign* s) {
    if (!s) return;
    if (s->fd >= 0) close(s->fd);
    vb_free(&s->cd); vb_free(&s->req); vb_free(&s->attrs); vb_free(&s->certs);
    free(s->leaf_copy);
    free(s);
}

VpCodesign* vp_codesign_begin(const char* path, const char* identifier, const uint8_t* leaf, size_t leaf_len,
                              const uint8_t* const* chain, const size_t* chain_lens, int chain_count,
                              char* err, size_t err_len) {
    VpCodesign* s = (VpCodesign*)calloc(1, sizeof *s);
    if (!s) { set_err(err, err_len, "out of memory"); return NULL; }
    s->fd = -1;
    s->leaf_copy = (uint8_t*)malloc(leaf_len ? leaf_len : 1);
    if (!s->leaf_copy) { set_err(err, err_len, "out of memory"); vp_codesign_abort(s); return NULL; }
    memcpy(s->leaf_copy, leaf, leaf_len);
    s->leaf_len = leaf_len;
    if (cert_parse(s->leaf_copy, leaf_len, &s->leaf)) { set_err(err, err_len, "the certificate is not readable DER"); vp_codesign_abort(s); return NULL; }
    if (!s->leaf.ou[0]) { set_err(err, err_len, "the certificate has no team (subject OU)"); vp_codesign_abort(s); return NULL; }
    vb_put(&s->certs, leaf, leaf_len);
    for (int i = 0; i < chain_count; ++i) vb_put(&s->certs, chain[i], chain_lens[i]);

    s->fd = open(path, O_RDWR);
    if (s->fd < 0) { set_err(err, err_len, "cannot open %s: %s", path, strerror(errno)); vp_codesign_abort(s); return NULL; }
    struct stat st;
    if (fstat(s->fd, &st) || st.st_size < 32) { set_err(err, err_len, "not a Mach-O file"); vp_codesign_abort(s); return NULL; }
    uint8_t mh[32];
    if (pread_all(s->fd, mh, 32, 0) || rd32(mh) != MH_MAGIC_64 || rd32(mh + 4) != 0x0100000cu) {
        set_err(err, err_len, "not a thin arm64 Mach-O");
        vp_codesign_abort(s);
        return NULL;
    }
    const uint32_t ncmds = rd32(mh + 16), sizeofcmds = rd32(mh + 20);
    if (sizeofcmds > (uint32_t)st.st_size - 32 || sizeofcmds > (64u << 20)) { set_err(err, err_len, "bad load commands"); vp_codesign_abort(s); return NULL; }
    uint8_t* lc = (uint8_t*)malloc(sizeofcmds);
    if (!lc || pread_all(s->fd, lc, sizeofcmds, 32)) { free(lc); set_err(err, err_len, "cannot read load commands"); vp_codesign_abort(s); return NULL; }
    uint8_t* cs = NULL;
    uint8_t* linkedit = NULL;
    uint64_t text_off = 0, text_size = 0;
    uint32_t off = 0;
    for (uint32_t i = 0; i < ncmds; ++i) {
        if (off + 8 > sizeofcmds) break;
        const uint32_t cmd = rd32(lc + off), size = rd32(lc + off + 4);
        if (size < 8 || size > sizeofcmds - off) break; /* off + 8 <= sizeofcmds: no wrap */
        if (cmd == LC_CODE_SIGNATURE && size >= 16) cs = lc + off;
        if (cmd == LC_SEGMENT_64 && size >= 72) {
            if (!strncmp((const char*)lc + off + 8, "__LINKEDIT", 16)) linkedit = lc + off;
            if (!strncmp((const char*)lc + off + 8, "__TEXT", 16)) { text_off = rd64(lc + off + 40); text_size = rd64(lc + off + 48); }
        }
        off += size;
    }
    if (!cs || !linkedit) {
        free(lc);
        set_err(err, err_len, "no LC_CODE_SIGNATURE or __LINKEDIT (link the pack with -adhoc_codesign)");
        vp_codesign_abort(s);
        return NULL;
    }
    s->dataoff = rd32(cs + 8);
    const uint64_t le_off = rd64(linkedit + 40);
    if (s->dataoff < le_off || s->dataoff > (uint64_t)st.st_size || (s->dataoff & 15)) {
        free(lc);
        set_err(err, err_len, "the signature is not at the end of __LINKEDIT");
        vp_codesign_abort(s);
        return NULL;
    }
    const uint32_t code_limit = s->dataoff;
    const uint32_t nslots = (code_limit + CS_PAGE - 1) / CS_PAGE;

    /* Requirements: an empty set (the designated requirement is then the implicit one). */
    vb_be32(&s->req, 0xfade0c01u); vb_be32(&s->req, 12); vb_be32(&s->req, 0);

    /* The CodeDirectory's size is known before its contents: lay out the new signature first, so
     * that the headers (which the page hashes cover) say its final size. */
    const char* team = s->leaf.ou;
    const uint32_t ident_off = 88;
    const uint32_t team_off = ident_off + (uint32_t)strlen(identifier) + 1;
    const uint32_t nspecial = 2;
    const uint32_t hash_off = team_off + (uint32_t)strlen(team) + 1 + nspecial * 32;
    const uint32_t cd_len = hash_off + nslots * 32;
    /* The attributes have a fixed size (the hash values do not change it): build them once from
     * a placeholder CodeDirectory of the right size, to size the CMS. */
    s->cd.n = 0;
    { uint8_t* z = (uint8_t*)calloc(1, cd_len); if (z) { vb_put(&s->cd, z, cd_len); free(z); } }
    build_attrs(s);
    const size_t cms = cms_room(s);
    const uint32_t sb_header = 12 + 3 * 8;
    uint32_t total = sb_header + cd_len + (uint32_t)s->req.n + 8 + (uint32_t)cms;
    total = (total + 15) & ~15u;
    s->datasize = total;
    vb_free(&s->attrs);
    vb_free(&s->cd);

    /* Headers: the signature's new size, __LINKEDIT to the new end of the file. */
    wr32(cs + 12, s->datasize);
    const uint64_t le_filesize = (uint64_t)s->dataoff + s->datasize - le_off;
    wr64(linkedit + 32, (le_filesize + 0x3fff) & ~(uint64_t)0x3fff); /* vmsize */
    wr64(linkedit + 48, le_filesize);                                 /* filesize */
    if (pwrite_all(s->fd, lc, sizeofcmds, 32) || ftruncate(s->fd, (off_t)s->dataoff + s->datasize)) {
        free(lc);
        set_err(err, err_len, "cannot write %s: %s", path, strerror(errno));
        vp_codesign_abort(s);
        return NULL;
    }
    free(lc);

    /* The CodeDirectory. */
    VpBuf* cd = &s->cd;
    vb_be32(cd, 0xfade0c02u); vb_be32(cd, cd_len); vb_be32(cd, 0x20400); vb_be32(cd, 0); /* flags */
    vb_be32(cd, hash_off); vb_be32(cd, ident_off); vb_be32(cd, nspecial); vb_be32(cd, nslots);
    vb_be32(cd, code_limit);
    vb_byte(cd, 32); vb_byte(cd, 2); vb_byte(cd, 0); vb_byte(cd, CS_PAGE_SHIFT); /* SHA-256, 4 KiB pages */
    vb_be32(cd, 0);              /* spare2 */
    vb_be32(cd, 0);              /* scatterOffset */
    vb_be32(cd, team_off);
    vb_be32(cd, 0);              /* spare3 */
    vb_be64(cd, 0);              /* codeLimit64 */
    vb_be64(cd, text_off); vb_be64(cd, text_size); vb_be64(cd, 0); /* exec segment: __TEXT, a library */
    vb_put(cd, identifier, strlen(identifier) + 1);
    vb_put(cd, team, strlen(team) + 1);
    uint8_t h[32];
    vp_sha256(s->req.p, s->req.n, h);
    vb_put(cd, h, 32);                       /* slot -2: requirements */
    memset(h, 0, 32); vb_put(cd, h, 32);     /* slot -1: Info.plist (none) */
    uint8_t* page = (uint8_t*)malloc(CS_PAGE);
    if (!page) { set_err(err, err_len, "out of memory"); vp_codesign_abort(s); return NULL; }
    for (uint32_t i = 0; i < nslots; ++i) {
        const uint32_t at = i * CS_PAGE;
        const uint32_t n = code_limit - at < CS_PAGE ? code_limit - at : CS_PAGE;
        if (pread_all(s->fd, page, n, at)) { free(page); set_err(err, err_len, "cannot read page %u", i); vp_codesign_abort(s); return NULL; }
        vp_sha256(page, n, h);
        vb_put(cd, h, 32);
    }
    free(page);
    if (cd->oom || cd->n != cd_len) { set_err(err, err_len, "internal: CodeDirectory size"); vp_codesign_abort(s); return NULL; }
    build_attrs(s);
    if (s->attrs.oom || s->certs.oom) { set_err(err, err_len, "out of memory"); vp_codesign_abort(s); return NULL; }
    return s;
}

const uint8_t* vp_codesign_to_sign(VpCodesign* s, size_t* len) {
    *len = s->attrs.n;
    return s->attrs.p;
}

const char* vp_codesign_team(VpCodesign* s) { return s->leaf.ou; }

int vp_codesign_finish(VpCodesign* s, const uint8_t* signature, size_t signature_len, int is_ec, char* err, size_t err_len) {
    VpBuf cmsder = {0};
    build_cms(s, signature, signature_len, is_ec, &cmsder);
    const uint32_t sb_header = 12 + 3 * 8;
    const uint32_t cd_at = sb_header, req_at = cd_at + (uint32_t)s->cd.n, cms_at = req_at + (uint32_t)s->req.n;
    const uint32_t room = s->datasize - cms_at;
    if (cmsder.oom || cmsder.n + 8 > room) {
        set_err(err, err_len, "the signature does not fit (%zu bytes for %u)", cmsder.n, room);
        vb_free(&cmsder);
        vp_codesign_abort(s);
        return -1;
    }
    /* Lengths as Apple's codesign writes them: the CMS blob is its DER, the SuperBlob ends there;
     * the zeros up to the reserved size follow outside it. */
    const uint32_t cms_blob = 8 + (uint32_t)cmsder.n;
    VpBuf sb = {0};
    vb_be32(&sb, 0xfade0cc0u); vb_be32(&sb, cms_at + cms_blob); vb_be32(&sb, 3);
    vb_be32(&sb, 0); vb_be32(&sb, cd_at);
    vb_be32(&sb, 2); vb_be32(&sb, req_at);
    vb_be32(&sb, 0x10000); vb_be32(&sb, cms_at);
    vb_put(&sb, s->cd.p, s->cd.n);
    vb_put(&sb, s->req.p, s->req.n);
    vb_be32(&sb, 0xfade0b01u); vb_be32(&sb, cms_blob);
    vb_put(&sb, cmsder.p, cmsder.n);
    vb_free(&cmsder);
    while (sb.n < s->datasize && !sb.oom) vb_byte(&sb, 0);
    int r = 0;
    if (sb.oom || pwrite_all(s->fd, sb.p, sb.n, s->dataoff) || fsync(s->fd)) {
        set_err(err, err_len, "cannot write the signature: %s", strerror(errno));
        r = -1;
    }
    vb_free(&sb);
    vp_codesign_abort(s);
    return r;
}

int vp_codesign_cert_team(const uint8_t* der, size_t len, char* out, size_t out_len) {
    CertInfo ci;
    if (!out || !out_len) return -1;
    out[0] = 0;
    if (!der || cert_parse(der, len, &ci) || !ci.ou[0] || strlen(ci.ou) + 1 > out_len) return -1;
    memcpy(out, ci.ou, strlen(ci.ou) + 1);
    return 0;
}

/* The arm64 slice of a universal (fat) file, or the file itself: its offset in the file. */
static int arm64_slice(int fd, uint64_t* base) {
    uint8_t h[8];
    *base = 0;
    if (pread_all(fd, h, 8, 0)) return -1;
    const uint32_t magic = be32(h);
    if (magic != 0xcafebabeu && magic != 0xcafebabfu) return 0;
    const int wide = magic == 0xcafebabfu;
    const uint32_t n = be32(h + 4);
    if (n > 64) return -1;
    const uint32_t entry = wide ? 32 : 20;
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t a[32];
        if (pread_all(fd, a, entry, 8 + (off_t)i * entry)) return -1;
        if (be32(a) != 0x0100000cu) continue; /* CPU_TYPE_ARM64 */
        *base = wide ? ((uint64_t)be32(a + 8) << 32 | be32(a + 12)) : be32(a + 8);
        return 0;
    }
    return -1;
}

int vp_codesign_file_team(const char* path, char* out, size_t out_len) {
    if (!out || !out_len) return -1;
    out[0] = 0;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int r = -1;
    uint8_t mh[32];
    uint8_t* lc = NULL;
    uint8_t* sig = NULL;
    uint64_t base = 0;
    if (arm64_slice(fd, &base) || pread_all(fd, mh, 32, (off_t)base) || rd32(mh) != MH_MAGIC_64) goto done;
    {
        const uint32_t ncmds = rd32(mh + 16), sizeofcmds = rd32(mh + 20);
        if (sizeofcmds > (64u << 20)) goto done;
        lc = (uint8_t*)malloc(sizeofcmds);
        if (!lc || pread_all(fd, lc, sizeofcmds, (off_t)(base + 32))) goto done;
        uint32_t dataoff = 0, datasize = 0, off = 0;
        for (uint32_t i = 0; i < ncmds && off + 8 <= sizeofcmds; ++i) {
            const uint32_t cmd = rd32(lc + off), size = rd32(lc + off + 4);
            if (size < 8 || size > sizeofcmds - off) break;
            if (cmd == LC_CODE_SIGNATURE && size >= 16) { dataoff = rd32(lc + off + 8); datasize = rd32(lc + off + 12); }
            off += size;
        }
        if (datasize < 12 || datasize > (16u << 20)) goto done;
        sig = (uint8_t*)malloc(datasize);
        if (!sig || pread_all(fd, sig, datasize, (off_t)(base + dataoff)) || be32(sig) != 0xfade0cc0u) goto done;
        const uint32_t count = be32(sig + 8);
        for (uint32_t i = 0; i < count && i < (datasize - 12) / 8; ++i) {
            const uint32_t type = be32(sig + 12 + 8 * i), at = be32(sig + 16 + 8 * i);
            if ((type != 0 && (type < 0x1000 || type > 0x1004)) || at > datasize || datasize - at < 52) continue;
            const uint8_t* cd = sig + at;
            if (be32(cd) != 0xfade0c02u || be32(cd + 8) < 0x20200) continue;
            const uint32_t len = be32(cd + 4), team_off = be32(cd + 48), flags = be32(cd + 12);
            if (len < 52 || len > datasize - at) continue;
            if ((flags & 2) || !team_off || team_off >= len) continue; /* ad hoc: no team */
            const char* t = (const char*)cd + team_off;
            const size_t n = strnlen(t, len - team_off);
            if (n == len - team_off || n + 1 > out_len) continue;
            memcpy(out, t, n + 1);
            r = 0;
            break;
        }
    }
done:
    free(lc);
    free(sig);
    close(fd);
    return r;
}
