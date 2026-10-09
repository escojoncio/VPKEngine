/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * vp_pkcs12.h: a PKCS#12 reader (RFC 7292) for developer certificates. DER only (what OpenSSL,
 * iloader, SideStore and AltSign write); files with BER indefinite lengths (some keychain exports)
 * are left to Apple's SecPKCS12Import, which the app tries first.
 */
#include "vp_pkcs12.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the system's ciphers ---------------------------------------------------------------- */

enum { H_SHA1 = 1, H_SHA256 = 256, H_SHA512 = 512 };
enum { C_AES, C_3DES, C_RC2 };

#if defined(__APPLE__) && !defined(VP_PKCS12_OPENSSL)
#include <CommonCrypto/CommonCrypto.h>

static size_t hash_len(int h) { return h == H_SHA1 ? 20 : h == H_SHA256 ? 32 : 64; }

static void hash(int h, const uint8_t* d, size_t n, uint8_t* out) {
    if (h == H_SHA1) CC_SHA1(d, (CC_LONG)n, out);
    else if (h == H_SHA256) CC_SHA256(d, (CC_LONG)n, out);
    else CC_SHA512(d, (CC_LONG)n, out);
}

static void hmac(int h, const uint8_t* key, size_t klen, const uint8_t* d, size_t n, uint8_t* out) {
    CCHmac(h == H_SHA1 ? kCCHmacAlgSHA1 : h == H_SHA256 ? kCCHmacAlgSHA256 : kCCHmacAlgSHA512, key, klen, d, n, out);
}

static int pbkdf2(int h, const uint8_t* pw, size_t pwlen, const uint8_t* salt, size_t slen, unsigned iter, uint8_t* out, size_t n) {
    return CCKeyDerivationPBKDF(kCCPBKDF2, (const char*)pw, pwlen, salt, slen,
                                h == H_SHA1 ? kCCPRFHmacAlgSHA1 : h == H_SHA256 ? kCCPRFHmacAlgSHA256 : kCCPRFHmacAlgSHA512,
                                iter, out, n) == kCCSuccess ? 0 : -1;
}

/* CBC with PKCS#7 padding: -1 when the padding is wrong (a wrong key, usually). */
static int cbc_decrypt(int c, const uint8_t* key, size_t klen, const uint8_t* iv, const uint8_t* in, size_t n, uint8_t* out, size_t* out_len) {
    const CCAlgorithm alg = c == C_AES ? kCCAlgorithmAES : c == C_3DES ? kCCAlgorithm3DES : kCCAlgorithmRC2;
    size_t moved = 0;
    if (CCCrypt(kCCDecrypt, alg, kCCOptionPKCS7Padding, key, klen, iv, in, n, out, n, &moved) != kCCSuccess) return -1;
    *out_len = moved;
    return 0;
}
#else
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/provider.h>

static size_t hash_len(int h) { return h == H_SHA1 ? 20 : h == H_SHA256 ? 32 : 64; }
static const EVP_MD* md(int h) { return h == H_SHA1 ? EVP_sha1() : h == H_SHA256 ? EVP_sha256() : EVP_sha512(); }

static void hash(int h, const uint8_t* d, size_t n, uint8_t* out) {
    unsigned len = 0;
    EVP_Digest(d, n, out, &len, md(h), NULL);
}

static void hmac(int h, const uint8_t* key, size_t klen, const uint8_t* d, size_t n, uint8_t* out) {
    unsigned len = 0;
    HMAC(md(h), key, (int)klen, d, n, out, &len);
}

static int pbkdf2(int h, const uint8_t* pw, size_t pwlen, const uint8_t* salt, size_t slen, unsigned iter, uint8_t* out, size_t n) {
    return PKCS5_PBKDF2_HMAC((const char*)pw, (int)pwlen, salt, (int)slen, (int)iter, md(h), (int)n, out) == 1 ? 0 : -1;
}

static int cbc_decrypt(int c, const uint8_t* key, size_t klen, const uint8_t* iv, const uint8_t* in, size_t n, uint8_t* out, size_t* out_len) {
    static int providers;
    if (!providers) { OSSL_PROVIDER_load(NULL, "legacy"); OSSL_PROVIDER_load(NULL, "default"); providers = 1; }
    const EVP_CIPHER* cipher = c == C_3DES ? EVP_des_ede3_cbc()
                             : c == C_RC2 ? EVP_rc2_cbc()
                             : klen == 16 ? EVP_aes_128_cbc() : klen == 24 ? EVP_aes_192_cbc() : EVP_aes_256_cbc();
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    int ok = ctx && EVP_DecryptInit_ex(ctx, cipher, NULL, NULL, NULL) == 1;
    if (ok && c == C_RC2) {
        ok = EVP_CIPHER_CTX_set_key_length(ctx, (int)klen) == 1 &&
             EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_SET_RC2_KEY_BITS, (int)klen * 8, NULL) == 1;
    }
    int a = 0, b = 0;
    ok = ok && EVP_DecryptInit_ex(ctx, NULL, NULL, key, iv) == 1 && EVP_DecryptUpdate(ctx, out, &a, in, (int)n) == 1 &&
         EVP_DecryptFinal_ex(ctx, out + a, &b) == 1;
    EVP_CIPHER_CTX_free(ctx);
    if (!ok) return -1;
    *out_len = (size_t)(a + b);
    return 0;
}
#endif

/* ---- DER --------------------------------------------------------------------------------- */

typedef struct { const uint8_t* p; size_t n; } Span;

/* Real files use 2048 to a few hundred thousand: past this, a damaged file would hang the app. */
#define MAX_ITERATIONS 10000000UL

static void say(char* err, size_t err_len, const char* f, ...) {
    if (!err || !err_len) return;
    va_list ap;
    va_start(ap, f);
    vsnprintf(err, err_len, f, ap);
    va_end(ap);
}

/* One TLV off the front of `s`. 0 or -1 (malformed, or an indefinite BER length: -2). */
static int der_next(Span* s, int* tag, Span* value) {
    if (s->n < 2) return -1;
    const uint8_t t = s->p[0];
    if ((t & 0x1f) == 0x1f) return -1; /* multi-byte tags: none in PKCS#12 */
    size_t len = s->p[1], head = 2;
    if (len == 0x80) return -2;
    if (len & 0x80) {
        const size_t k = len & 0x7f;
        if (k == 0 || k > 4 || s->n < 2 + k) return -1;
        len = 0;
        for (size_t i = 0; i < k; ++i) len = len << 8 | s->p[2 + i];
        head = 2 + k;
    }
    if (len > s->n - head) return -1;
    *tag = t;
    value->p = s->p + head;
    value->n = len;
    s->p += head + len;
    s->n -= head + len;
    return 0;
}

static int der_expect(Span* s, int want, Span* value) {
    int tag = 0;
    const int r = der_next(s, &tag, value);
    if (r) return r;
    return tag == want ? 0 : -1;
}

static int der_uint(Span v, unsigned long* out) {
    if (v.n == 0 || v.n > 5 || (v.p[0] & 0x80)) return -1;
    unsigned long x = 0;
    for (size_t i = 0; i < v.n; ++i) x = x << 8 | v.p[i];
    if (x > 0xffffffffUL) return -1;
    *out = x;
    return 0;
}

static int oid_is(Span oid, const uint8_t* want, size_t n) { return oid.n == n && !memcmp(oid.p, want, n); }
#define OID_IS(span, ...) oid_is(span, (const uint8_t[]){__VA_ARGS__}, sizeof((const uint8_t[]){__VA_ARGS__}))

static void oid_text(Span oid, char* out, size_t out_len) {
    size_t at = 0;
    unsigned long v = 0;
    int first = 1;
    out[0] = 0;
    for (size_t i = 0; i < oid.n && at + 24 < out_len; ++i) {
        v = v << 7 | (oid.p[i] & 0x7f);
        if (oid.p[i] & 0x80) continue;
        if (first) {
            at += (size_t)snprintf(out + at, out_len - at, "%lu.%lu", v < 80 ? v / 40 : 2, v < 80 ? v % 40 : v - 80);
            first = 0;
        } else {
            at += (size_t)snprintf(out + at, out_len - at, ".%lu", v);
        }
        v = 0;
    }
}

/* ---- PKCS#12 key derivation (RFC 7292 appendix B) ----------------------------------------- */

static int p12_kdf(int h, const uint8_t* pw, size_t pwlen, const uint8_t* salt, size_t slen, int id, unsigned long iter,
                   uint8_t* out, size_t n) {
    const size_t u = hash_len(h), v = h == H_SHA512 ? 128 : 64;
    const size_t slen_v = slen ? v * ((slen + v - 1) / v) : 0, plen_v = pwlen ? v * ((pwlen + v - 1) / v) : 0;
    const size_t ilen = slen_v + plen_v;
    uint8_t* buf = malloc(v + ilen);
    if (!buf) return -1;
    uint8_t* I = buf + v;
    memset(buf, id, v);
    for (size_t i = 0; i < slen_v; ++i) I[i] = salt[i % slen];
    for (size_t i = 0; i < plen_v; ++i) I[slen_v + i] = pw[i % pwlen];
    uint8_t A[64], B[128];
    for (;;) {
        hash(h, buf, v + ilen, A);
        for (unsigned long r = 1; r < iter; ++r) hash(h, A, u, A);
        const size_t take = n < u ? n : u;
        memcpy(out, A, take);
        out += take;
        n -= take;
        if (!n) break;
        for (size_t i = 0; i < v; ++i) B[i] = A[i % u];
        for (size_t j = 0; j < ilen; j += v) { /* I_j = (I_j + B + 1) mod 2^(8v) */
            unsigned carry = 1;
            for (size_t k = v; k-- > 0;) {
                carry += (unsigned)I[j + k] + B[k];
                I[j + k] = (uint8_t)carry;
                carry >>= 8;
            }
        }
    }
    free(buf);
    return 0;
}

/* The password as PKCS#12 wants it: UTF-16 big-endian with a two-byte terminator. */
static uint8_t* bmp_password(const char* pw, size_t* len) {
    const size_t n = strlen(pw);
    uint8_t* out = malloc(n * 4 + 2);
    if (!out) return NULL;
    size_t at = 0;
    for (size_t i = 0; i < n;) {
        const uint8_t c = (uint8_t)pw[i];
        uint32_t cp;
        size_t k;
        if (c < 0x80) { cp = c; k = 1; }
        else if ((c & 0xe0) == 0xc0) { cp = c & 0x1f; k = 2; }
        else if ((c & 0xf0) == 0xe0) { cp = c & 0x0f; k = 3; }
        else { cp = c & 0x07; k = 4; }
        for (size_t j = 1; j < k && i + j < n; ++j) cp = cp << 6 | ((uint8_t)pw[i + j] & 0x3f);
        i += k;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            const uint32_t hi = 0xd800 | (cp >> 10), lo = 0xdc00 | (cp & 0x3ff);
            out[at++] = (uint8_t)(hi >> 8); out[at++] = (uint8_t)hi;
            out[at++] = (uint8_t)(lo >> 8); out[at++] = (uint8_t)lo;
        } else {
            out[at++] = (uint8_t)(cp >> 8); out[at++] = (uint8_t)cp;
        }
    }
    out[at++] = 0;
    out[at++] = 0;
    *len = at;
    return out;
}

typedef struct {
    const char* utf8; /* PBES2 uses the password's bytes as they are */
    const uint8_t* bmp; /* PKCS#12 PBE and the MAC: UTF-16BE */
    size_t bmp_len;
    char* err;
    size_t err_len;
} Password;

static const uint8_t OID_PKCS12_PBE[] = {0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x0c, 0x01};

/* Decrypts `in` with the algorithm `alg` (an AlgorithmIdentifier's contents). The plaintext is
 * malloc'd. Returns a VP_P12_* code. */
static int decrypt(Span alg, Span in, const Password* pw, uint8_t** out, size_t* out_len) {
    Span oid, params;
    if (der_expect(&alg, 0x06, &oid)) return VP_P12_MALFORMED;
    if (der_expect(&alg, 0x30, &params)) return VP_P12_MALFORMED;
    uint8_t key[32], iv[16];
    size_t key_len = 0;
    int cipher = 0;
    if (oid.n == sizeof OID_PKCS12_PBE + 1 && !memcmp(oid.p, OID_PKCS12_PBE, sizeof OID_PKCS12_PBE)) {
        /* pbeWithSHAAnd3-KeyTripleDES-CBC (3), 2-key (4), 128-bit RC2 (5), 40-bit RC2 (6) */
        const uint8_t which = oid.p[sizeof OID_PKCS12_PBE];
        Span salt, it;
        unsigned long iter = 0;
        if (der_expect(&params, 0x04, &salt) || der_expect(&params, 0x02, &it) || der_uint(it, &iter) || !iter || iter > MAX_ITERATIONS) return VP_P12_MALFORMED;
        if (which == 3) { cipher = C_3DES; key_len = 24; }
        else if (which == 4) { cipher = C_3DES; key_len = 16; }
        else if (which == 5) { cipher = C_RC2; key_len = 16; }
        else if (which == 6) { cipher = C_RC2; key_len = 5; }
        else { say(pw->err, pw->err_len, "PKCS#12 PBE 1.2.840.113549.1.12.1.%u", which); return VP_P12_UNSUPPORTED; }
        if (p12_kdf(H_SHA1, pw->bmp, pw->bmp_len, salt.p, salt.n, 1, iter, key, key_len) ||
            p12_kdf(H_SHA1, pw->bmp, pw->bmp_len, salt.p, salt.n, 2, iter, iv, 8)) return VP_P12_NO_MEMORY;
        if (key_len == 16 && cipher == C_3DES) { memcpy(key + 16, key, 8); key_len = 24; } /* K1 K2 K1 */
    } else if (OID_IS(oid, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x05, 0x0d)) { /* PBES2 */
        Span kdf, kdf_oid, kdf_params, enc, enc_oid, enc_iv, salt, it;
        unsigned long iter = 0, want_len = 0;
        if (der_expect(&params, 0x30, &kdf) || der_expect(&params, 0x30, &enc)) return VP_P12_MALFORMED;
        if (der_expect(&kdf, 0x06, &kdf_oid) || der_expect(&kdf, 0x30, &kdf_params)) return VP_P12_MALFORMED;
        if (!OID_IS(kdf_oid, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x05, 0x0c)) {
            char t[64]; oid_text(kdf_oid, t, sizeof t);
            say(pw->err, pw->err_len, "key derivation %s", t);
            return VP_P12_UNSUPPORTED;
        }
        if (der_expect(&kdf_params, 0x04, &salt) || der_expect(&kdf_params, 0x02, &it) || der_uint(it, &iter) || !iter || iter > MAX_ITERATIONS) return VP_P12_MALFORMED;
        int prf = H_SHA1;
        while (kdf_params.n) {
            int tag = 0;
            Span v;
            if (der_next(&kdf_params, &tag, &v)) return VP_P12_MALFORMED;
            if (tag == 0x02) { if (der_uint(v, &want_len)) return VP_P12_MALFORMED; }
            else if (tag == 0x30) {
                Span prf_oid;
                if (der_expect(&v, 0x06, &prf_oid)) return VP_P12_MALFORMED;
                if (OID_IS(prf_oid, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x02, 0x07)) prf = H_SHA1;
                else if (OID_IS(prf_oid, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x02, 0x09)) prf = H_SHA256;
                else if (OID_IS(prf_oid, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x02, 0x0b)) prf = H_SHA512;
                else { char t[64]; oid_text(prf_oid, t, sizeof t); say(pw->err, pw->err_len, "PBKDF2 PRF %s", t); return VP_P12_UNSUPPORTED; }
            }
        }
        if (der_expect(&enc, 0x06, &enc_oid) || der_expect(&enc, 0x04, &enc_iv)) return VP_P12_MALFORMED;
        if (enc_oid.n == 9 && !memcmp(enc_oid.p, "\x60\x86\x48\x01\x65\x03\x04\x01", 8) &&
            (enc_oid.p[8] == 0x02 || enc_oid.p[8] == 0x16 || enc_oid.p[8] == 0x2a)) {
            cipher = C_AES;
            key_len = enc_oid.p[8] == 0x02 ? 16 : enc_oid.p[8] == 0x16 ? 24 : 32;
            if (enc_iv.n != 16) return VP_P12_MALFORMED;
        } else if (OID_IS(enc_oid, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x03, 0x07)) {
            cipher = C_3DES;
            key_len = 24;
            if (enc_iv.n != 8) return VP_P12_MALFORMED;
        } else {
            char t[64]; oid_text(enc_oid, t, sizeof t);
            say(pw->err, pw->err_len, "cipher %s", t);
            return VP_P12_UNSUPPORTED;
        }
        if (want_len && want_len != key_len) return VP_P12_MALFORMED;
        memcpy(iv, enc_iv.p, enc_iv.n);
        if (pbkdf2(prf, (const uint8_t*)pw->utf8, strlen(pw->utf8), salt.p, salt.n, (unsigned)iter, key, key_len)) return VP_P12_NO_MEMORY;
    } else {
        char t[64]; oid_text(oid, t, sizeof t);
        say(pw->err, pw->err_len, "encryption %s", t);
        return VP_P12_UNSUPPORTED;
    }
    if (in.n == 0 || in.n % (cipher == C_AES ? 16 : 8)) return VP_P12_MALFORMED;
    uint8_t* plain = malloc(in.n);
    if (!plain) return VP_P12_NO_MEMORY;
    size_t plain_len = 0;
    if (cbc_decrypt(cipher, key, key_len, iv, in.p, in.n, plain, &plain_len)) {
        free(plain);
        say(pw->err, pw->err_len, "the password does not decrypt the contents");
        return VP_P12_BAD_PASSWORD;
    }
    *out = plain;
    *out_len = plain_len;
    return VP_P12_OK;
}

/* ---- bags -------------------------------------------------------------------------------- */

typedef struct {
    VpPkcs12* out;
    const Password* pw;
    uint8_t key_id[64];
    size_t key_id_len;
    uint8_t cert_ids[VP_P12_MAX_CERTS][64];
    size_t cert_id_lens[VP_P12_MAX_CERTS];
} Reader;

static void local_key_id(Span attrs, uint8_t* id, size_t* id_len) {
    *id_len = 0;
    while (attrs.n) {
        Span attr, type, values, value;
        if (der_expect(&attrs, 0x30, &attr) || der_expect(&attr, 0x06, &type) || der_expect(&attr, 0x31, &values)) return;
        if (OID_IS(type, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x09, 0x15) && !der_expect(&values, 0x04, &value) && value.n <= 64) {
            memcpy(id, value.p, value.n);
            *id_len = value.n;
            return;
        }
    }
}

static uint8_t* dup(const uint8_t* p, size_t n) {
    uint8_t* d = malloc(n ? n : 1);
    if (d) memcpy(d, p, n);
    return d;
}

/* PrivateKeyInfo → the key in SecKeyCreateWithData's form. */
static int take_key(Reader* r, Span info) {
    Span pki, version, alg, alg_oid, key;
    if (der_expect(&info, 0x30, &pki) || der_expect(&pki, 0x02, &version) || der_expect(&pki, 0x30, &alg) ||
        der_expect(&alg, 0x06, &alg_oid) || der_expect(&pki, 0x04, &key)) return VP_P12_MALFORMED;
    if (r->out->key) return VP_P12_OK; /* the first key */
    if (OID_IS(alg_oid, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01)) {
        r->out->key = dup(key.p, key.n);
        if (!r->out->key) return VP_P12_NO_MEMORY;
        r->out->key_len = key.n;
        r->out->key_type = VP_P12_KEY_RSA;
        return VP_P12_OK;
    }
    if (OID_IS(alg_oid, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01)) {
        /* ECPrivateKey { version, privateKey OCTET STRING, [0] parameters, [1] publicKey } */
        Span ec, v, d, pub = {0};
        if (der_expect(&key, 0x30, &ec) || der_expect(&ec, 0x02, &v) || der_expect(&ec, 0x04, &d)) return VP_P12_MALFORMED;
        while (ec.n) {
            int tag = 0;
            Span x;
            if (der_next(&ec, &tag, &x)) return VP_P12_MALFORMED;
            if (tag == 0xa1 && !der_expect(&x, 0x03, &pub)) break;
        }
        if (pub.n < 2 || pub.p[0] != 0 || pub.p[1] != 0x04) {
            say(r->pw->err, r->pw->err_len, "an EC key without its public point");
            return VP_P12_UNSUPPORTED;
        }
        const size_t coord = (pub.n - 2) / 2;
        if (d.n > coord) return VP_P12_MALFORMED;
        uint8_t* k = calloc(1, pub.n - 1 + coord);
        if (!k) return VP_P12_NO_MEMORY;
        memcpy(k, pub.p + 1, pub.n - 1);
        memcpy(k + pub.n - 1 + (coord - d.n), d.p, d.n);
        r->out->key = k;
        r->out->key_len = pub.n - 1 + coord;
        r->out->key_type = VP_P12_KEY_EC;
        return VP_P12_OK;
    }
    char t[64]; oid_text(alg_oid, t, sizeof t);
    say(r->pw->err, r->pw->err_len, "key type %s", t);
    return VP_P12_UNSUPPORTED;
}

static int read_safe_contents(Reader* r, Span contents) {
    Span bags;
    if (der_expect(&contents, 0x30, &bags)) return VP_P12_MALFORMED;
    while (bags.n) {
        Span bag, type, wrapped, value;
        if (der_expect(&bags, 0x30, &bag) || der_expect(&bag, 0x06, &type) || der_expect(&bag, 0xa0, &wrapped)) return VP_P12_MALFORMED;
        Span attrs = {0};
        if (bag.n && der_expect(&bag, 0x31, &attrs)) return VP_P12_MALFORMED;
        uint8_t id[64];
        size_t id_len = 0;
        local_key_id(attrs, id, &id_len);
        int rc = VP_P12_OK;
        if (OID_IS(type, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x0c, 0x0a, 0x01, 0x01)) { /* keyBag */
            const int had = r->out->key != NULL;
            rc = take_key(r, wrapped);
            if (!rc && !had && r->out->key) { memcpy(r->key_id, id, id_len); r->key_id_len = id_len; }
        } else if (OID_IS(type, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x0c, 0x0a, 0x01, 0x02)) { /* shrouded key */
            Span epki, alg, data;
            if (der_expect(&wrapped, 0x30, &epki) || der_expect(&epki, 0x30, &alg) || der_expect(&epki, 0x04, &data)) return VP_P12_MALFORMED;
            uint8_t* plain = NULL;
            size_t plain_len = 0;
            rc = decrypt(alg, data, r->pw, &plain, &plain_len);
            if (!rc) {
                const int had = r->out->key != NULL;
                rc = take_key(r, (Span){plain, plain_len});
                if (rc == VP_P12_MALFORMED) { /* garbage that happened to pad well: a wrong key */
                    say(r->pw->err, r->pw->err_len, "the password does not decrypt the key");
                    rc = VP_P12_BAD_PASSWORD;
                }
                if (!rc && !had && r->out->key) { memcpy(r->key_id, id, id_len); r->key_id_len = id_len; }
                memset(plain, 0, plain_len);
                free(plain);
            }
        } else if (OID_IS(type, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x0c, 0x0a, 0x01, 0x03)) { /* certBag */
            Span cert_bag, cert_type, cert_wrapped, cert;
            if (der_expect(&wrapped, 0x30, &cert_bag) || der_expect(&cert_bag, 0x06, &cert_type) ||
                der_expect(&cert_bag, 0xa0, &cert_wrapped)) return VP_P12_MALFORMED;
            if (OID_IS(cert_type, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x09, 0x16, 0x01) &&
                !der_expect(&cert_wrapped, 0x04, &cert) && r->out->cert_count < VP_P12_MAX_CERTS) {
                const int i = r->out->cert_count;
                r->out->certs[i] = dup(cert.p, cert.n);
                if (!r->out->certs[i]) return VP_P12_NO_MEMORY;
                r->out->cert_lens[i] = cert.n;
                memcpy(r->cert_ids[i], id, id_len);
                r->cert_id_lens[i] = id_len;
                r->out->cert_count = i + 1;
            }
        }
        /* Other bags (CRLs, secrets, nested SafeContents) are not needed. */
        (void)value;
        if (rc) return rc;
    }
    return VP_P12_OK;
}

/* ---- the file ---------------------------------------------------------------------------- */

static int verify_mac(Span mac_data, Span auth_content, const Password* pw) {
    Span digest_info, alg, alg_oid, digest, salt, it;
    unsigned long iter = 1;
    if (der_expect(&mac_data, 0x30, &digest_info) || der_expect(&digest_info, 0x30, &alg) ||
        der_expect(&alg, 0x06, &alg_oid) || der_expect(&digest_info, 0x04, &digest) || der_expect(&mac_data, 0x04, &salt)) return VP_P12_MALFORMED;
    if (mac_data.n && (der_expect(&mac_data, 0x02, &it) || der_uint(it, &iter) || !iter || iter > MAX_ITERATIONS)) return VP_P12_MALFORMED;
    int h;
    if (OID_IS(alg_oid, 0x2b, 0x0e, 0x03, 0x02, 0x1a)) h = H_SHA1;
    else if (OID_IS(alg_oid, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01)) h = H_SHA256;
    else if (OID_IS(alg_oid, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03)) h = H_SHA512;
    else { char t[64]; oid_text(alg_oid, t, sizeof t); say(pw->err, pw->err_len, "integrity check %s", t); return VP_P12_UNSUPPORTED; }
    const size_t u = hash_len(h);
    if (digest.n != u) return VP_P12_MALFORMED;
    uint8_t key[64], mac[64];
    if (p12_kdf(h, pw->bmp, pw->bmp_len, salt.p, salt.n, 3, iter, key, u)) return VP_P12_NO_MEMORY;
    hmac(h, key, u, auth_content.p, auth_content.n, mac);
    if (memcmp(mac, digest.p, u)) {
        say(pw->err, pw->err_len, "wrong password (the file's integrity check does not match)");
        return VP_P12_BAD_PASSWORD;
    }
    return VP_P12_OK;
}

static int read_with(const uint8_t* der, size_t len, const Password* pw, VpPkcs12* out) {
    Span file = {der, len}, pfx, version, auth, auth_oid, auth_wrapped, auth_content;
    int r = der_expect(&file, 0x30, &pfx);
    if (r == -2) { say(pw->err, pw->err_len, "BER indefinite lengths"); return VP_P12_UNSUPPORTED; }
    if (r || der_expect(&pfx, 0x02, &version) || der_expect(&pfx, 0x30, &auth) || der_expect(&auth, 0x06, &auth_oid) ||
        der_expect(&auth, 0xa0, &auth_wrapped) || der_expect(&auth_wrapped, 0x04, &auth_content)) {
        say(pw->err, pw->err_len, "not a PKCS#12 file (or BER encoded)");
        return VP_P12_MALFORMED;
    }
    if (!OID_IS(auth_oid, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x07, 0x01)) {
        say(pw->err, pw->err_len, "signed (not password) integrity: not supported");
        return VP_P12_UNSUPPORTED;
    }
    if (pfx.n) {
        Span mac_data;
        if (der_expect(&pfx, 0x30, &mac_data)) return VP_P12_MALFORMED;
        if ((r = verify_mac(mac_data, auth_content, pw))) return r;
    }
    Reader reader = {.out = out, .pw = pw};
    Span infos;
    Span content = auth_content;
    if (der_expect(&content, 0x30, &infos)) return VP_P12_MALFORMED;
    while (infos.n) {
        Span info, type, wrapped;
        if (der_expect(&infos, 0x30, &info) || der_expect(&info, 0x06, &type) || der_expect(&info, 0xa0, &wrapped)) return VP_P12_MALFORMED;
        if (OID_IS(type, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x07, 0x01)) { /* data */
            Span safe;
            if (der_expect(&wrapped, 0x04, &safe)) return VP_P12_MALFORMED;
            if ((r = read_safe_contents(&reader, safe))) return r;
        } else if (OID_IS(type, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x07, 0x06)) { /* encryptedData */
            Span ed, ver, eci, ctype, alg;
            if (der_expect(&wrapped, 0x30, &ed) || der_expect(&ed, 0x02, &ver) || der_expect(&ed, 0x30, &eci) ||
                der_expect(&eci, 0x06, &ctype) || der_expect(&eci, 0x30, &alg)) return VP_P12_MALFORMED;
            int tag = 0;
            Span enc;
            if (der_next(&eci, &tag, &enc)) return VP_P12_MALFORMED;
            uint8_t* joined = NULL;
            if (tag == 0xa0) { /* constructed: OCTET STRING pieces */
                joined = malloc(enc.n ? enc.n : 1);
                if (!joined) return VP_P12_NO_MEMORY;
                size_t at = 0;
                while (enc.n) {
                    Span piece;
                    if (der_expect(&enc, 0x04, &piece)) { free(joined); return VP_P12_MALFORMED; }
                    memcpy(joined + at, piece.p, piece.n);
                    at += piece.n;
                }
                enc = (Span){joined, at};
            } else if (tag != 0x80) {
                return VP_P12_MALFORMED;
            }
            uint8_t* plain = NULL;
            size_t plain_len = 0;
            r = decrypt(alg, enc, pw, &plain, &plain_len);
            free(joined);
            if (r) return r;
            r = read_safe_contents(&reader, (Span){plain, plain_len});
            if (r == VP_P12_MALFORMED && !pfx.n) { /* no MAC to have caught a wrong password */
                say(pw->err, pw->err_len, "the password does not decrypt the certificates");
                r = VP_P12_BAD_PASSWORD;
            }
            free(plain);
            if (r) return r;
        }
        /* Other content types (enveloped: public-key protected) are not used for these. */
    }
    if (!out->key || !out->cert_count) {
        say(pw->err, pw->err_len, "%s", !out->key ? "no private key in the file" : "no certificate in the file");
        return VP_P12_NO_KEY;
    }
    /* The key's certificate first. */
    int leaf = 0;
    if (reader.key_id_len) {
        for (int i = 0; i < out->cert_count; ++i) {
            if (reader.cert_id_lens[i] == reader.key_id_len && !memcmp(reader.cert_ids[i], reader.key_id, reader.key_id_len)) { leaf = i; break; }
        }
    }
    if (leaf) {
        uint8_t* c = out->certs[0]; size_t n = out->cert_lens[0];
        out->certs[0] = out->certs[leaf]; out->cert_lens[0] = out->cert_lens[leaf];
        out->certs[leaf] = c; out->cert_lens[leaf] = n;
    }
    return VP_P12_OK;
}

const uint8_t* vp_pkcs12_cert(const VpPkcs12* p, int i, size_t* len) {
    if (!p || i < 0 || i >= p->cert_count) { if (len) *len = 0; return NULL; }
    if (len) *len = p->cert_lens[i];
    return p->certs[i];
}

void vp_pkcs12_free(VpPkcs12* p) {
    if (!p) return;
    if (p->key) { memset(p->key, 0, p->key_len); free(p->key); }
    for (int i = 0; i < p->cert_count; ++i) free(p->certs[i]);
    memset(p, 0, sizeof *p);
}

int vp_pkcs12_read(const uint8_t* der, size_t len, const char* password, VpPkcs12* out, char* err, size_t err_len) {
    if (!out) return VP_P12_MALFORMED;
    memset(out, 0, sizeof *out);
    if (err && err_len) err[0] = 0;
    if (!der || !len) { say(err, err_len, "empty file"); return VP_P12_MALFORMED; }
    if (!password) password = "";
    size_t bmp_len = 0;
    uint8_t* bmp = bmp_password(password, &bmp_len);
    if (!bmp) return VP_P12_NO_MEMORY;
    Password pw = {password, bmp, bmp_len, err, err_len};
    int r = read_with(der, len, &pw, out);
    if (r == VP_P12_BAD_PASSWORD && !password[0]) {
        /* An empty password is written by some tools as no bytes at all, not as a terminator. */
        vp_pkcs12_free(out);
        Password none = {password, bmp, 0, err, err_len};
        r = read_with(der, len, &none, out);
    }
    free(bmp);
    if (r) vp_pkcs12_free(out);
    return r;
}
