/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * A reader of PKCS#12 (.p12) files of its own: the certificate and private key a developer
 * certificate comes in. Apple's SecPKCS12Import does not read the encryption OpenSSL 3 and
 * tools built on it use by default (PBES2: PBKDF2 + AES-CBC, with a SHA-256 MAC), which is how
 * iloader exports the certificate it gives SideStore; this reads that and the older forms
 * (PKCS#12 PBE with SHA-1 and 3DES or RC2).
 *
 * The ciphers come from the system: CommonCrypto on Apple systems, OpenSSL's libcrypto elsewhere
 * (tests: build with -DVP_PKCS12_OPENSSL and -lcrypto).
 */
#ifndef VP_PKCS12_H
#define VP_PKCS12_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VP_P12_MAX_CERTS 8

enum {
    VP_P12_OK = 0,
    VP_P12_MALFORMED = -1,     /* not a PKCS#12 file this reader understands */
    VP_P12_BAD_PASSWORD = -2,  /* the integrity MAC does not match (or the decryption does not) */
    VP_P12_UNSUPPORTED = -3,   /* an algorithm it does not have (named in the error text) */
    VP_P12_NO_KEY = -4,        /* no private key, or no certificate */
    VP_P12_NO_MEMORY = -5,
};

enum { VP_P12_KEY_RSA = 1, VP_P12_KEY_EC = 2 };

typedef struct VpPkcs12 {
    /* The private key in the form SecKeyCreateWithData takes: RSA → the PKCS#1 RSAPrivateKey DER;
     * EC → ANSI X9.63 (04 || X || Y || D). */
    int key_type;
    uint8_t* key;
    size_t key_len;
    /* Certificates (DER): certs[0] is the one of the private key (by localKeyID, else the first). */
    int cert_count;
    uint8_t* certs[VP_P12_MAX_CERTS];
    size_t cert_lens[VP_P12_MAX_CERTS];
} VpPkcs12;

/* Reads `der` with `password` (UTF-8, may be empty). On VP_P12_OK the caller frees with
 * vp_pkcs12_free; otherwise `err` says what failed (no secrets in it). */
int vp_pkcs12_read(const uint8_t* der, size_t len, const char* password, VpPkcs12* out, char* err, size_t err_len);
void vp_pkcs12_free(VpPkcs12* p);
/* Certificate `i` (0 = the key's), or NULL past the last: C arrays reach Swift as tuples. */
const uint8_t* vp_pkcs12_cert(const VpPkcs12* p, int i, size_t* len);

#ifdef __cplusplus
}
#endif
#endif
