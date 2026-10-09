/* Test driver for runtime/vp_codesign.c: signs a Mach-O with a certificate and key from PEM files
 * (OpenSSL stands in for Security.framework, which holds the key on the headset).
 *   sign FILE IDENTIFIER leaf.pem key.pem [chain.pem...]      then prints the team it reads back */
#include "vp_codesign.h"
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned char* der_of(const char* pem, int* len) {
    FILE* f = fopen(pem, "r");
    if (!f) return NULL;
    X509* x = PEM_read_X509(f, NULL, NULL, NULL);
    fclose(f);
    if (!x) return NULL;
    unsigned char* out = NULL;
    *len = i2d_X509(x, &out);
    X509_free(x);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: sign FILE IDENTIFIER leaf.pem key.pem [chain.pem...]\n"); return 2; }
    int leaf_len;
    unsigned char* leaf = der_of(argv[3], &leaf_len);
    const uint8_t* chain[8];
    size_t lens[8];
    int n = 0;
    for (int i = 5; i < argc && n < 8; ++i) {
        int l;
        chain[n] = der_of(argv[i], &l);
        lens[n++] = (size_t)l;
    }
    FILE* kf = fopen(argv[4], "r");
    EVP_PKEY* key = kf ? PEM_read_PrivateKey(kf, NULL, NULL, NULL) : NULL;
    if (kf) fclose(kf);
    if (!leaf || !key) { fprintf(stderr, "cannot read the certificate or key\n"); return 1; }
    char err[256];
    VpCodesign* s = vp_codesign_begin(argv[1], argv[2], leaf, (size_t)leaf_len, chain, lens, n, err, sizeof err);
    if (!s) { fprintf(stderr, "begin: %s\n", err); return 1; }
    size_t tl;
    const uint8_t* tbs = vp_codesign_to_sign(s, &tl);
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    size_t sl = 0;
    unsigned char sig[1024];
    const int is_ec = EVP_PKEY_base_id(key) == EVP_PKEY_EC;
    if (EVP_DigestSignInit(ctx, NULL, EVP_sha256(), NULL, key) != 1 || EVP_DigestSign(ctx, NULL, &sl, tbs, tl) != 1 ||
        sl > sizeof sig || EVP_DigestSign(ctx, sig, &sl, tbs, tl) != 1) {
        fprintf(stderr, "signing failed\n");
        return 1;
    }
    EVP_MD_CTX_free(ctx);
    if (vp_codesign_finish(s, sig, sl, is_ec, err, sizeof err)) { fprintf(stderr, "finish: %s\n", err); return 1; }
    char team[64];
    if (vp_codesign_file_team(argv[1], team, sizeof team)) { fprintf(stderr, "no team read back\n"); return 1; }
    printf("signed %s, team %s\n", argv[1], team);
    return 0;
}
