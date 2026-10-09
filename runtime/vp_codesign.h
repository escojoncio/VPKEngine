/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Signs a thin arm64 Mach-O (a game pack built with `ld64.lld -adhoc_codesign`) with a
 * developer certificate, the way codesign does, so that the app that was signed with the same
 * certificate (by SideStore) can dlopen it without JIT: the kernel only maps code whose signature
 * chains to Apple and whose team matches the app's.
 *
 * Portable C, no crypto library: SHA-256 is here; the one private-key operation (an RSA PKCS#1
 * v1.5 or ECDSA signature over SHA-256) is done by the caller in between, with whatever holds the
 * key (Security.framework on the headset, OpenSSL in the tests):
 *
 *   VpCodesign* s = vp_codesign_begin(path, "com.example.pack", leaf, n, chain, lens, k, err, sizeof err);
 *   size_t len; const uint8_t* attrs = vp_codesign_to_sign(s, &len);
 *   ... signature = sign(key, attrs, len) ...
 *   vp_codesign_finish(s, signature, sig_len, is_ec, err, sizeof err);   (also frees s)
 *
 * The file at `path` is changed in place (its signature replaced, __LINKEDIT resized): work on a
 * copy.
 */
#ifndef VP_CODESIGN_H
#define VP_CODESIGN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VpCodesign VpCodesign;

/* Prepares the signature of the Mach-O at `path`: rewrites its headers for the new signature size,
 * hashes its pages, and builds the attributes to sign. `leaf` is the signing certificate (DER);
 * `chain` the certificates to embed after it (intermediate, root: DER). The team identifier is the
 * leaf's subject OU. NULL on error (message in `err`). */
VpCodesign* vp_codesign_begin(const char* path, const char* identifier, const uint8_t* leaf, size_t leaf_len,
                              const uint8_t* const* chain, const size_t* chain_lens, int chain_count,
                              char* err, size_t err_len);

/* The same with what makes a signature look like another one (the app's own, which the system
 * accepts): `requirements` the requirement set to carry (a 0xfade0c01 blob, e.g. the app's, read
 * with vp_codesign_file_requirements; NULL/0: an empty set), `page_shift` the code page size
 * hashed (12: 4 KiB, 14: 16 KiB). */
VpCodesign* vp_codesign_begin_ex(const char* path, const char* identifier, const uint8_t* requirements, size_t requirements_len,
                                 int page_shift, const uint8_t* leaf, size_t leaf_len,
                                 const uint8_t* const* chain, const size_t* chain_lens, int chain_count,
                                 char* err, size_t err_len);

/* The requirement set (0xfade0c01 blob) in the signature of the Mach-O at `path`, copied to `out`:
 * its length, or -1 (none, unreadable, or longer than `out_len`). */
long vp_codesign_file_requirements(const char* path, uint8_t* out, size_t out_len);

/* The bytes to sign (DER of the CMS signed attributes) with the leaf's private key, SHA-256. */
const uint8_t* vp_codesign_to_sign(VpCodesign* s, size_t* len);

/* The team identifier taken from the certificate (for checking against the app's own). */
const char* vp_codesign_team(VpCodesign* s);

/* Writes the signature (`is_ec`: ECDSA, else RSA). Frees `s` whatever the result. 0 on success. */
int vp_codesign_finish(VpCodesign* s, const uint8_t* signature, size_t signature_len, int is_ec, char* err, size_t err_len);

/* Frees a VpCodesign without finishing (the file is left with a placeholder signature). */
void vp_codesign_abort(VpCodesign* s);

/* The team identifier (subject OU) of a certificate (DER). 0 and the team in `out`, else -1. */
int vp_codesign_cert_team(const uint8_t* der, size_t len, char* out, size_t out_len);

/* SHA-256 (also used by the app to tell whether a pack changed). */
void vp_sha256(const void* data, size_t len, uint8_t out[32]);

/* The team identifier in the signature of the Mach-O at `path` (the app's own executable: what a
 * pack's signature must match). 0 and the team in `out` (NUL-terminated); -1 if unsigned, ad hoc
 * or unreadable. */
int vp_codesign_file_team(const char* path, char* out, size_t out_len);

/* Diagnostics: a text description of a Mach-O file's signature (SuperBlob slots, CodeDirectory
 * fields, CMS certificates and signed attributes), and one line about a certificate (CN, OU,
 * SHA-256 prefix). To compare what the system accepts (the app's own) with what a pack carries. */
int vp_codesign_describe(const char* path, char* out, size_t out_len);
void vp_codesign_cert_describe(const uint8_t* der, size_t len, char* out, size_t out_len);

#ifdef __cplusplus
}
#endif
#endif
