#!/bin/sh
# runtime/vp_pkcs12.c: no secret left in freed memory, and the providers loaded once under threads.
# Needs openssl (CLI, headers, libcrypto) and gcc/TSan.   pkcs12_secrets.sh [OUT_DIR]
set -u
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..
OUT=${1:-${TMPDIR:-/tmp}/vp_pkcs12_secrets}; mkdir -p "$OUT"
PW='Zq7-pässwörd-N4'
command -v openssl >/dev/null || { echo "SKIP pkcs12_secrets (no openssl)"; exit 0; }
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$OUT/k.pem" -out "$OUT/c.pem" -subj /CN=vp -days 2 2>/dev/null || exit 1
openssl pkcs12 -export -inkey "$OUT/k.pem" -in "$OUT/c.pem" -out "$OUT/modern.p12" -passout "pass:$PW" || exit 1
openssl pkcs12 -export -legacy -certpbe PBE-SHA1-3DES -keypbe PBE-SHA1-3DES -macalg sha1 -inkey "$OUT/k.pem" -in "$OUT/c.pem" \
    -out "$OUT/legacy.p12" -passout "pass:$PW" || exit 1
CF="-std=gnu11 -g -DVP_PKCS12_OPENSSL -I$R/runtime"
rc=0
cc $CF -O2 -o "$OUT/wipe" "$D/pkcs12_secrets.c" "$R/runtime/vp_pkcs12.c" -Wl,--wrap=free -Wl,--wrap=OSSL_PROVIDER_load -lcrypto || exit 1
for f in modern legacy; do "$OUT/wipe" wipe "$OUT/$f.p12" "$PW" || rc=1; done
cc $CF -O1 -fsanitize=thread -o "$OUT/threads" "$D/pkcs12_secrets.c" "$R/runtime/vp_pkcs12.c" -Wl,--wrap=free -Wl,--wrap=OSSL_PROVIDER_load -lcrypto -lpthread || exit 1
for i in 1 2 3 4 5; do "$OUT/threads" threads "$OUT/modern.p12" "$PW" || { rc=1; break; }; done # a race depends on the schedule
exit $rc
