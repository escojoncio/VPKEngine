#!/bin/sh
# runtime/vp_codesign.c: signs an arm64 Mach-O library (as a game pack is) with a certificate
# chain made here (root -> intermediate -> leaf with a team in its OU, like Apple's development
# certificates), then checks it independently: page hashes, CodeDirectory, SuperBlob layout
# (verify.py) and the CMS signature with OpenSSL. Needs clang, ld64.lld (LD64=...), OpenSSL.
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; B=$R/build/codesign; mkdir -p "$B"
CC=${CC:-cc}; CLANG=${CLANG:-clang}; LD64=${LD64:-ld64.lld}
cd "$B"
# A chain like Apple's: root CA, intermediate CA, leaf with OU = team identifier.
cat > ext.cnf <<'EOF'
[ca]
basicConstraints=critical,CA:TRUE
keyUsage=critical,keyCertSign,cRLSign
[leaf]
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=codeSigning
EOF
openssl req -x509 -newkey rsa:2048 -nodes -keyout root.key -out root.pem -days 30 -subj "/CN=Test Root CA/O=VPEngine test" -addext "basicConstraints=critical,CA:TRUE" 2>/dev/null
openssl req -newkey rsa:2048 -nodes -keyout int.key -out int.csr -subj "/CN=Test Intermediate/OU=G3/O=VPEngine test" 2>/dev/null
openssl x509 -req -in int.csr -CA root.pem -CAkey root.key -CAcreateserial -out int.pem -days 30 -extfile ext.cnf -extensions ca 2>/dev/null
openssl req -newkey rsa:2048 -nodes -keyout leaf.key -out leaf.csr -subj "/UID=ABCDE12345/CN=Apple Development: test (XYZ)/OU=TEAM123456/O=test/C=US" 2>/dev/null
openssl x509 -req -in leaf.csr -CA int.pem -CAkey int.key -CAcreateserial -out leaf.pem -days 30 -extfile ext.cnf -extensions leaf 2>/dev/null
# A library big enough for many pages and a partial last one.
cat > lib.c <<'EOF'
int table[50000] = {1};
int f(int x) { return table[x % 50000] + x; }
EOF
$CLANG -target arm64-apple-macos11 -O1 -c -o lib.o lib.c
$LD64 -arch arm64 -platform_version macos 11.0 11.0 -dylib -adhoc_codesign -install_name @rpath/lib.dylib -o lib.dylib lib.o "$R/tools/sdk/libSystem.tbd"
$CC -O2 -I "$R/runtime" -o sign "$D/sign.c" "$R/runtime/vp_codesign.c" -lcrypto
cp lib.dylib signed.dylib
./sign signed.dylib com.vpengine.test leaf.pem leaf.key int.pem root.pem
python3 "$D/verify.py" signed.dylib "$B" TEAM123456 com.vpengine.test
openssl cms -verify -inform DER -in cms.der -binary -content cd.bin -CAfile root.pem -purpose any -out /dev/null
# Signing again (a pack re-signed with a new certificate) gives a valid signature too.
./sign signed.dylib com.vpengine.test leaf.pem leaf.key int.pem root.pem
python3 "$D/verify.py" signed.dylib "$B" TEAM123456 com.vpengine.test
openssl cms -verify -inform DER -in cms.der -binary -content cd.bin -CAfile root.pem -purpose any -out /dev/null
# Looking like the app's signature (VPGamePack.swift's variants): a designated requirement of
# the app's kind carried, then read back from that file into another signed with 16 KiB pages.
python3 - req.bin <<'PY'
import struct, sys
def s(b): return struct.pack(">I", len(b)) + b + bytes(-len(b) % 4)
ident = b"com.example.app"
expr = (struct.pack(">I", 6) + struct.pack(">I", 6) + struct.pack(">I", 2) + s(ident) + struct.pack(">I", 15)
        + struct.pack(">I", 11) + struct.pack(">i", 0) + s(b"subject.OU") + struct.pack(">I", 1) + s(b"TEAM123456"))
r = struct.pack(">III", 0xFADE0C00, 12 + len(expr), 1) + expr
open(sys.argv[1], "wb").write(struct.pack(">III", 0xFADE0C01, 20 + len(r), 1) + struct.pack(">II", 3, 20) + r)
PY
cp lib.dylib req4k.dylib
VP_REQ_FILE=req.bin ./sign req4k.dylib com.example.app leaf.pem leaf.key int.pem root.pem
VP_REQ_BLOB=req.bin python3 "$D/verify.py" req4k.dylib "$B" TEAM123456 com.example.app
openssl cms -verify -inform DER -in cms.der -binary -content cd.bin -CAfile root.pem -purpose any -out /dev/null
cp lib.dylib req16k.dylib
VP_REQ_FROM=req4k.dylib VP_PAGE_SHIFT=14 ./sign req16k.dylib com.example.app leaf.pem leaf.key int.pem root.pem
VP_REQ_BLOB=req.bin VP_PAGE_SHIFT=14 python3 "$D/verify.py" req16k.dylib "$B" TEAM123456 com.example.app
openssl cms -verify -inform DER -in cms.der -binary -content cd.bin -CAfile root.pem -purpose any -out /dev/null
echo "app-like signatures (requirement set carried, 16 KiB pages): OK"
# A tampered page must not verify.
cp signed.dylib bad.dylib
printf 'X' | dd of=bad.dylib bs=1 seek=5000 conv=notrunc 2>/dev/null
if python3 "$D/verify.py" bad.dylib "$B" TEAM123456 com.vpengine.test >/dev/null 2>&1; then echo "tampered file passed"; exit 1; fi
echo "tampered page: rejected OK"
# An implementation of Apple's formats written by others: parses the signature and checks the CMS.
if [ -n "$RCODESIGN" ]; then
    "$RCODESIGN" print-signature-info signed.dylib > rcodesign.txt 2>&1
    grep -q "signature_verifies: true" rcodesign.txt && grep -q "team_name: TEAM123456" rcodesign.txt || { cat rcodesign.txt; exit 1; }
    "$RCODESIGN" verify signed.dylib 2>&1 | grep -q "no problems detected" || { "$RCODESIGN" verify signed.dylib; exit 1; }
    for f in req4k.dylib req16k.dylib; do
        "$RCODESIGN" verify "$f" 2>&1 | grep -q "no problems detected" || { "$RCODESIGN" verify "$f"; exit 1; }
    done
    echo "rcodesign: signature verifies OK"
fi
