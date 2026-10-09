#!/usr/bin/env bash
# LLVM, clang and lld as static libraries for visionOS (arm64), for converting games on the headset
# (tools/vpconvert): only what an in-process compiler and Mach-O linker need, AArch64 only.
#
#   build-llvm-visionos.sh OUT_DIR          (macOS with Xcode's visionOS SDK, cmake, ninja)
#
# OUT_DIR gets lib/*.a, include/ (LLVM, clang and lld headers, generated ones included) and
# resource-include/ (clang's own headers: stddef.h, stdint.h...). Takes 1-2 hours on a CI runner;
# the workflow keeps the result.
set -euo pipefail
OUT=${1:?usage: $0 OUT_DIR}
VER=${LLVM_VERSION:-21.1.0}
DEPLOY=${VISIONOS_DEPLOYMENT_TARGET:-26.0}
WORK=${LLVM_WORK:-$PWD/build/llvm-visionos}
mkdir -p "$WORK" "$OUT"
OUT=$(cd "$OUT" && pwd)
SRC=$WORK/llvm-project-$VER.src
if [[ ! -d "$SRC/llvm" ]]; then
  curl -sSL "https://github.com/llvm/llvm-project/releases/download/llvmorg-$VER/llvm-project-$VER.src.tar.xz" | tar -xJ -C "$WORK"
fi

COMMON=(
  -DCMAKE_BUILD_TYPE=Release
  -DLLVM_TARGETS_TO_BUILD=AArch64
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF
  -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_LIBPFM=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_ENABLE_PLUGINS=OFF
  -DCLANG_ENABLE_ARCMT=OFF -DCLANG_ENABLE_STATIC_ANALYZER=OFF -DCLANG_PLUGIN_SUPPORT=OFF
)

# 1. The table generators, for this Mac (they run during the build).
HOST=$WORK/host
if [[ ! -x "$HOST/bin/clang-tblgen" ]]; then
  cmake -S "$SRC/llvm" -B "$HOST" -G Ninja "${COMMON[@]}" -DLLVM_ENABLE_PROJECTS="clang;lld"
  ninja -C "$HOST" llvm-tblgen clang-tblgen
  ninja -C "$HOST" llvm-min-tblgen || true # LLVM 20+
fi

# 2. The libraries, for the headset.
TARGET=$WORK/xros
cmake -S "$SRC/llvm" -B "$TARGET" -G Ninja "${COMMON[@]}" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_OSX_SYSROOT=xros -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOY" \
  -DCMAKE_C_COMPILER="$(xcrun --find clang)" -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)" \
  -DLLVM_HOST_TRIPLE="arm64-apple-xros$DEPLOY" -DLLVM_DEFAULT_TARGET_TRIPLE="arm64-apple-xros$DEPLOY" \
  -DLLVM_NATIVE_TOOL_DIR="$HOST/bin" -DLLVM_TABLEGEN="$HOST/bin/llvm-tblgen" \
  -DCLANG_TABLEGEN="$HOST/bin/clang-tblgen" -DLLVM_MIN_TABLEGEN="$HOST/bin/llvm-min-tblgen" \
  -DLLVM_BUILD_TOOLS=OFF -DLLVM_BUILD_UTILS=OFF -DCLANG_BUILD_TOOLS=OFF -DLLD_BUILD_TOOLS=OFF \
  -DLLVM_ENABLE_UNWIND_TABLES=OFF -DLLVM_ENABLE_ASSERTIONS=OFF
ninja -C "$TARGET" clangFrontend clangCodeGen clangDriver clangSerialization lldMachO lldCommon \
  LLVMAArch64CodeGen LLVMAArch64AsmParser LLVMAArch64Desc LLVMAArch64Info LLVMAArch64Utils

# 3. What vpconvert builds and links against.
rm -rf "$OUT/lib" "$OUT/include" "$OUT/resource-include"
mkdir -p "$OUT/lib" "$OUT/include" "$OUT/resource-include"
find "$TARGET/lib" -maxdepth 1 -name '*.a' -exec cp {} "$OUT/lib/" \;
for d in "$SRC/llvm/include" "$TARGET/include" "$SRC/clang/include" "$TARGET/tools/clang/include" "$SRC/lld/include"; do
  rsync -a --include='*/' --include='*.h' --include='*.def' --include='*.inc' --include='*.td' --exclude='*' "$d/" "$OUT/include/"
done
cp -R "$SRC/clang/lib/Headers/." "$OUT/resource-include/"
echo "$VER" > "$OUT/VERSION"
du -sh "$OUT/lib" "$OUT/include" "$OUT/resource-include"
ls "$OUT/lib" | wc -l
