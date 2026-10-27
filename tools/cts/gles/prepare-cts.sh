#!/usr/bin/env bash
# Adds the "vpgles" dEQP target to a VK-GL-CTS checkout. Usage: prepare-cts.sh <cts dir>
# The platform is the CTS's own surfaceless EGL platform (framework/platform/surfaceless) with the library names of macOS:
# the libraries are @executable_path/lib{EGL,GLESv2}.dylib, the shim next to the binary.
set -euo pipefail
CTS="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
P="$CTS/framework/platform"
mkdir -p "$P/vpgles" "$CTS/targets/vpgles"
for f in tcuSurfacelessPlatform.cpp tcuSurfacelessPlatform.hpp; do
  sed -e 's|"libEGL\.so"|"@executable_path/libEGL.dylib"|g' \
      -e 's|"libGLESv2\.so"|"@executable_path/libGLESv2.dylib"|g' \
      -e 's|"libGL\.so"|"@executable_path/libGLESv2.dylib"|g' \
      "$P/surfaceless/$f" > "$P/vpgles/$f"
done
if grep -n '\.so"' "$P/vpgles/tcuSurfacelessPlatform.cpp" | grep -v "VULKAN_LIBRARY_SUFFIX"; then
  echo "prepare-cts: library names left in the platform"; exit 1
fi
grep -q "@executable_path/libEGL.dylib" "$P/vpgles/tcuSurfacelessPlatform.cpp"
cp "$HERE/vpgles.cmake" "$CTS/targets/vpgles/vpgles.cmake"
echo "vpgles target ready in $CTS"
