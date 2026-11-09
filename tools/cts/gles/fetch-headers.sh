#!/usr/bin/env bash
# Khronos EGL and OpenGL ES headers (not in the CTS sources) into <dir>/{EGL,KHR,GLES2,GLES3}. Usage: fetch-headers.sh <dir>
set -euo pipefail
D="$(mkdir -p "$1" && cd "$1" && pwd)"
T="$(mktemp -d)"
for r in EGL-Registry OpenGL-Registry; do
  git clone -q --depth 1 --filter=blob:none --sparse "https://github.com/KhronosGroup/$r" "$T/$r"
  git -C "$T/$r" sparse-checkout set api
done
cp -R "$T/EGL-Registry/api/EGL" "$T/EGL-Registry/api/KHR" "$D/"
cp -R "$T/OpenGL-Registry/api/GLES2" "$T/OpenGL-Registry/api/GLES3" "$D/"
rm -rf "$T"
ls "$D"
