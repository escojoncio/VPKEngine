#!/bin/sh
# CMAKE_CROSSCOMPILING_EMULATOR for the visionOS simulator build: CTest prefixes every test command
# with this, so each test binary runs inside the booted simulator named by VPGPU_SIM_UDID.
# A plain simulator-platform executable can be spawned directly (no app bundle, no signing beyond
# the linker's ad-hoc signature); simctl waits for it, forwards stdout/stderr and its exit status.
set -eu
: "${VPGPU_SIM_UDID:?export the UDID of a booted visionOS simulator}"
exec xcrun simctl spawn "$VPGPU_SIM_UDID" "$@"
