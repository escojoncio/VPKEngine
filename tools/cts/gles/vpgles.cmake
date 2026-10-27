# dEQP target "vpgles": the GLES modules (deqp-gles2/3/31) on the EGL + GLES shim of tools/cts/gles (libEGL.dylib, libGLESv2.dylib
# re-exporting libVPGLES.dylib, which forwards to vpgles inside VPVulkan). prepare-cts.sh copies this file to targets/vpgles/
# and the EGL platform (surfaceless, library names patched for macOS) to framework/platform/vpgles/.
# Variables from the command line: VPGLES_SHIM_DIR (the built dylibs), VPGLES_INCLUDE (Khronos EGL/GLES headers).
message("*** vpgles target")

set(DEQP_TARGET_NAME "VPGLES")
set(DEQP_SUPPORT_GLES2 ON)
set(DEQP_SUPPORT_GLES3 ON)
set(DEQP_SUPPORT_EGL ON)

if (NOT VPGLES_SHIM_DIR OR NOT VPGLES_INCLUDE)
	message(FATAL_ERROR "vpgles target: pass -DVPGLES_SHIM_DIR=... and -DVPGLES_INCLUDE=...")
endif ()

set(DEQP_GLES2_LIBRARIES ${VPGLES_SHIM_DIR}/libGLESv2.dylib)
set(DEQP_GLES3_LIBRARIES ${VPGLES_SHIM_DIR}/libGLESv2.dylib)
set(DEQP_EGL_LIBRARIES ${VPGLES_SHIM_DIR}/libEGL.dylib)
include_directories(${VPGLES_INCLUDE})

# Target file may define TCUTIL_PLATFORM_SRCS (framework/platform/CMakeLists.txt)
set(TCUTIL_PLATFORM_SRCS
	vpgles/tcuSurfacelessPlatform.hpp
	vpgles/tcuSurfacelessPlatform.cpp
	)
