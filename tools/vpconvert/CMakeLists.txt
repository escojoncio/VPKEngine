# vpconvert: translate + compile + link a game on the device (see vpconvert.h).
#   cmake -S tools/vpconvert -B build/vpconvert -DVPCONVERT_LLVM_ROOT=<LLVM: include/ and lib/*.a>
# Builds libvpconvert.a (vpaot, Zydis and the conversion; LLVM, clang and lld are linked by whoever
# links this) and, outside Apple's mobile systems, vpconvert-cli to try it on a computer.
cmake_minimum_required(VERSION 3.20)
project(vpconvert LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(VPCONVERT_LLVM_ROOT "" CACHE PATH "LLVM/clang/lld build: include/ and lib/*.a")
if(NOT EXISTS "${VPCONVERT_LLVM_ROOT}/include/clang/Frontend/CompilerInstance.h")
  message(FATAL_ERROR "VPCONVERT_LLVM_ROOT=${VPCONVERT_LLVM_ROOT} has no clang headers")
endif()

set(VPENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
option(ZYDIS_BUILD_TOOLS "" OFF)
option(ZYDIS_BUILD_EXAMPLES "" OFF)
option(ZYDIS_BUILD_DOXYGEN "" OFF)
option(ZYDIS_BUILD_MAN "" OFF)
add_subdirectory(${VPENGINE_ROOT}/third_party/zydis zydis EXCLUDE_FROM_ALL)

# vpaot as a library: its main() becomes vpaot_main().
add_library(vpaot_lib STATIC
  ${VPENGINE_ROOT}/tools/vpaot/main.cpp
  ${VPENGINE_ROOT}/tools/vpaot/image.cpp
  ${VPENGINE_ROOT}/tools/vpaot/translate.cpp)
set_target_properties(vpaot_lib PROPERTIES CXX_STANDARD 20)
target_compile_definitions(vpaot_lib PRIVATE main=vpaot_main)
target_link_libraries(vpaot_lib PUBLIC Zydis)

add_library(vpconvert STATIC vpconvert.cpp)
target_include_directories(vpconvert PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${VPCONVERT_LLVM_ROOT}/include)
# LLVM is built without RTTI.
target_compile_options(vpconvert PRIVATE -fno-rtti)
target_link_libraries(vpconvert PUBLIC vpaot_lib)

if(NOT APPLE)
  file(GLOB llvm_libs "${VPCONVERT_LLVM_ROOT}/lib/libLLVM*.a" "${VPCONVERT_LLVM_ROOT}/lib/libclang*.a" "${VPCONVERT_LLVM_ROOT}/lib/liblld*.a")
  list(FILTER llvm_libs EXCLUDE REGEX "libclang(-cpp|Daemon|Tidy|Doc|IncludeCleaner|Move|Query|Reorder|ChangeNamespace|ApplyReplacements|Interpreter|Repl).*")
  add_executable(vpconvert-cli cli.cpp)
  target_link_libraries(vpconvert-cli PRIVATE vpconvert -Wl,--start-group ${llvm_libs} -Wl,--end-group pthread dl)
  find_library(ZLIB_LIB z)
  find_library(ZSTD_LIB zstd)
  foreach(l ${ZLIB_LIB} ${ZSTD_LIB})
    if(l)
      target_link_libraries(vpconvert-cli PRIVATE ${l})
    endif()
  endforeach()
endif()
