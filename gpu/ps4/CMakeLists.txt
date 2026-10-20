# vpps4: the PS4 renderer of VPKEngine (Liverpool draw state -> gpu/ps4/engine/core work).
add_library(vpps4 STATIC src/formats.cpp src/renderer.cpp src/tiling.cpp src/fetch.cpp src/translation_cache.cpp)

# Fingerprint of the translation chain for the persistent shader cache: the GCN lifter, the IR, the
# MSL emitter and this renderer's own descriptor handling. A change in any of them makes the saved
# translations stale (the file is replaced at the next start).
file(GLOB_RECURSE _vpps4_fp_sources CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/engine/gcn/src/* ${CMAKE_CURRENT_SOURCE_DIR}/engine/gcn/include/*
     ${CMAKE_CURRENT_SOURCE_DIR}/engine/ir/src/* ${CMAKE_CURRENT_SOURCE_DIR}/engine/ir/include/*
     ${CMAKE_CURRENT_SOURCE_DIR}/engine/msl/src/* ${CMAKE_CURRENT_SOURCE_DIR}/engine/msl/include/*
     ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/formats.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/fetch.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/translation_cache.cpp)
list(SORT _vpps4_fp_sources)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_vpps4_fp_sources})
set(_vpps4_fp_text "")
foreach(f ${_vpps4_fp_sources})
  file(READ ${f} _vpps4_fp_c)
  file(RELATIVE_PATH _vpps4_fp_rel ${CMAKE_CURRENT_SOURCE_DIR} ${f})
  string(APPEND _vpps4_fp_text "${_vpps4_fp_rel}\n${_vpps4_fp_c}\n")
endforeach()
string(SHA256 _vpps4_fp_hash "${_vpps4_fp_text}")
string(SUBSTRING ${_vpps4_fp_hash} 0 16 _vpps4_fp_hash)
set_source_files_properties(src/translation_cache.cpp PROPERTIES COMPILE_DEFINITIONS VPPS4_TRANSLATOR_HASH="${_vpps4_fp_hash}")
target_include_directories(vpps4 PUBLIC include)
target_link_libraries(vpps4 PUBLIC vp4gpu_core vpgcn_lift vp4msl)
target_compile_features(vpps4 PUBLIC cxx_std_20)
if(NOT MSVC)
  target_compile_options(vpps4 PRIVATE -Wall -Wextra -Werror)
endif()

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  add_executable(vpps4_tests tests/renderer_tests.cpp)
  target_link_libraries(vpps4_tests PRIVATE vpps4)
  target_compile_definitions(vpps4_tests PRIVATE VPPS4_LIFT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/engine/gcn/tests/lift")
  if(NOT MSVC)
    target_compile_options(vpps4_tests PRIVATE -Wall -Wextra -Werror)
  endif()
  add_executable(vpps4_tiling_tests tests/tiling_tests.cpp)
  target_link_libraries(vpps4_tiling_tests PRIVATE vpps4)
  if(NOT MSVC)
    target_compile_options(vpps4_tiling_tests PRIVATE -Wall -Wextra -Werror)
  endif()
  add_test(NAME vpps4.tiling COMMAND vpps4_tiling_tests)
  foreach(t clear_and_draw textured fan shader_error linear_upload tiled_upload fetch_subroutine indirect gds_append dma content_check msaa_limit resolve runaway storage_alias busy_ranges translation_cache draw_bench draw_bench_heavy)
    add_test(NAME vpps4.${t} COMMAND vpps4_tests ${t})
  endforeach()
endif()
