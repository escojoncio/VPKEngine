# VPVulkan: VPKEngine's own Vulkan driver on gpu/core (+ gpu/metal2 on visionOS). Clients that load
# MoltenVK by path (Klepton's kl_vulkan.c) load libVPVulkan.dylib instead. Khronos headers only
# (Apache-2.0/MIT, third_party/vulkan-headers); no other Vulkan code.
set(VPVK_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan-headers/include)
set(VPVK_COMMON src/entry.cpp src/physical_device.cpp src/commands.cpp src/config.cpp src/pipelines.cpp src/recipes.cpp src/precompile.cpp src/ext.cpp src/taa_native.cpp src/foveation.cpp)
# The static gpu/ libraries end up inside a shared library.
set_target_properties(vpgpu_core vpir vpmsl vpspirv PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
  enable_language(OBJCXX)
  # VPKTAA nativo: el MSL de gpu/visionos/shaders embebido como cadena (compilado en el visor al
  # primer uso, como los shaders traducidos).
  set(VPKN_MSL ${CMAKE_CURRENT_SOURCE_DIR}/../visionos/shaders/vpktaa_native.metal)
  set(VPKN_HDR ${CMAKE_CURRENT_BINARY_DIR}/gen/vpktaa_native_msl.h)
  add_custom_command(OUTPUT ${VPKN_HDR}
    COMMAND ${CMAKE_COMMAND} -DIN=${VPKN_MSL} -DOUT=${VPKN_HDR} -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_msl.cmake
    DEPENDS ${VPKN_MSL} ${CMAKE_CURRENT_SOURCE_DIR}/embed_msl.cmake)
  add_library(VPVulkan SHARED ${VPVK_COMMON} src/platform_metal.mm src/taa_native_metal.mm ${VPKN_HDR})
  target_include_directories(VPVulkan PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
  set_source_files_properties(src/taa_native_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  set_target_properties(vpgpu_metal2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(VPVulkan PRIVATE vpgpu_metal2 vpspirv vpmsl)
  set_source_files_properties(src/platform_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  add_library(VPVulkan SHARED ${VPVK_COMMON} src/platform_null.cpp)
  target_link_libraries(VPVulkan PRIVATE vpgpu_core vpspirv vpmsl)
endif()
target_include_directories(VPVulkan PUBLIC include ${VPVK_HEADERS} PRIVATE src)
# VPKEngine's own GLES/EGL (gpu/gles) ships inside the driver: Klepton finds vpgles_* in the dylib it
# already loads. Whole archive: nothing in the driver references it.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gles/CMakeLists.txt)
  target_link_libraries(VPVulkan PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vpgles>")
endif()
set_target_properties(VPVulkan PROPERTIES CXX_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  # Vulkan structs are written `T x{sType}` and filled field by field: the rest is value-initialised.
  target_compile_options(VPVulkan PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    # The driver sources are compiled into the test too, so internals (translate) are testable;
    # the `exports` group dlopens the real libVPVulkan.so the way kl_vulkan.c dlopens a driver.
    add_executable(vpvk_tests tests/vk_tests.cpp ${VPVK_COMMON} src/platform_null.cpp)
    target_link_libraries(vpvk_tests PRIVATE vpgpu_core vpspirv vpmsl ${CMAKE_DL_LIBS})
    target_include_directories(vpvk_tests PRIVATE include src ${VPVK_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/../metal2/include)
    target_compile_definitions(vpvk_tests PRIVATE VPVK_SPIRV_TESTS="${CMAKE_CURRENT_SOURCE_DIR}/../shader/spirv/tests")
    add_dependencies(vpvk_tests VPVulkan)
    if(TARGET vpgles_tests)
      add_dependencies(vpgles_tests VPVulkan)
    endif()
    target_compile_options(vpvk_tests PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    foreach(t exports physical memory passes submit pipeline recipes taa msaa queries timeline renderpass_ops linear images image_lifetime vertex_stores profile profile13 report_geom pipeline_policy eds inline_uniform bda ycbcr taa_native taa_eye_depth taa_uimask aot aniso zeroinit foveation archive_index stage_keys async_pipelines pass_drop skip_reasons)
      add_test(NAME gpu_vulkan.${t} COMMAND vpvk_tests ${t} $<TARGET_FILE:VPVulkan>)
    endforeach()
    # msaaTileOn() reads VPVK_MSAA_TILE once per process: its own run with it off.
    add_test(NAME gpu_vulkan.msaa_tile_off COMMAND vpvk_tests msaa_tile_off $<TARGET_FILE:VPVulkan>)
    set_tests_properties(gpu_vulkan.msaa_tile_off PROPERTIES ENVIRONMENT "VPVK_MSAA_TILE=0")
  endif()
endif()
