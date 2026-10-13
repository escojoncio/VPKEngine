# VPVulkan: VPEngine's own Vulkan driver on gpu/core (+ gpu/metal2 on visionOS). Clients that load
# MoltenVK by path (Klepton's kl_vulkan.c) load libVPVulkan.dylib instead. Khronos headers only
# (Apache-2.0/MIT, third_party/vulkan-headers); no other Vulkan code.
set(VPVK_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan-headers/include)
set(VPVK_COMMON src/entry.cpp src/physical_device.cpp src/commands.cpp src/config.cpp src/pipelines.cpp src/recipes.cpp src/precompile.cpp)
# The static gpu/ libraries end up inside a shared library.
set_target_properties(vpgpu_core vpir vpmsl vpspirv PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
  enable_language(OBJCXX)
  add_library(VPVulkan SHARED ${VPVK_COMMON} src/platform_metal.mm)
  set_target_properties(vpgpu_metal2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(VPVulkan PRIVATE vpgpu_metal2 vpspirv vpmsl)
  set_source_files_properties(src/platform_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  add_library(VPVulkan SHARED ${VPVK_COMMON} src/platform_null.cpp)
  target_link_libraries(VPVulkan PRIVATE vpgpu_core vpspirv vpmsl)
endif()
target_include_directories(VPVulkan PUBLIC include ${VPVK_HEADERS} PRIVATE src)
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
    target_include_directories(vpvk_tests PRIVATE include src ${VPVK_HEADERS})
    target_compile_definitions(vpvk_tests PRIVATE VPVK_SPIRV_TESTS="${CMAKE_CURRENT_SOURCE_DIR}/../shader/spirv/tests")
    add_dependencies(vpvk_tests VPVulkan)
    target_compile_options(vpvk_tests PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    foreach(t exports physical memory passes submit pipeline recipes taa)
      add_test(NAME gpu_vulkan.${t} COMMAND vpvk_tests ${t} $<TARGET_FILE:VPVulkan>)
    endforeach()
  endif()
endif()
