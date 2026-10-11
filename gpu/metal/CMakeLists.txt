# Metal backend: Objective-C++ against Metal.framework, Apple platforms only.
if(NOT APPLE)
  return()
endif()

enable_language(OBJCXX)

add_library(vpgpu_metal STATIC src/metal_backend.mm)
target_include_directories(vpgpu_metal PUBLIC include)
target_link_libraries(vpgpu_metal PUBLIC vpgpu_core "-framework Metal" "-framework Foundation")
target_compile_options(vpgpu_metal PRIVATE -fobjc-arc -Wall -Wextra)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  add_executable(vpgpu_metal_tests tests/metal_tests.cpp)
  target_link_libraries(vpgpu_metal_tests PRIVATE vpgpu_metal)
  target_compile_options(vpgpu_metal_tests PRIVATE -Wall -Wextra)
  foreach(t clear triangle textured depth mrt compute merged deferred stencil msaa blendconst timings)
    add_test(NAME gpu_metal.${t} COMMAND vpgpu_metal_tests ${t})
  endforeach()
endif()
