# Metal backend: Objective-C++ against Metal.framework. The product runs only on Apple Vision Pro,
# so this builds only for visionOS (device or simulator SDK); other hosts build the neutral parts.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "visionOS")
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
  # The binary is always built (proves it compiles for the headset). It only runs where CTest can
  # launch it: the simulator build sets CMAKE_CROSSCOMPILING_EMULATOR to ci/simctl-run.sh.
  if(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR)
    foreach(t clear triangle textured depth mrt compute merged deferred stencil msaa blendconst timings)
      add_test(NAME gpu_metal.${t} COMMAND vpgpu_metal_tests ${t})
      # Exit 77 = this GPU lacks the feature (the simulator is not an Apple GPU): reported as skipped.
      set_tests_properties(gpu_metal.${t} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
  endif()
endif()
