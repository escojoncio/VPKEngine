# Copy of gpu/metal owned by VPVulkan (VisionQuestPro) until VPKEngine is finished and the two are merged.
# gpu/metal stays the PS4 renderer's; changes here never touch it.
# Metal backend: Objective-C++ against Metal.framework. The product runs only on Apple Vision Pro,
# so this builds only for visionOS (device or simulator SDK); other hosts build the neutral parts.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "visionOS")
  return()
endif()

enable_language(OBJCXX)

add_library(vpgpu_metal2 STATIC src/metal_backend.mm)
target_include_directories(vpgpu_metal2 PUBLIC include)
target_link_libraries(vpgpu_metal2 PUBLIC vpgpu_core "-framework Metal" "-framework Foundation" "-framework CoreGraphics" "-framework ImageIO")
target_compile_options(vpgpu_metal2 PRIVATE -fobjc-arc -Wall -Wextra)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  add_executable(vpgpu_metal2_tests tests/metal_tests.cpp)
  target_link_libraries(vpgpu_metal2_tests PRIVATE vpgpu_metal2)
  target_compile_options(vpgpu_metal2_tests PRIVATE -Wall -Wextra)
  # The binary is always built (proves it compiles for the headset). It only runs where CTest can
  # launch it: the simulator build sets CMAKE_CROSSCOMPILING_EMULATOR to ci/simctl-run.sh.
  if(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR)
    foreach(t clear triangle textured depth mrt compute merged deferred stencil msaa blendconst timings native)
      add_test(NAME gpu_metal2.${t} COMMAND vpgpu_metal2_tests ${t})
      # Exit 77 = this GPU lacks the feature (the simulator is not an Apple GPU): reported as skipped.
      set_tests_properties(gpu_metal2.${t} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
  endif()
endif()
