# Frozen copy for the PS4 renderer (AstroVisionPro); evolves independently of gpu/metal2.
# Metal backend: Objective-C++ against Metal.framework. The product runs only on Apple Vision Pro,
# so this builds only for visionOS (device or simulator SDK); other hosts build the neutral parts.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "visionOS")
  return()
endif()

enable_language(OBJCXX)

add_library(vp4gpu_metal STATIC src/metal_backend.mm)
target_include_directories(vp4gpu_metal PUBLIC include)
target_link_libraries(vp4gpu_metal PUBLIC vp4gpu_core "-framework Metal" "-framework Foundation" "-framework CoreGraphics" "-framework ImageIO")
target_compile_options(vp4gpu_metal PRIVATE -fobjc-arc -Wall -Wextra)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  add_executable(vp4gpu_metal_tests tests/metal_tests.cpp)
  target_link_libraries(vp4gpu_metal_tests PRIVATE vp4gpu_metal)
  target_compile_options(vp4gpu_metal_tests PRIVATE -Wall -Wextra)
  # The binary is always built (proves it compiles for the headset). It only runs where CTest can
  # launch it: the simulator build sets CMAKE_CROSSCOMPILING_EMULATOR to ci/simctl-run.sh.
  if(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR)
    foreach(t clear triangle textured depth mrt compute merged deferred stencil msaa blendconst timings native interop)
      add_test(NAME vp4gpu_metal.${t} COMMAND vp4gpu_metal_tests ${t})
      # Exit 77 = this GPU lacks the feature (the simulator is not an Apple GPU): reported as skipped.
      set_tests_properties(vp4gpu_metal.${t} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
  endif()
endif()
