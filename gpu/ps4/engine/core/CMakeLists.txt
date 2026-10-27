# Frozen copy for the PS4 renderer (AstroVisionPro); evolves independently of gpu/core.
add_library(vp4gpu_core STATIC
  src/resources.cpp
  src/commands.cpp
  src/frame_compiler.cpp
  src/pipeline_cache.cpp
  src/diagnostics.cpp
  src/null_backend.cpp
  src/label_queue.cpp
  src/device.cpp)
target_include_directories(vp4gpu_core PUBLIC include)
target_compile_features(vp4gpu_core PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(vp4gpu_core PUBLIC Threads::Threads)
if(NOT MSVC)
  target_compile_options(vp4gpu_core PRIVATE -Wall -Wextra)
endif()

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  add_executable(vp4gpu_core_tests tests/core_tests.cpp)
  target_link_libraries(vp4gpu_core_tests PRIVATE vp4gpu_core)
  if(NOT MSVC)
    target_compile_options(vp4gpu_core_tests PRIVATE -Wall -Wextra)
  endif()
  # One CTest entry per test case so failures are reported by name.
  foreach(t resources commands merge copies hazards subresources cuts diagnostics pipelines device tile bindings astro fuzz submit hoist hold)
    add_test(NAME vp4gpu_core.${t} COMMAND vp4gpu_core_tests ${t})
  endforeach()
endif()
