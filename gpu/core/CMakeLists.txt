add_library(vpgpu_core STATIC
  src/resources.cpp
  src/commands.cpp
  src/frame_compiler.cpp
  src/pipeline_cache.cpp
  src/diagnostics.cpp
  src/null_backend.cpp
  src/device.cpp)
target_include_directories(vpgpu_core PUBLIC include)
target_compile_features(vpgpu_core PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(vpgpu_core PUBLIC Threads::Threads)
if(NOT MSVC)
  target_compile_options(vpgpu_core PRIVATE -Wall -Wextra)
endif()

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  add_executable(vpgpu_core_tests tests/core_tests.cpp)
  target_link_libraries(vpgpu_core_tests PRIVATE vpgpu_core)
  if(NOT MSVC)
    target_compile_options(vpgpu_core_tests PRIVATE -Wall -Wextra)
  endif()
  # One CTest entry per test case so failures are reported by name.
  foreach(t resources commands merge copies hazards subresources cuts diagnostics pipelines device tile bindings astro fuzz)
    add_test(NAME gpu_core.${t} COMMAND vpgpu_core_tests ${t})
  endforeach()
endif()
