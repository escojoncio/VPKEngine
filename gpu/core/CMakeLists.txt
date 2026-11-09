add_library(vpgpu_core STATIC
  src/resources.cpp
  src/commands.cpp
  src/frame_compiler.cpp
  src/pipeline_cache.cpp
  src/diagnostics.cpp
  src/null_backend.cpp
  src/label_queue.cpp
  src/device.cpp
  src/probe.cpp
  src/background.cpp)
target_include_directories(vpgpu_core PUBLIC include)
target_compile_features(vpgpu_core PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(vpgpu_core PUBLIC Threads::Threads)
if(APPLE)
  # scoped_pool.h calls objc_autoreleasePoolPush/Pop; every consumer (VPVulkan, tests; macOS host and visionOS) links libobjc.
  target_link_libraries(vpgpu_core PUBLIC objc)
endif()
if(NOT MSVC)
  target_compile_options(vpgpu_core PRIVATE -Wall -Wextra)
endif()

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  add_executable(vpgpu_core_tests tests/core_tests.cpp)
  target_link_libraries(vpgpu_core_tests PRIVATE vpgpu_core)
  if(NOT MSVC)
    target_compile_options(vpgpu_core_tests PRIVATE -Wall -Wextra)
  endif()
  add_executable(vpgpu_probe_tests tests/probe_tests.cpp)
  target_link_libraries(vpgpu_probe_tests PRIVATE vpgpu_core)
  if(NOT MSVC)
    target_compile_options(vpgpu_probe_tests PRIVATE -Wall -Wextra)
  endif()
  foreach(t hist lines names csv summary mutex bigalloc log)
    add_test(NAME gpu_probe.${t} COMMAND vpgpu_probe_tests ${t})
  endforeach()
  # One CTest entry per test case so failures are reported by name.
  foreach(t resources commands merge copies hazards clearbarrier subresources cuts diagnostics pipelines device tile bindings astro fuzz copyrects bigframe rectenv mipchain deadstore rtmemory scopedpool background viewsplit tilehot submit hoist hold label_queue defaults locks parallel nullrun activity harvest automsaa required_dispatch gpuchain)
    add_test(NAME gpu_core.${t} COMMAND vpgpu_core_tests ${t})
  endforeach()
endif()
