add_library(vpgpu_core STATIC
  src/resources.cpp
  src/commands.cpp
  src/frame_compiler.cpp
  src/pipeline_cache.cpp
  src/diagnostics.cpp
  src/null_backend.cpp
  src/device.cpp
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
  # One CTest entry per test case so failures are reported by name.
  foreach(t resources commands merge copies hazards subresources cuts diagnostics pipelines device tile bindings astro fuzz copyrects bigframe rectenv mipchain deadstore rtmemory scopedpool background viewsplit tilehot activity harvest automsaa passfuse required_dispatch gpuchain passdump)
    add_test(NAME gpu_core.${t} COMMAND vpgpu_core_tests ${t})
  endforeach()
  # Commit order and completion frontier of the Metal backends (header-only, tested here without Metal).
  add_executable(vpgpu_submit_order_tests tests/submit_order_tests.cpp)
  target_link_libraries(vpgpu_submit_order_tests PRIVATE vpgpu_core)
  if(NOT MSVC)
    target_compile_options(vpgpu_submit_order_tests PRIVATE -Wall -Wextra)
  endif()
  foreach(t wouldpark deferred resumeflush flushrules closing noparkunderlock parkunderlock frontier gaugewait oncemap)
    add_test(NAME gpu_core.submit_order.${t} COMMAND vpgpu_submit_order_tests ${t})
  endforeach()
  # Regressions of the misc audit (pipeline cache re-queue, key decode range, pass dump mip shift, range saturation).
  add_executable(vpgpu_core_audit_tests tests/audit_misc_tests.cpp)
  target_link_libraries(vpgpu_core_audit_tests PRIVATE vpgpu_core)
  if(NOT MSVC)
    target_compile_options(vpgpu_core_audit_tests PRIVATE -Wall -Wextra)
  endif()
  foreach(t requeue decoderange dumpdeepmip rangesat extentmip save background)
    add_test(NAME gpu_core.audit.${t} COMMAND vpgpu_core_audit_tests ${t})
    # Under UBSan a finding must stop the test (the default is to print and go on).
    set_tests_properties(gpu_core.audit.${t} PROPERTIES ENVIRONMENT "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
  endforeach()
endif()
