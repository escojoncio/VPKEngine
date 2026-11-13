# VPKEngine's own OpenGL ES 2/3 + EGL layer (gpu/gles) on top of VPVulkan. Own code only: the GLSL ES
# compiler (glsl/) emits SPIR-V that gpu/shader/spirv lowers like any Vulkan shader.
add_library(vpglsl STATIC glsl/pp.cpp glsl/types.cpp glsl/builtins.cpp glsl/parser.cpp glsl/emit.cpp glsl/link.cpp)
target_include_directories(vpglsl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glsl)
target_link_libraries(vpglsl PUBLIC vpir)  # vpir::StackGuard / stackLow: the compiler stops before the stack of a small worker thread does
target_compile_options(vpglsl PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
set_target_properties(vpglsl PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  if(TARGET vpspirv AND TARGET vpmsl)
    # The single-invocation VPIR interpreter of the GCN differential tests runs compute shaders here.
    add_executable(vpglsl_tests tests/glsl_tests.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../shader/gcn/tests/diff/vpir_eval.cpp)
    target_include_directories(vpglsl_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../shader/gcn/tests/diff)
    target_link_libraries(vpglsl_tests PRIVATE vpglsl vpspirv vpmsl)
    target_compile_options(vpglsl_tests PRIVATE -Wall -Wextra)
    foreach(t pp ppfuzz parse errors es100 es300 multiview blocks control builtins textures external link compute ssbo images atomics gather fbfetch exts texbuffer exec packed shlight lenient xfb invariant semantics limits nesting memory constconv)
      add_test(NAME gpu_gles.glsl.${t} COMMAND vpglsl_tests ${t})
    endforeach()
  endif()
endif()

# The GL/EGL layer. Talks to the Vulkan driver only through vkGetInstanceProcAddr (VPVulkan's).
add_library(vpgles STATIC src/vk.cpp src/formats.cpp src/device.cpp src/textures.cpp src/draw.cpp src/xfb.cpp src/egl.cpp src/gl_api.cpp)
target_include_directories(vpgles PUBLIC include PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/third_party/vulkan-headers/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/../core/include)  # vpgpu/core/perf.h only (header-only counters)
target_link_libraries(vpgles PUBLIC vpglsl)
target_compile_options(vpgles PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter)
set_target_properties(vpgles vpglsl PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    # Test hooks (vpgles_test_unlocked_hook, vpgles_test_stat) exist only in vpgles_test: the same sources with VPGLES_TESTING, linked into
    # VPVulkanTest (gpu/vulkan), the driver the tests below load. The library that ships (vpgles, in VPVulkan) is built without the define.
    add_library(vpgles_test STATIC src/vk.cpp src/formats.cpp src/device.cpp src/textures.cpp src/draw.cpp src/xfb.cpp src/egl.cpp src/gl_api.cpp)
    target_include_directories(vpgles_test PUBLIC include PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/third_party/vulkan-headers/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/../core/include)
    target_link_libraries(vpgles_test PUBLIC vpglsl)
    target_compile_options(vpgles_test PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter)
    target_compile_definitions(vpgles_test PRIVATE VPGLES_TESTING)
    set_target_properties(vpgles_test PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
    # vpgles_test inside VPVulkanTest, the real VPVulkan driver sources (null platform), dlopened like Klepton does: the test
    # reaches vpgles_* and vkGetInstanceProcAddr through dlsym only. formats.cpp is compiled in for
    # the table checks.
    add_executable(vpgles_tests tests/gles_tests.cpp src/formats.cpp src/xfb.cpp)
    find_package(Threads REQUIRED)
    target_link_libraries(vpgles_tests PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    target_include_directories(vpgles_tests PRIVATE include src ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/third_party/vulkan-headers/include)
    target_compile_options(vpgles_tests PRIVATE -Wall -Wextra -Werror -Wno-missing-field-initializers -Wno-unused-parameter -Wno-mismatched-new-delete)  # (the test replaces global new / delete by malloc / free on purpose)
    foreach(t egl eglthreads formats buffers mapunsync textures fbo msrbo msrtt_depth program draw invalidate multiview errors ubo readback external es31 progbin recipe xfb uniform_raw uniform_strict audit_lost audit_calls audit_compile audit_xfb audit2_xfer audit2_dyn audit2_buf audit2_pbo audit2_upload audit2_egl audit2_link audit2_link2 audit3_egl_last audit3_egl_refs audit3_pbo audit3_xfer_shared audit3_stale_stamp audit4_xfer_own_pass audit4_reads_b audit4_reads_ctl audit4_copygrow audit4_reinit audit5_perf audit5_race audit5_atomic_ro audit5_index_inplace audit5_pass_split audit5_after_shutdown audit6_compile_hook audit7_big_subdata audit8_mixed_subdata audit9_ring_narrow audit9_threshold audit9_staging audit9_presents audit9_swap_alloc audit8_present_throw audit6_link_hook audit6_compile_shutdown audit6_after_create audit7_present_shutdown audit7_swap_destroy audit7_reinit audit7_init_race audit5_readonly audit5_texbuf audit5_index audit5_balance audit5_shutdown0 audit5_shutdown1)
      add_test(NAME gpu_gles.gl.${t} COMMAND vpgles_tests ${t} $<TARGET_FILE:VPVulkanTest>)
    endforeach()
    # GL_RENDERER follows the Quest identity's GPU name (VPVK_QUEST_GPU_NAME, latched at the first query): its own run.
    add_test(NAME gpu_gles.gl.egl_adreno650 COMMAND vpgles_tests egl $<TARGET_FILE:VPVulkanTest>)
    set_tests_properties(gpu_gles.gl.egl_adreno650 PROPERTIES ENVIRONMENT "VPVK_QUEST_GPU_NAME=Adreno (TM) 650")
  endif()
endif()

include(tests/audit_gles3.cmake)  # audit tests of formats / xfb / the GLSL compiler (own files)
