# VPKEngine's own OpenGL ES 2/3 + EGL layer (gpu/gles) on top of VPVulkan. Own code only: the GLSL ES
# compiler (glsl/) emits SPIR-V that gpu/shader/spirv lowers like any Vulkan shader.
add_library(vpglsl STATIC glsl/pp.cpp glsl/types.cpp glsl/builtins.cpp glsl/parser.cpp glsl/emit.cpp glsl/link.cpp)
target_include_directories(vpglsl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glsl)
target_compile_options(vpglsl PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
set_target_properties(vpglsl PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  if(TARGET vpspirv AND TARGET vpmsl)
    # The single-invocation VPIR interpreter of the GCN differential tests runs compute shaders here.
    add_executable(vpglsl_tests tests/glsl_tests.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../shader/gcn/tests/diff/vpir_eval.cpp)
    target_include_directories(vpglsl_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../shader/gcn/tests/diff)
    target_link_libraries(vpglsl_tests PRIVATE vpglsl vpspirv vpmsl)
    target_compile_options(vpglsl_tests PRIVATE -Wall -Wextra)
    foreach(t pp parse errors es100 es300 multiview blocks control builtins textures external link compute ssbo images atomics gather fbfetch exts exec)
      add_test(NAME gpu_gles.glsl.${t} COMMAND vpglsl_tests ${t})
    endforeach()
  endif()
endif()

# The GL/EGL layer. Talks to the Vulkan driver only through vkGetInstanceProcAddr (VPVulkan's).
add_library(vpgles STATIC src/vk.cpp src/formats.cpp src/device.cpp src/textures.cpp src/draw.cpp src/egl.cpp src/gl_api.cpp)
target_include_directories(vpgles PUBLIC include PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/third_party/vulkan-headers/include)
target_link_libraries(vpgles PUBLIC vpglsl)
target_compile_options(vpgles PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter)
set_target_properties(vpgles vpglsl PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    # vpgles inside the real VPVulkan driver (null platform), dlopened like Klepton does: the test
    # reaches vpgles_* and vkGetInstanceProcAddr through dlsym only. formats.cpp is compiled in for
    # the table checks.
    add_executable(vpgles_tests tests/gles_tests.cpp src/formats.cpp)
    find_package(Threads REQUIRED)
    target_link_libraries(vpgles_tests PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    target_include_directories(vpgles_tests PRIVATE include src ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/third_party/vulkan-headers/include)
    foreach(t egl eglthreads formats buffers textures fbo program draw invalidate multiview errors readback external es31)
      add_test(NAME gpu_gles.gl.${t} COMMAND vpgles_tests ${t} $<TARGET_FILE:VPVulkan>)
    endforeach()
  endif()
endif()
