# VPKEngine's own OpenGL ES 2/3 + EGL layer (gpu/gles) on top of VPVulkan. Own code only: the GLSL ES
# compiler (glsl/) emits SPIR-V that gpu/shader/spirv lowers like any Vulkan shader.
add_library(vpglsl STATIC glsl/pp.cpp glsl/types.cpp glsl/builtins.cpp glsl/parser.cpp glsl/emit.cpp glsl/link.cpp)
target_include_directories(vpglsl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glsl)
target_compile_options(vpglsl PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
set_target_properties(vpglsl PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  if(TARGET vpspirv AND TARGET vpmsl)
    add_executable(vpglsl_tests tests/glsl_tests.cpp)
    target_link_libraries(vpglsl_tests PRIVATE vpglsl vpspirv vpmsl)
    target_compile_options(vpglsl_tests PRIVATE -Wall -Wextra)
    foreach(t pp parse errors es100 es300 multiview blocks control builtins textures link)
      add_test(NAME gpu_gles.glsl.${t} COMMAND vpglsl_tests ${t})
    endforeach()
  endif()
endif()
