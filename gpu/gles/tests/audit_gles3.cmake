# Audit tests of the GLES layer (formats.cpp, xfb.cpp) and of the GLSL ES compiler (glsl/): each expected value comes from the
# specification. Included from gpu/gles/CMakeLists.txt.
if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  get_filename_component(AUDIT_GLES_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
  # formats.cpp / xfb.cpp need no GPU: the table, the upload / readback conversions and the xfb primitive assembly.
  add_executable(vpgles_audit_formats_tests ${AUDIT_GLES_DIR}/tests/audit_formats_tests.cpp ${AUDIT_GLES_DIR}/src/formats.cpp ${AUDIT_GLES_DIR}/src/xfb.cpp)
  target_include_directories(vpgles_audit_formats_tests PRIVATE ${AUDIT_GLES_DIR}/include ${AUDIT_GLES_DIR}/src
                             ${AUDIT_GLES_DIR}/../vulkan/third_party/vulkan-headers/include)
  target_compile_options(vpgles_audit_formats_tests PRIVATE -Wall -Wextra)
  add_test(NAME gpu_gles.audit.formats COMMAND vpgles_audit_formats_tests)
  # vk.cpp resolves the Vulkan entry points through an injected loader: tested with a fake one, no libvulkan.
  add_executable(vpgles_audit_vk_tests ${AUDIT_GLES_DIR}/tests/audit_vk_tests.cpp ${AUDIT_GLES_DIR}/src/vk.cpp)
  target_include_directories(vpgles_audit_vk_tests PRIVATE ${AUDIT_GLES_DIR}/src ${AUDIT_GLES_DIR}/../vulkan/third_party/vulkan-headers/include)
  target_compile_options(vpgles_audit_vk_tests PRIVATE -Wall -Wextra)
  add_test(NAME gpu_gles.audit.vk COMMAND vpgles_audit_vk_tests)
  if(TARGET vpspirv AND TARGET vpmsl)
    add_executable(vpglsl_audit_tests ${AUDIT_GLES_DIR}/tests/audit_glsl_tests.cpp ${AUDIT_GLES_DIR}/../shader/gcn/tests/diff/vpir_eval.cpp)
    target_include_directories(vpglsl_audit_tests PRIVATE ${AUDIT_GLES_DIR}/../shader/gcn/tests/diff)
    target_link_libraries(vpglsl_audit_tests PRIVATE vpglsl vpspirv vpmsl)
    target_compile_options(vpglsl_audit_tests PRIVATE -Wall -Wextra)
    foreach(t sem link pp iface fuzz)
      add_test(NAME gpu_gles.audit.glsl_${t} COMMAND vpglsl_audit_tests ${t})
    endforeach()
  endif()
endif()
