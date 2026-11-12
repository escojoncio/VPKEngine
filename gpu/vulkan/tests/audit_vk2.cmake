# Tests of the second audit of gpu/vulkan (notas/auditoria-vk2.md). A light executable: the few driver sources it checks
# (foveation, pass dump, VPCAP codec) are compiled in, libVPVulkan and vpreplay are driven as a host does.
add_executable(vpvk_audit2 tests/audit_vk2_tests.cpp src/foveation.cpp src/passdump.cpp src/vpcap.cpp)
target_include_directories(vpvk_audit2 PRIVATE include src tools ${VPVK_HEADERS})
target_link_libraries(vpvk_audit2 PRIVATE vpgpu_core vpspirv vpmsl ${VPVK_ZLIB} ${CMAKE_DL_LIBS} pthread)
target_compile_options(vpvk_audit2 PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
add_dependencies(vpvk_audit2 VPVulkan vpreplay)
foreach(t fov_nan config_nan png_hostile json passdump_race pool_shutdown capture_endtext capture_filenames capture_resolve)
  add_test(NAME gpu_vulkan.audit2_${t} COMMAND vpvk_audit2 ${t} $<TARGET_FILE:VPVulkan>)
endforeach()
add_test(NAME gpu_vulkan.audit2_replay_imagedata COMMAND vpvk_audit2 replay_imagedata $<TARGET_FILE:VPVulkan> $<TARGET_FILE:vpreplay>)
