# VPVulkan: VPKEngine's own Vulkan driver on gpu/core (+ gpu/metal2 on visionOS). Clients that load
# MoltenVK by path (Klepton's kl_vulkan.c) load libVPVulkan.dylib instead. Khronos headers only
# (Apache-2.0/MIT, third_party/vulkan-headers); no other Vulkan code.
set(VPVK_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan-headers/include)
# Build identity, logged once as "vpk.version sha=<VPK_GIT_SHA>" (entry.cpp). -DVPK_GIT_SHA=<sha> wins; else the HEAD of the
# checkout, with "+dirty" if the work tree has changes; else "unknown" (no git / no .git: source tarballs).
# Re-configures when HEAD or the current branch's ref file changes (a detached HEAD has no ref: only .git/HEAD).
if(NOT VPK_GIT_SHA)
  execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=12 HEAD
                  OUTPUT_VARIABLE VPK_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE VPK_GIT_RC)
  if(NOT VPK_GIT_RC EQUAL 0 OR NOT VPK_GIT_SHA)
    set(VPK_GIT_SHA unknown)
  else()
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} status --porcelain
                    OUTPUT_VARIABLE VPK_GIT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE VPK_GIT_SRC)
    if(VPK_GIT_SRC EQUAL 0 AND VPK_GIT_STATUS)
      set(VPK_GIT_SHA "${VPK_GIT_SHA}+dirty")
    endif()
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --path-format=absolute --git-path HEAD
                    OUTPUT_VARIABLE VPK_GIT_HEAD_FILE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE VPK_GIT_HRC)
    if(VPK_GIT_HRC EQUAL 0 AND EXISTS "${VPK_GIT_HEAD_FILE}")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${VPK_GIT_HEAD_FILE}")
    endif()
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} symbolic-ref -q HEAD
                    OUTPUT_VARIABLE VPK_GIT_BRANCH_REF OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE VPK_GIT_BRC)
    if(VPK_GIT_BRC EQUAL 0 AND VPK_GIT_BRANCH_REF)
      execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --path-format=absolute --git-path ${VPK_GIT_BRANCH_REF}
                      OUTPUT_VARIABLE VPK_GIT_REF_FILE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE VPK_GIT_RRC)
      if(VPK_GIT_RRC EQUAL 0 AND EXISTS "${VPK_GIT_REF_FILE}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${VPK_GIT_REF_FILE}")
      endif()
    endif()
  endif()
endif()
set(VPVK_COMMON src/entry.cpp src/physical_device.cpp src/commands.cpp src/config.cpp src/pipelines.cpp src/recipes.cpp src/precompile.cpp src/ext.cpp src/taa_native.cpp src/foveation.cpp src/capture.cpp src/vpcap.cpp src/texdecode.cpp src/pending.cpp)
# VPCAP files are deflated with the system zlib (zstd is not in third_party/).
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
  set(VPVK_ZLIB ZLIB::ZLIB)
else()
  set(VPVK_ZLIB z)
endif()
# The static gpu/ libraries end up inside a shared library.
set_target_properties(vpgpu_core vpir vpmsl vpspirv PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(APPLE)  # visionOS (the product) and macOS (the Vulkan CTS on CI)
  enable_language(OBJCXX)
  # VPKTAA nativo: el MSL de gpu/visionos/shaders embebido como cadena (compilado en el visor al
  # primer uso, como los shaders traducidos).
  set(VPKN_MSL ${CMAKE_CURRENT_SOURCE_DIR}/../visionos/shaders/vpktaa_native.metal)
  set(VPKN_HDR ${CMAKE_CURRENT_BINARY_DIR}/gen/vpktaa_native_msl.h)
  add_custom_command(OUTPUT ${VPKN_HDR}
    COMMAND ${CMAKE_COMMAND} -DIN=${VPKN_MSL} -DOUT=${VPKN_HDR} -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_msl.cmake
    DEPENDS ${VPKN_MSL} ${CMAKE_CURRENT_SOURCE_DIR}/embed_msl.cmake)
  add_library(VPVulkan SHARED ${VPVK_COMMON} src/platform_metal.mm src/taa_native_metal.mm ${VPKN_HDR})
  target_include_directories(VPVulkan PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
  set_source_files_properties(src/taa_native_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  set_target_properties(vpgpu_metal2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(VPVulkan PRIVATE vpgpu_metal2 vpspirv vpmsl ${VPVK_ZLIB})
  set_source_files_properties(src/platform_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  add_library(VPVulkan SHARED ${VPVK_COMMON} src/platform_null.cpp)
  target_link_libraries(VPVulkan PRIVATE vpgpu_core vpspirv vpmsl ${VPVK_ZLIB})
endif()
set_source_files_properties(src/entry.cpp PROPERTIES COMPILE_DEFINITIONS "VPK_GIT_SHA=\"${VPK_GIT_SHA}\"")
target_include_directories(VPVulkan PUBLIC include ${VPVK_HEADERS} PRIVATE src)
# VPKEngine's own GLES/EGL (gpu/gles) ships inside the driver: Klepton finds vpgles_* in the dylib it
# already loads. Whole archive: nothing in the driver references it.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gles/CMakeLists.txt)
  target_link_libraries(VPVulkan PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,vpgles>")
endif()
set_target_properties(VPVulkan PROPERTIES CXX_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  # Vulkan structs are written `T x{sType}` and filled field by field: the rest is value-initialised.
  target_compile_options(VPVulkan PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()

# vpreplay: replays a .vpcap against any Vulkan driver loaded at run time (libVPVulkan or the system loader). Always built off
# visionOS; for the visionOS simulator (the Metal leg of the replay comparison) only with -DVPVK_REPLAY_TOOLS=ON.
option(VPVK_REPLAY_TOOLS "Build vpreplay for visionOS too" OFF)
if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "visionOS" OR VPVK_REPLAY_TOOLS)
    add_executable(vpreplay tools/vpreplay.cpp src/vpcap.cpp)
    target_include_directories(vpreplay PRIVATE include ${VPVK_HEADERS})
    target_link_libraries(vpreplay PRIVATE ${VPVK_ZLIB} ${CMAKE_DL_LIBS})
    if(NOT MSVC)
      target_compile_options(vpreplay PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    endif()
  endif()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    # The driver sources are compiled into the test too, so internals (translate) are testable;
    # the `exports` group dlopens the real libVPVulkan.so the way kl_vulkan.c dlopens a driver.
    add_executable(vpvk_tests tests/vk_tests.cpp ${VPVK_COMMON} src/platform_null.cpp)
    target_link_libraries(vpvk_tests PRIVATE vpgpu_core vpspirv vpmsl ${VPVK_ZLIB} ${CMAKE_DL_LIBS})
    # vpcap_dump: lists the calls of a .vpcap file (phase 2's vpreplay reads it through the same vpcap.cpp).
    add_executable(vpcap_dump tools/vpcap_dump.cpp src/vpcap.cpp)
    target_include_directories(vpcap_dump PRIVATE include ${VPVK_HEADERS})
    target_link_libraries(vpcap_dump PRIVATE ${VPVK_ZLIB})
    # vpimgdiff: compares two dump directories of vpreplay (vpreplay itself is defined above).
    add_executable(vpimgdiff tools/vpimgdiff.cpp)
    target_include_directories(vpimgdiff PRIVATE ${VPVK_HEADERS})
    target_link_libraries(vpimgdiff PRIVATE ${VPVK_ZLIB})
    if(NOT MSVC)
      target_compile_options(vpimgdiff PRIVATE -Wall -Wextra)
    endif()
    target_include_directories(vpvk_tests PRIVATE include src tools ${VPVK_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/../metal2/include)
    target_compile_definitions(vpvk_tests PRIVATE VPVK_SPIRV_TESTS="${CMAKE_CURRENT_SOURCE_DIR}/../shader/spirv/tests")
    add_dependencies(vpvk_tests VPVulkan vpreplay)
    if(TARGET vpgles_tests)
      add_dependencies(vpgles_tests VPVulkan)
    endif()
    target_compile_options(vpvk_tests PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    foreach(t exports physical memory passes submit pipeline recipes recipe_keys taa msaa queries query_not_ready timeline queues timeline_order pending renderpass_ops linear images image_lifetime vertex_stores profile profile13 quest_extras push_descriptor report_geom pipeline_policy eds native_identity inline_uniform bda ycbcr taa_native taa_eye_depth taa_uimask msaa_force msaa_force_marked prewarm_variants aot aniso zeroinit foveation archive_index stage_keys async_pipelines sync_pipelines create_default pso_census pass_drop skip_reasons indirect_patch astc3d phase1b suspend_resume translator_modes perf capture capture_images capture_pools capture_host_frames capture_corrupt capture_hooks capture_codec capture_devices capture_unhooked capture_budget capture_locks png imgdiff)
      add_test(NAME gpu_vulkan.${t} COMMAND vpvk_tests ${t} $<TARGET_FILE:VPVulkan>)
    endforeach()
    # vpreplay of a capture made by the test: replayed in libVPVulkan itself, with VPVulkan's own capture on the replay.
    # The reference captures of the replay comparison (tools/replay/compare.sh), written to the build directory.
    add_test(NAME gpu_vulkan.refcaps COMMAND vpvk_tests refcaps $<TARGET_FILE:VPVulkan> ${CMAKE_CURRENT_BINARY_DIR}/refcaps)
    add_test(NAME gpu_vulkan.replay COMMAND vpvk_tests replay $<TARGET_FILE:VPVulkan> $<TARGET_FILE:vpreplay>)
    # With lavapipe installed (mesa-vulkan-drivers) the same triangle capture is replayed on it and the PNG is checked.
    find_file(VPVK_LVP_ICD NAMES lvp_icd.json lvp_icd.x86_64.json lvp_icd.aarch64.json
              PATHS /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/local/share/vulkan/icd.d NO_DEFAULT_PATH)
    if(VPVK_LVP_ICD)
      add_test(NAME gpu_vulkan.replay_lvp COMMAND vpvk_tests replay_lvp $<TARGET_FILE:VPVulkan> $<TARGET_FILE:vpreplay> ${VPVK_LVP_ICD})
    endif()
    # msaaTileOn() reads VPVK_MSAA_TILE once per process: its own run with it off.
    add_test(NAME gpu_vulkan.msaa_tile_off COMMAND vpvk_tests msaa_tile_off $<TARGET_FILE:VPVulkan>)
    set_tests_properties(gpu_vulkan.msaa_tile_off PROPERTIES ENVIRONMENT "VPVK_MSAA_TILE=0")
    # The same multiview pipeline with the vertex-amplification translator mode.
    # VPVK_CAPTURE_FRAME arms the capture from the environment, read when the first entry point is looked up.
    set(VPCAP_ENV_DIR ${CMAKE_CURRENT_BINARY_DIR}/vpcap_env)
    file(MAKE_DIRECTORY ${VPCAP_ENV_DIR})
    add_test(NAME gpu_vulkan.capture_env COMMAND vpvk_tests capture_env $<TARGET_FILE:VPVulkan>)
    set_tests_properties(gpu_vulkan.capture_env PROPERTIES ENVIRONMENT "VPVK_CAPTURE_FRAME=2;VPVK_CAPTURE_DIR=${VPCAP_ENV_DIR}")
    set(VPCAP_ENV_HOST_DIR ${CMAKE_CURRENT_BINARY_DIR}/vpcap_env_host)
    file(MAKE_DIRECTORY ${VPCAP_ENV_HOST_DIR})
    add_test(NAME gpu_vulkan.capture_env_host COMMAND vpvk_tests capture_env_host $<TARGET_FILE:VPVulkan>)
    set_tests_properties(gpu_vulkan.capture_env_host PROPERTIES ENVIRONMENT "VPVK_CAPTURE_FRAME=2;VPVK_CAPTURE_DIR=${VPCAP_ENV_HOST_DIR}")
    set(VPCAP_ENV_DIR_DIR ${CMAKE_CURRENT_BINARY_DIR}/vpcap_env_dir)
    file(MAKE_DIRECTORY ${VPCAP_ENV_DIR_DIR})
    add_test(NAME gpu_vulkan.capture_env_dir COMMAND vpvk_tests capture_env_dir $<TARGET_FILE:VPVulkan>)
    set_tests_properties(gpu_vulkan.capture_env_dir PROPERTIES ENVIRONMENT "VPVK_CAPTURE=1;VPVK_CAPTURE_DIR=${VPCAP_ENV_DIR_DIR}")
    add_test(NAME gpu_vulkan.pipeline_amplify COMMAND vpvk_tests pipeline $<TARGET_FILE:VPVulkan>)
    set_tests_properties(gpu_vulkan.pipeline_amplify PROPERTIES ENVIRONMENT "VPVK_AMPLIFY=1")
  endif()
endif()
