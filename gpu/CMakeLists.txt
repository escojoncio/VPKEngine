cmake_minimum_required(VERSION 3.24)
project(vpgpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# Components land one by one; each is built as soon as its directory exists.
foreach(dir shader/ir shader/msl shader/spirv core metal metal2 gles vulkan)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endforeach()

# The PS4 path (AstroVisionPro): frozen vp4* copies of core, metal2, ir and msl plus the GCN lifter
# live in ps4/engine and coexist with the originals above (distinct targets, namespaces, headers).
foreach(dir ir msl gcn core metal)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ps4/engine/${dir}/CMakeLists.txt)
    add_subdirectory(ps4/engine/${dir})
  endif()
endforeach()

# The PS4 renderer (gpu/ps4) builds on the vp4 copies only.
if(TARGET vpgcn_lift AND TARGET vp4msl AND TARGET vp4gpu_core)
  add_subdirectory(ps4)
endif()
