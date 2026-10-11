cmake_minimum_required(VERSION 3.20)
project(vpgpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# Components land one by one; each is built as soon as its directory exists.
foreach(dir shader/ir shader/msl shader/gcn core)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endforeach()
