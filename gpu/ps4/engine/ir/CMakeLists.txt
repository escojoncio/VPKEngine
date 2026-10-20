# Frozen copy for the PS4 renderer (AstroVisionPro); evolves independently of gpu/shader/ir.
add_library(vp4ir
    src/ir.cpp
    src/builder.cpp
    src/parse.cpp
    src/print.cpp
    src/verify.cpp
    src/optimize.cpp
    src/fbfetch.cpp
    src/wave.cpp
    src/f64.cpp)
target_include_directories(vp4ir PUBLIC include)

# The MSL text of the soft-float f64 library: prelude + 2/pi table + core, embedded as bytes (a string
# literal would hit compiler limits). Content changes re-run configure.
set(_f64_parts ${CMAKE_CURRENT_SOURCE_DIR}/src/f64_msl_prelude.inc ${CMAKE_CURRENT_SOURCE_DIR}/src/f64_table.inc
               ${CMAKE_CURRENT_SOURCE_DIR}/src/f64_core.inc)
set(_f64_hex "")
foreach(f ${_f64_parts})
  file(READ ${f} _f64_part HEX)
  string(APPEND _f64_hex "${_f64_part}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${f})
endforeach()
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _f64_bytes "${_f64_hex}")
set(_f64_out ${CMAKE_CURRENT_BINARY_DIR}/f64_msl_data.inc)
set(_f64_old "")
if(EXISTS ${_f64_out})
  file(READ ${_f64_out} _f64_old)
endif()
if(NOT _f64_old STREQUAL "${_f64_bytes}\n")
  file(WRITE ${_f64_out} "${_f64_bytes}\n")
endif()
target_include_directories(vp4ir PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(vp4ir PUBLIC cxx_std_20)
target_compile_options(vp4ir PRIVATE -Wall -Wextra)

add_subdirectory(tests)
