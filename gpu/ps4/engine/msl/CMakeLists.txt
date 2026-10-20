# Frozen copy for the PS4 renderer (AstroVisionPro); evolves independently of gpu/shader/msl.
add_library(vp4msl src/msl.cpp)
target_include_directories(vp4msl PUBLIC include)
target_link_libraries(vp4msl PUBLIC vp4ir)
target_compile_options(vp4msl PRIVATE -Wall -Wextra)

# Translator fingerprint (vp4msl::translatorFingerprint): a hash of every source that decides the
# MSL text (VPIR, this emitter). Builds with the same translator sources get the
# same value, so on-disk caches keyed by MSL can tell a new translator from a rebuild.
# The ir directory is spelled without ".." so Ninja does not see the same file under two names
# (the ir target's own globs use the canonical path: "multiple rules generate ...").
get_filename_component(_vpmsl_ir_dir ${CMAKE_CURRENT_SOURCE_DIR}/../ir ABSOLUTE)
file(GLOB_RECURSE _vpmsl_translator_sources CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
     ${_vpmsl_ir_dir}/src/*.cpp ${_vpmsl_ir_dir}/src/*.h
     ${_vpmsl_ir_dir}/src/*.inc
     ${_vpmsl_ir_dir}/include/*.h)
list(SORT _vpmsl_translator_sources)
# Content changes re-run configure, so the fingerprint never goes stale in an incremental build.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_vpmsl_translator_sources})
set(_vpmsl_translator_text "")
foreach(f ${_vpmsl_translator_sources})
  file(READ ${f} _vpmsl_c)
  file(RELATIVE_PATH _vpmsl_rel ${CMAKE_CURRENT_SOURCE_DIR}/.. ${f})
  string(APPEND _vpmsl_translator_text "${_vpmsl_rel}\n${_vpmsl_c}\n")
endforeach()
string(SHA256 _vpmsl_translator_hash "${_vpmsl_translator_text}")
string(SUBSTRING ${_vpmsl_translator_hash} 0 16 _vpmsl_translator_hash)
set_source_files_properties(src/msl.cpp PROPERTIES COMPILE_DEFINITIONS VPMSL_TRANSLATOR_HASH="${_vpmsl_translator_hash}")
message(STATUS "vp4msl translator fingerprint ${_vpmsl_translator_hash}")

add_subdirectory(tests)
