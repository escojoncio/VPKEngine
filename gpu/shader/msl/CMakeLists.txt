add_library(vpmsl src/msl.cpp)
target_include_directories(vpmsl PUBLIC include)
target_link_libraries(vpmsl PUBLIC vpir)
target_compile_options(vpmsl PRIVATE -Wall -Wextra)

# Translator fingerprint (vpmsl::translatorFingerprint): a hash of every source that decides the
# MSL text (SPIR-V frontend, VPIR, this emitter). Builds with the same translator sources get the
# same value, so on-disk caches keyed by MSL can tell a new translator from a rebuild.
file(GLOB_RECURSE _vpmsl_translator_sources CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/../ir/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../ir/src/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/../ir/include/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/../spirv/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../spirv/src/*.h)
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
message(STATUS "vpmsl translator fingerprint ${_vpmsl_translator_hash}")

add_subdirectory(tests)
