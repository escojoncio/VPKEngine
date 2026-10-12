add_executable(vpmsl_golden golden.cpp)
target_link_libraries(vpmsl_golden PRIVATE vpmsl)
target_compile_options(vpmsl_golden PRIVATE -Wall -Wextra)

# Each NAME.vpir must translate to exactly NAME.metal. Set VPMSL_UPDATE=1 to rewrite the goldens.
file(GLOB cases CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.vpir)
foreach(f ${cases})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vpmsl.golden.${n} COMMAND vpmsl_golden ${f} ${CMAKE_CURRENT_SOURCE_DIR}/${n}.metal)
endforeach()


# Colour-output masks: lighting writes only colour 3 (G-buffer 0-2 are fetched, never written);
# two outputs at 0 and 1; a vertex shader has none.
add_test(NAME vpmsl.outputs.deferred_light COMMAND vpmsl_golden ${CMAKE_CURRENT_SOURCE_DIR}/deferred_light_fetched_fs.vpir
         ${CMAKE_CURRENT_SOURCE_DIR}/deferred_light_fetched_fs.metal --color-outputs 0x8)
add_test(NAME vpmsl.outputs.precise_fma COMMAND vpmsl_golden ${CMAKE_CURRENT_SOURCE_DIR}/precise_fma.vpir
         ${CMAKE_CURRENT_SOURCE_DIR}/precise_fma.metal --color-outputs 0x3)
add_test(NAME vpmsl.outputs.vertex COMMAND vpmsl_golden ${CMAKE_CURRENT_SOURCE_DIR}/passthrough_vs.vpir
         ${CMAKE_CURRENT_SOURCE_DIR}/passthrough_vs.metal --color-outputs 0)

# A wave-64 module using subgroup ops has no 32-wide MSL form: the frontend must lower it first.
add_test(NAME vpmsl.reject.wave64 COMMAND vpmsl_golden ${CMAKE_CURRENT_SOURCE_DIR}/reject/wave64_ballot.vpir --reject "wave_size 64")
# Per-invocation memory has no atomics.
add_test(NAME vpmsl.reject.private_atomic COMMAND vpmsl_golden ${CMAKE_CURRENT_SOURCE_DIR}/reject/private_atomic.vpir --reject "atomics are not allowed on private memory")
