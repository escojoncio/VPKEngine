add_executable(vp4msl_golden golden.cpp)
target_link_libraries(vp4msl_golden PRIVATE vp4msl)
target_compile_options(vp4msl_golden PRIVATE -Wall -Wextra)

# Each NAME.vpir must translate to exactly NAME.metal. Set VPMSL_UPDATE=1 to rewrite the goldens.
file(GLOB cases CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.vpir)
foreach(f ${cases})
  get_filename_component(n ${f} NAME_WE)
  if(n STREQUAL "rate_map_fs")  # emitted with Module::rateMapPosition (foveation)
    add_test(NAME vp4msl.golden.${n} COMMAND vp4msl_golden ${f} ${CMAKE_CURRENT_SOURCE_DIR}/${n}.metal --rate-map)
  else()
    add_test(NAME vp4msl.golden.${n} COMMAND vp4msl_golden ${f} ${CMAKE_CURRENT_SOURCE_DIR}/${n}.metal)
  endif()
endforeach()


# Colour-output masks: lighting writes only colour 3 (G-buffer 0-2 are fetched, never written);
# two outputs at 0 and 1; a vertex shader has none.
add_test(NAME vp4msl.outputs.deferred_light COMMAND vp4msl_golden ${CMAKE_CURRENT_SOURCE_DIR}/deferred_light_fetched_fs.vpir
         ${CMAKE_CURRENT_SOURCE_DIR}/deferred_light_fetched_fs.metal --color-outputs 0x8)
add_test(NAME vp4msl.outputs.precise_fma COMMAND vp4msl_golden ${CMAKE_CURRENT_SOURCE_DIR}/precise_fma.vpir
         ${CMAKE_CURRENT_SOURCE_DIR}/precise_fma.metal --color-outputs 0x3)
add_test(NAME vp4msl.outputs.vertex COMMAND vp4msl_golden ${CMAKE_CURRENT_SOURCE_DIR}/passthrough_vs.vpir
         ${CMAKE_CURRENT_SOURCE_DIR}/passthrough_vs.metal --color-outputs 0)

# A wave-64 module using subgroup ops lowers to threadgroup exchanges only in a compute workgroup of at most 64 invocations.
add_test(NAME vp4msl.reject.wave64 COMMAND vp4msl_golden ${CMAKE_CURRENT_SOURCE_DIR}/reject/wave64_workgroup.vpir --reject "at most 64 invocations")
# Per-invocation memory has no atomics.
add_test(NAME vp4msl.reject.private_atomic COMMAND vp4msl_golden ${CMAKE_CURRENT_SOURCE_DIR}/reject/private_atomic.vpir --reject "atomics are not allowed on private memory")
# Lod queries need derivatives.
add_test(NAME vp4msl.reject.query_lod_vertex COMMAND vp4msl_golden ${CMAKE_CURRENT_SOURCE_DIR}/reject/query_lod_vertex.vpir --reject "lod queries are fragment-only")

# f64 ops: the soft-float library is emitted once per shader that uses it, and only then.
add_executable(vp4msl_f64_test f64_msl_test.cpp)
target_link_libraries(vp4msl_f64_test PRIVATE vp4msl)
target_compile_options(vp4msl_f64_test PRIVATE -Wall -Wextra)
add_test(NAME vp4msl.f64.library COMMAND vp4msl_f64_test ${CMAKE_CURRENT_SOURCE_DIR}/f64/f64_ops.vpir ${CMAKE_CURRENT_SOURCE_DIR}/compute_loop.vpir)
