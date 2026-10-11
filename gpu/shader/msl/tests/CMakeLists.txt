add_executable(vpmsl_golden golden.cpp)
target_link_libraries(vpmsl_golden PRIVATE vpmsl)
target_compile_options(vpmsl_golden PRIVATE -Wall -Wextra)

# Each NAME.vpir must translate to exactly NAME.metal. Set VPMSL_UPDATE=1 to rewrite the goldens.
file(GLOB cases CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.vpir)
foreach(f ${cases})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vpmsl.golden.${n} COMMAND vpmsl_golden ${f} ${CMAKE_CURRENT_SOURCE_DIR}/${n}.metal)
endforeach()


# A wave-64 module using subgroup ops has no 32-wide MSL form: the frontend must lower it first.
add_test(NAME vpmsl.reject.wave64 COMMAND vpmsl_golden ${CMAKE_CURRENT_SOURCE_DIR}/reject/wave64_ballot.vpir --reject "wave_size 64")
