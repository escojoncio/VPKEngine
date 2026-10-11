add_executable(vpir_test vpir_test.cpp)
target_link_libraries(vpir_test PRIVATE vpir)
target_compile_options(vpir_test PRIVATE -Wall -Wextra)

add_test(NAME vpir.builder COMMAND vpir_test builder)

# Canonical files (the MSL golden inputs): parse, verify clean, print back byte for byte.
file(GLOB good CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../msl/tests/*.vpir)
foreach(f ${good})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vpir.roundtrip.${n} COMMAND vpir_test roundtrip ${f})
endforeach()

# Negative files: the first line is "; expect: <substring of the parse or verifier error>".
file(GLOB bad CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bad/*.vpir)
foreach(f ${bad})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vpir.reject.${n} COMMAND vpir_test reject ${f})
endforeach()
