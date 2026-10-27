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

# Framebuffer-fetch pass: the deferred-lighting golden input must rewrite to its fetched twin (both
# are also MSL goldens). fbfetch/*.vpir are negative cases that must come out unchanged with the rejections their "; reject:" lines name.
set(msl_tests ${CMAKE_CURRENT_SOURCE_DIR}/../../msl/tests)
add_test(NAME vpir.fbfetch.deferred_light
         COMMAND vpir_test fbfetch ${msl_tests}/deferred_light_fs.vpir ${msl_tests}/deferred_light_fetched_fs.vpir)
file(GLOB fetch CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fbfetch/*.vpir)
foreach(f ${fetch})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vpir.fbfetch.${n} COMMAND vpir_test fbfetch ${f})
endforeach()

# Optimizer: NAME.vpir must optimize to NAME.out (canonical VPIR text).
file(GLOB opt CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/optimize/*.vpir)
foreach(f ${opt})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vpir.optimize.${n} COMMAND vpir_test optimize ${f} ${CMAKE_CURRENT_SOURCE_DIR}/optimize/${n}.out)
endforeach()

# Soft-float f64 against the host's IEEE double arithmetic (all rounding modes).
add_executable(vpir_f64_test f64_test.cpp)
target_link_libraries(vpir_f64_test PRIVATE vpir)
target_compile_options(vpir_f64_test PRIVATE -Wall -Wextra -frounding-math -fno-fast-math)
add_test(NAME vpir.f64.softfloat COMMAND vpir_f64_test)

# The f64 op prints and parses back byte for byte (function names in the text form).
add_test(NAME vpir.roundtrip.f64_ops COMMAND vpir_test roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/../../msl/tests/f64/f64_ops.vpir)
