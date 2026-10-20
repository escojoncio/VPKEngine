add_executable(vp4ir_test vp4ir_test.cpp)
target_link_libraries(vp4ir_test PRIVATE vp4ir)
target_compile_options(vp4ir_test PRIVATE -Wall -Wextra)

add_test(NAME vp4ir.builder COMMAND vp4ir_test builder)

# Canonical files (the MSL golden inputs): parse, verify clean, print back byte for byte.
file(GLOB good CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../msl/tests/*.vpir)
foreach(f ${good})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vp4ir.roundtrip.${n} COMMAND vp4ir_test roundtrip ${f})
endforeach()

# Negative files: the first line is "; expect: <substring of the parse or verifier error>".
file(GLOB bad CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bad/*.vpir)
foreach(f ${bad})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vp4ir.reject.${n} COMMAND vp4ir_test reject ${f})
endforeach()

# Framebuffer-fetch pass: the deferred-lighting golden input must rewrite to its fetched twin (both
# are also MSL goldens). fbfetch/*.vpir are negative cases that must come out unchanged with the rejections their "; reject:" lines name.
set(msl_tests ${CMAKE_CURRENT_SOURCE_DIR}/../../msl/tests)
add_test(NAME vp4ir.fbfetch.deferred_light
         COMMAND vp4ir_test fbfetch ${msl_tests}/deferred_light_fs.vpir ${msl_tests}/deferred_light_fetched_fs.vpir)
file(GLOB fetch CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fbfetch/*.vpir)
foreach(f ${fetch})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vp4ir.fbfetch.${n} COMMAND vp4ir_test fbfetch ${f})
endforeach()

# Optimizer: NAME.vpir must optimize to NAME.out (canonical VPIR text).
file(GLOB opt CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/optimize/*.vpir)
foreach(f ${opt})
  get_filename_component(n ${f} NAME_WE)
  add_test(NAME vp4ir.optimize.${n} COMMAND vp4ir_test optimize ${f} ${CMAKE_CURRENT_SOURCE_DIR}/optimize/${n}.out)
endforeach()

# Soft-float f64 against the host's IEEE double arithmetic (all rounding modes).
add_executable(vp4ir_f64_test f64_test.cpp)
target_link_libraries(vp4ir_f64_test PRIVATE vp4ir)
target_compile_options(vp4ir_f64_test PRIVATE -Wall -Wextra -frounding-math -fno-fast-math)
add_test(NAME vp4ir.f64.softfloat COMMAND vp4ir_f64_test)

# The f64 op prints and parses back byte for byte (function names in the text form).
add_test(NAME vp4ir.roundtrip.f64_ops COMMAND vp4ir_test roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/../../msl/tests/f64/f64_ops.vpir)
