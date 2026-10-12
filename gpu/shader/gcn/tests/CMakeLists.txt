add_executable(vpgcn_test gcn_test.cpp)
target_link_libraries(vpgcn_test PRIVATE vpgcn)
target_compile_options(vpgcn_test PRIVATE -Wall -Wextra -Werror)

add_test(NAME gcn.corpus COMMAND vpgcn_test ${CMAKE_CURRENT_SOURCE_DIR}/corpus.txt)

# Lifter: each lift/NAME.lift (assembled from NAME.s by lift/assemble.py) must lift, verify and
# translate to exactly lift/NAME.metal, with lift/NAME.txt as its bindings/notes summary.
# VPGCN_UPDATE=1 rewrites the goldens.
if(TARGET vpmsl)
  add_executable(vpgcn_lift_test lift_test.cpp)
  target_link_libraries(vpgcn_lift_test PRIVATE vpgcn_lift vpmsl)
  target_compile_options(vpgcn_lift_test PRIVATE -Wall -Wextra -Werror)
  file(GLOB lift_cases CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lift/*.lift)
  foreach(f ${lift_cases})
    get_filename_component(n ${f} NAME_WE)
    add_test(NAME gcn.lift.${n} COMMAND vpgcn_lift_test ${f} ${CMAKE_CURRENT_SOURCE_DIR}/lift/${n}.metal
             ${CMAKE_CURRENT_SOURCE_DIR}/lift/${n}.txt)
  endforeach()
endif()

# Independent reference interpreter for SALU/VALU ops (differential testing of the translator).
# Contraction off: v_mad_f32 / v_mac_f32 are unfused on the hardware.
add_library(vpgcn_ref ref/gcn_ref.cpp)
target_link_libraries(vpgcn_ref PUBLIC vpgcn)
target_include_directories(vpgcn_ref PUBLIC ref)
target_compile_options(vpgcn_ref PRIVATE -Wall -Wextra -Werror -ffp-contract=off)

add_executable(vpgcn_ref_selftest ref/gcn_ref_test.cpp)
target_link_libraries(vpgcn_ref_selftest PRIVATE vpgcn_ref)
target_compile_options(vpgcn_ref_selftest PRIVATE -Wall -Wextra -Werror)
add_test(NAME gcn.ref.selftest COMMAND vpgcn_ref_selftest)

# Differential test of the lifter: reference wave interpreter vs the lifted VPIR, interpreted
# (diff/programs.txt from diff/gen_programs.py).
if(TARGET vpgcn_ref AND TARGET vpgcn_lift)
  add_executable(vpgcn_diff diff/gcn_diff.cpp diff/vpir_eval.cpp)
  target_link_libraries(vpgcn_diff PRIVATE vpgcn_ref vpgcn_lift)
  target_compile_options(vpgcn_diff PRIVATE -Wall -Wextra -Werror)
  add_test(NAME gcn.lift.differential COMMAND vpgcn_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs.txt)
endif()
