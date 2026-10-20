add_executable(vpgcn_test gcn_test.cpp)
target_link_libraries(vpgcn_test PRIVATE vpgcn)
target_compile_options(vpgcn_test PRIVATE -Wall -Wextra -Werror)

add_test(NAME gcn.corpus COMMAND vpgcn_test ${CMAKE_CURRENT_SOURCE_DIR}/corpus.txt)

# Lifter: each lift/NAME.lift (assembled from NAME.s by lift/assemble.py) must lift, verify and
# translate to exactly lift/NAME.metal, with lift/NAME.txt as its bindings/notes summary.
# VPGCN_UPDATE=1 rewrites the goldens.
if(TARGET vp4msl)
  add_executable(vpgcn_lift_test lift_test.cpp)
  target_link_libraries(vpgcn_lift_test PRIVATE vpgcn_lift vp4msl)
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
add_library(vpgcn_ref ref/gcn_ref.cpp ref/gcn_ref_ds.cpp)
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
  find_package(Threads REQUIRED)
  target_link_libraries(vpgcn_diff PRIVATE vpgcn_ref vpgcn_lift vp4msl Threads::Threads)
  target_compile_options(vpgcn_diff PRIVATE -Wall -Wextra -Werror)
  add_test(NAME gcn.lift.differential COMMAND vpgcn_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs.txt)
  # The same with control flow: saveexec if/else (nested), uniform loops, per-lane loops.
  add_test(NAME gcn.lift.differential_isa COMMAND vpgcn_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_isa.txt)
  add_test(NAME gcn.lift.differential_cf COMMAND vpgcn_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_cf.txt)
  # Wave mode: compute shaders of one 64-lane wave with cross-lane ops, lane masks as data and wave-wide branches,
  # lifted as is and lowered to two 32-lane SIMD groups (diff/gen_wave.py).
  add_executable(vpgcn_wave_diff diff/gcn_wave_diff.cpp diff/vpir_eval.cpp)
  target_link_libraries(vpgcn_wave_diff PRIVATE vpgcn_ref vpgcn_lift vp4ir Threads::Threads)
  target_compile_options(vpgcn_wave_diff PRIVATE -Wall -Wextra -Werror)
  add_test(NAME gcn.lift.differential_wave COMMAND vpgcn_wave_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_wave.txt)
  set_tests_properties(gcn.lift.differential_wave PROPERTIES TIMEOUT 300)
endif()

# The wave-64 lowering (vp4ir::lowerWave64) against the module's native 64-lane semantics.
if(TARGET vp4ir)
  add_executable(vpgcn_wave_lower_test diff/wave_lower_test.cpp diff/vpir_eval.cpp)
  target_link_libraries(vpgcn_wave_lower_test PRIVATE vp4ir)
  target_compile_options(vpgcn_wave_lower_test PRIVATE -Wall -Wextra -Werror)
  find_package(Threads REQUIRED)
  target_link_libraries(vpgcn_wave_lower_test PRIVATE Threads::Threads)
  file(GLOB wave_cases CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/wave/*.vpir)
  foreach(f ${wave_cases})
    get_filename_component(n ${f} NAME_WE)
    add_test(NAME gcn.wave.lower.${n} COMMAND vpgcn_wave_lower_test ${f})
  endforeach()
endif()

# Opcode audit: every opcode of the tables lifted alone (implemented / approximate / rejected).
if(TARGET vpgcn_lift)
  add_executable(vpgcn_audit audit/gcn_audit.cpp)
  target_link_libraries(vpgcn_audit PRIVATE vpgcn_lift)
  target_compile_options(vpgcn_audit PRIVATE -Wall -Wextra -Werror)
  # The rejected count may only go down (lower the bound as opcodes are added).
  add_test(NAME gcn.audit COMMAND vpgcn_audit --expect-max-rejected 264)
endif()
