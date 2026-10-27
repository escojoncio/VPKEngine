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
add_library(vpgcn_ref ref/gcn_ref.cpp ref/gcn_ref_ds.cpp ref/gcn_ref_buf.cpp ref/gcn_ref_img.cpp)
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
  target_link_libraries(vpgcn_diff PRIVATE vpgcn_ref vpgcn_lift vpmsl Threads::Threads)
  target_compile_options(vpgcn_diff PRIVATE -Wall -Wextra -Werror)
  add_test(NAME gcn.lift.differential COMMAND vpgcn_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs.txt)
  # The same with control flow: saveexec if/else (nested), uniform loops, per-lane loops.
  add_test(NAME gcn.lift.differential_isa COMMAND vpgcn_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_isa.txt)
  add_test(NAME gcn.lift.differential_cf COMMAND vpgcn_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_cf.txt)
  # The 64-bit float opcodes (soft-float f64) against the reference's host-double model.
  # (no mismatch is allowed: the reference answers GCN's NaN / signed-zero / NaN-payload rules on every host, see
  # ref/gcn_ref.cpp. A divergence that cannot be removed goes to --allow-index <program indices>, with its reason here.)
  add_test(NAME gcn.lift.differential_f64 COMMAND vpgcn_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_f64.txt)
  # Wave mode: compute shaders of one 64-lane wave with cross-lane ops, lane masks as data and wave-wide branches,
  # lifted as is and lowered to two 32-lane SIMD groups (diff/gen_wave.py).
  add_executable(vpgcn_wave_diff diff/gcn_wave_diff.cpp diff/vpir_eval.cpp)
  target_link_libraries(vpgcn_wave_diff PRIVATE vpgcn_ref vpgcn_lift vpir Threads::Threads)
  target_compile_options(vpgcn_wave_diff PRIVATE -Wall -Wextra -Werror)
  add_test(NAME gcn.lift.differential_wave COMMAND vpgcn_wave_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_wave.txt)
  set_tests_properties(gcn.lift.differential_wave PROPERTIES TIMEOUT 300)
  add_test(NAME gcn.lift.differential_ds COMMAND vpgcn_wave_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_ds.txt --lds 2048)
  add_test(NAME gcn.lift.differential_ds_src2 COMMAND vpgcn_wave_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_ds_src2.txt --lds 2048)
  set_tests_properties(gcn.lift.differential_ds gcn.lift.differential_ds_src2 PROPERTIES TIMEOUT 600)
  # MUBUF: loads / stores / atomics (32 and 64 bit) through V#s of every addressing mode (diff/gen_buf.py), against the
  # reference in ref/gcn_ref_buf.cpp ref/gcn_ref_img.cpp; the second run tells the lifter the V#'s mode up front (shorter code).
  add_test(NAME gcn.lift.differential_buf COMMAND vpgcn_wave_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_buf.txt --buf --lds 8192)
  add_test(NAME gcn.lift.differential_buf_known COMMAND vpgcn_wave_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_buf.txt --buf --known-layout --lds 8192)
  set_tests_properties(gcn.lift.differential_buf gcn.lift.differential_buf_known PROPERTIES TIMEOUT 600)
  # MIMG on storage images of every dimension: load / load_mip / store (dmask) / the 17 atomics / get_resinfo (diff/gen_img.py,
  # ref/gcn_ref_img.cpp); cubes are seen as 2D arrays of faces.
  add_test(NAME gcn.lift.differential_img COMMAND vpgcn_wave_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_img.txt --img)
  add_test(NAME gcn.lift.differential_spill COMMAND vpgcn_wave_diff ${CMAKE_CURRENT_SOURCE_DIR}/diff/programs_spill.txt --per-lane)
endif()

# The wave-64 lowering (vpir::lowerWave64) against the module's native 64-lane semantics.
if(TARGET vpir)
  add_executable(vpgcn_wave_lower_test diff/wave_lower_test.cpp diff/vpir_eval.cpp)
  target_link_libraries(vpgcn_wave_lower_test PRIVATE vpir)
  target_compile_options(vpgcn_wave_lower_test PRIVATE -Wall -Wextra -Werror)
  find_package(Threads REQUIRED)
  target_link_libraries(vpgcn_wave_lower_test PRIVATE Threads::Threads)
  file(GLOB wave_cases CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/wave/*.vpir)
  foreach(f ${wave_cases})
    get_filename_component(n ${f} NAME_WE)
    add_test(NAME gcn.wave.lower.${n} COMMAND vpgcn_wave_lower_test ${f})
  endforeach()
  # The evaluator's unpack_* ops (shared by the SPIR-V and GLSL tests) against the formulas of their definition.
  add_executable(vpgcn_unpack_eval_test diff/unpack_eval_test.cpp diff/vpir_eval.cpp)
  target_link_libraries(vpgcn_unpack_eval_test PRIVATE vpir Threads::Threads)
  target_compile_options(vpgcn_unpack_eval_test PRIVATE -Wall -Wextra -Werror)
  add_test(NAME gcn.eval.unpack COMMAND vpgcn_unpack_eval_test ${CMAKE_CURRENT_SOURCE_DIR}/../../msl/tests/unpack.vpir)
endif()

# FLAT lowering: hand-encoded FLAT programs through the GuestMap binding on the VPIR interpreter.
if(TARGET vpgcn_lift)
  add_executable(vpgcn_flat_test diff/flat_test.cpp diff/vpir_eval.cpp)
  find_package(Threads REQUIRED)
  target_link_libraries(vpgcn_flat_test PRIVATE vpgcn_lift vpmsl Threads::Threads)
  target_include_directories(vpgcn_flat_test PRIVATE diff)
  target_compile_options(vpgcn_flat_test PRIVATE -Wall -Wextra -Werror)
  add_test(NAME gcn.flat.lowering COMMAND vpgcn_flat_test)
endif()

# Opcode audit: every opcode of the tables lifted alone (implemented / approximate / rejected).
if(TARGET vpgcn_lift)
  add_executable(vpgcn_audit audit/gcn_audit.cpp)
  target_link_libraries(vpgcn_audit PRIVATE vpgcn_lift)
  target_compile_options(vpgcn_audit PRIVATE -Wall -Wextra -Werror)
  # The rejected count may only go down (lower the bound as opcodes are added).
  add_test(NAME gcn.audit COMMAND vpgcn_audit --expect-max-rejected 13)
endif()

# Format conversions: every (data format, number format) of the CPU reference against the VPIR
# emitters (run on the interpreter), in both directions.
if(TARGET vpgcn_lift)
  add_executable(vpgcn_format_test format_test.cpp diff/vpir_eval.cpp)
  target_link_libraries(vpgcn_format_test PRIVATE vpgcn_lift)
  target_compile_options(vpgcn_format_test PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
  add_test(NAME gcn.formats COMMAND vpgcn_format_test)
  add_executable(vpgcn_format_lift_test format_lift_test.cpp diff/vpir_eval.cpp)
  target_link_libraries(vpgcn_format_lift_test PRIVATE vpgcn_lift)
  target_compile_options(vpgcn_format_lift_test PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
  add_test(NAME gcn.formats.lift COMMAND vpgcn_format_lift_test)
endif()

# Geometry pipeline (ES / GS / copy shader rings, RectList, pos1-pos3 exports) on the VPIR interpreter.
if(TARGET vpgcn_lift)
  add_executable(vpgcn_geom_test geom_test.cpp diff/vpir_eval.cpp)
  target_include_directories(vpgcn_geom_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(vpgcn_geom_test PRIVATE vpgcn_lift vpir)
  target_compile_definitions(vpgcn_geom_test PRIVATE VPGCN_LIFT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lift")
  target_compile_options(vpgcn_geom_test PRIVATE -Wall -Wextra -Werror)
  foreach(t triangles stereo16 stereo32 strip lines conditional instanced rect misc astro)
    add_test(NAME gcn.geom.${t} COMMAND vpgcn_geom_test ${t})
  endforeach()
endif()
