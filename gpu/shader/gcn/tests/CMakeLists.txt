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
