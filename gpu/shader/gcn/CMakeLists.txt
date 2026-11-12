add_library(vpgcn src/gcn.cpp)

# GCN -> VPIR lifter.
add_library(vpgcn_lift src/lift.cpp)
target_link_libraries(vpgcn_lift PUBLIC vpgcn vpir)
target_compile_options(vpgcn_lift PRIVATE -Wall -Wextra -Werror)
target_include_directories(vpgcn PUBLIC include)
target_compile_features(vpgcn PUBLIC cxx_std_20)
target_compile_options(vpgcn PRIVATE -Wall -Wextra -Werror)

add_subdirectory(tests)

# Batch lifter over a shader dump folder (game data: never committed).
if(TARGET vpmsl)
  add_executable(vpgcn_lift_batch tools/lift_batch.cpp)
  target_link_libraries(vpgcn_lift_batch PRIVATE vpgcn_lift vpmsl)
  target_compile_options(vpgcn_lift_batch PRIVATE -Wall -Wextra -Werror)
  # It survives file names shorter than "<stage>_<hash>".
  add_test(NAME gcn.lift_batch.short_names
           COMMAND ${CMAKE_COMMAND} -DBATCH=$<TARGET_FILE:vpgcn_lift_batch> -DDIR=${CMAKE_CURRENT_BINARY_DIR}/lift_batch_names
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/lift_batch_names.cmake)
endif()
