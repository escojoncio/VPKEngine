add_library(vpgcn src/gcn.cpp)

# GCN -> VPIR lifter.
add_library(vpgcn_lift src/lift.cpp)
target_link_libraries(vpgcn_lift PUBLIC vpgcn vpir)
target_compile_options(vpgcn_lift PRIVATE -Wall -Wextra -Werror)
target_include_directories(vpgcn PUBLIC include)
target_compile_features(vpgcn PUBLIC cxx_std_20)
target_compile_options(vpgcn PRIVATE -Wall -Wextra -Werror)

add_subdirectory(tests)

# Batch lifter over a shadPS4 shader dump (game data: never committed).
if(TARGET vpmsl)
  add_executable(vpgcn_lift_batch tools/lift_batch.cpp)
  target_link_libraries(vpgcn_lift_batch PRIVATE vpgcn_lift vpmsl)
  target_compile_options(vpgcn_lift_batch PRIVATE -Wall -Wextra -Werror)
endif()
