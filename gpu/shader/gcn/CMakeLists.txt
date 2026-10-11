add_library(vpgcn src/gcn.cpp)

# GCN -> VPIR lifter.
add_library(vpgcn_lift src/lift.cpp)
target_link_libraries(vpgcn_lift PUBLIC vpgcn vpir)
target_compile_options(vpgcn_lift PRIVATE -Wall -Wextra -Werror)
target_include_directories(vpgcn PUBLIC include)
target_compile_features(vpgcn PUBLIC cxx_std_20)
target_compile_options(vpgcn PRIVATE -Wall -Wextra -Werror)

add_subdirectory(tests)
