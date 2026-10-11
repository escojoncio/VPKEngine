add_library(vpgcn src/gcn.cpp)
target_include_directories(vpgcn PUBLIC include)
target_compile_features(vpgcn PUBLIC cxx_std_20)
target_compile_options(vpgcn PRIVATE -Wall -Wextra -Werror)

add_subdirectory(tests)
