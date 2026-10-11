add_library(vpir
    src/ir.cpp
    src/builder.cpp
    src/parse.cpp
    src/print.cpp
    src/verify.cpp
    src/optimize.cpp
    src/fbfetch.cpp)
target_include_directories(vpir PUBLIC include)
target_compile_features(vpir PUBLIC cxx_std_20)
target_compile_options(vpir PRIVATE -Wall -Wextra)

add_subdirectory(tests)
