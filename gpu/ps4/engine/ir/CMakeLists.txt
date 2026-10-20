# Frozen copy for the PS4 renderer (AstroVisionPro); evolves independently of gpu/shader/ir.
add_library(vp4ir
    src/ir.cpp
    src/builder.cpp
    src/parse.cpp
    src/print.cpp
    src/verify.cpp
    src/optimize.cpp
    src/fbfetch.cpp)
target_include_directories(vp4ir PUBLIC include)
target_compile_features(vp4ir PUBLIC cxx_std_20)
target_compile_options(vp4ir PRIVATE -Wall -Wextra)

add_subdirectory(tests)
