add_library(vpmsl src/msl.cpp)
target_include_directories(vpmsl PUBLIC include)
target_link_libraries(vpmsl PUBLIC vpir)
target_compile_options(vpmsl PRIVATE -Wall -Wextra)

add_subdirectory(tests)
