add_executable (bench bench.c bench_utils.c)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set_property(TARGET bench PROPERTY C_STANDARD 11)
target_link_libraries(bench m lib)

add_executable(mnist_bench mnist_bench.c bench_utils.c)
set_property(TARGET mnist_bench PROPERTY C_STANDARD 11)
target_link_libraries(mnist_bench m lib)