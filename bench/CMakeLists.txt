add_executable (matmul_bench matmul_bench.c bench_utils.c)

set_property(TARGET matmul_bench PROPERTY C_STANDARD 11)
target_link_libraries(matmul_bench m lib)

add_executable(mnist_bench mnist_bench.c bench_utils.c)
set_property(TARGET mnist_bench PROPERTY C_STANDARD 11)
target_link_libraries(mnist_bench m lib)

add_executable(xor_bench xor_bench.c bench_utils.c)
set_property(TARGET xor_bench PROPERTY C_STANDARD 11)
target_link_libraries(xor_bench m lib)

add_executable(muladd_fp32_64_bench muladd_fp32_64_bench.c bench_utils.c)
set_property(TARGET muladd_fp32_64_bench PROPERTY C_STANDARD 11)
target_link_libraries(muladd_fp32_64_bench m lib)