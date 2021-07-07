# kind of artificial target, I just want one build which tries to build everything, so I know there are no compilation errors
add_library(just_build_all proxy.c)
set_property(TARGET just_build_all PROPERTY C_STANDARD 11)
target_link_libraries(just_build_all tests m lnn matmul_bench mnist_bench xor_bench muladd_fp32_64_bench)