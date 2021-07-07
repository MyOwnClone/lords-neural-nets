add_executable (xor xor_double.c)
set_property(TARGET xor PROPERTY C_STANDARD 11)
target_link_libraries(xor m lib)

add_executable (mnist mnist_double.c)
set_property(TARGET mnist PROPERTY C_STANDARD 11)
target_link_libraries(mnist m lib)

add_executable (xor_float xor_float.c)
set_property(TARGET xor_float PROPERTY C_STANDARD 11)
target_link_libraries(xor_float m lib)