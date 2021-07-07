add_executable (xor_double xor_double.c)
set_property(TARGET xor_double PROPERTY C_STANDARD 11)
target_link_libraries(xor_double m lib)

add_executable (mnist_double mnist_double.c)
set_property(TARGET mnist_double PROPERTY C_STANDARD 11)
target_link_libraries(mnist_double m lib)

add_executable (xor_float xor_float.c)
set_property(TARGET xor_float PROPERTY C_STANDARD 11)
target_link_libraries(xor_float m lib)