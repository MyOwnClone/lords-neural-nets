add_executable (xor xor.c)
set_property(TARGET xor PROPERTY C_STANDARD 11)
target_link_libraries(xor m lib)

add_executable (mnist mnist.c)
set_property(TARGET mnist PROPERTY C_STANDARD 11)
target_link_libraries(mnist m lib)