set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast")

set(CMAKE_C_FLAGS "")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast")

add_executable (xor xor.c)
set_property(TARGET xor PROPERTY C_STANDARD 11)
target_link_libraries(xor m lib)

add_executable (mnist mnist.c)
set_property(TARGET mnist PROPERTY C_STANDARD 11)
target_link_libraries(mnist m lib)

add_executable (xor_float xor_float.c)
set_property(TARGET xor_float PROPERTY C_STANDARD 11)
target_link_libraries(xor_float m lib)