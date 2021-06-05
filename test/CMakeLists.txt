set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast")

set(CMAKE_C_FLAGS "")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast")

add_executable (tests test.c test_functions.c test_layer.c test_matrix.c test_network.c test_utils.c)
set_property(TARGET tests PROPERTY C_STANDARD 11)
target_link_libraries(tests m lib)