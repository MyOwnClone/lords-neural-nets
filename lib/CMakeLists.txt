set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast")

set(CMAKE_C_FLAGS "")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast")

add_library (lib functions.c layer.c matrix.c network.c utils.c)
set_property(TARGET lib PROPERTY C_STANDARD 11)

target_link_libraries(lib m)
target_include_directories (lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})