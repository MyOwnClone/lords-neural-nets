project(nns C)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -mfpmath=sse -ffast-math")

set(CMAKE_C_FLAGS "")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -mfpmath=sse -ffast-math")

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(bench)
add_subdirectory(test)

