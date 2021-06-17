project(nns C)
cmake_minimum_required(VERSION 3.16)

if(${CMAKE_SYSTEM} MATCHES "Darwin") # applies only for Apple Silicon Macs, TODO: update and test with ${CMAKE_HOST_SYSTEM_PROCESSOR}
    set(CMAKE_SYSTEM_NAME Darwin)
elseif(${CMAKE_SYSTEM} MATCHES "^?Windows$")    # see: https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-To-Write-Platform-Checks
    set(CMAKE_SYSTEM_PROCESSOR x86)
endif()

# to get supported archs on macOS: clang --print-supported-cpus

set(CMAKE_C_FLAGS "")
set(CMAKE_C_FLAGS_DEBUG "-g")

if(${CMAKE_SYSTEM} MATCHES "Darwin")
    # -mcpu=apple-a14 is like -march on gcc, I guess
    set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -ffast-math -mcpu=apple-a14")
elseif(${CMAKE_SYSTEM} MATCHES "^?Windows$")
    # SSE2 for Intel, using -march rocketlake for my machine, because it is cometlake, but only rocketlake (predecessor) is supported for now
    set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -mfpmath=sse2 -ffast-math -march=rocketlake")
endif()

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(bench)
add_subdirectory(test)

message(STATUS "Cmake detected CPU arch: " ${CMAKE_HOST_SYSTEM_PROCESSOR})
message(STATUS "Cmake detected operating system: " ${CMAKE_SYSTEM_NAME})

