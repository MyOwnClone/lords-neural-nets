project(nns C)
cmake_minimum_required(VERSION 3.16)

# Store in CMAKE_UNIX_HOST_ARCH var the current build architecture (but this works only on Unix, not Windows :-( )
execute_process(COMMAND
        arch
        OUTPUT_VARIABLE
        CMAKE_UNIX_HOST_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

if(${CMAKE_UNIX_HOST_ARCH} MATCHES "arm64") # applies only for Apple Silicon Macs, TODO: update and test with ${CMAKE_HOST_SYSTEM_PROCESSOR}
    set(CMAKE_SYSTEM_NAME Darwin)
    set(CMAKE_SYSTEM_PROCESSOR arm64)
elseif(${CMAKE_SYSTEM} MATCHES "^?Windows$")    # see: https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-To-Write-Platform-Checks
    set(CMAKE_SYSTEM_NAME Windows)
    set(CMAKE_SYSTEM_PROCESSOR x86)
endif()

# to get supported archs on macOS: clang --print-supported-cpus
# pure C setup----------------------------------------------------------------------------------------------------------

set(CMAKE_C_FLAGS "")
set(CMAKE_C_FLAGS_DEBUG "-g")

if(${CMAKE_UNIX_HOST_ARCH} MATCHES "arm64")
    # -mcpu=apple-a14 is like -march on gcc, I guess
    set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -ffast-math -mcpu=apple-a14")
elseif(${CMAKE_SYSTEM} MATCHES "^?Windows$")
    # SSE2 for Intel, using -march rocketlake for my machine, because it is cometlake, but only rocketlake (predecessor) is supported for now
    set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -mfpmath=sse2 -ffast-math -march=rocketlake")
endif()
# / pure C setup -------------------------------------------------------------------------------------------------------

# C++ setup (not needed for this project)-------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if(${CMAKE_UNIX_HOST_ARCH} MATCHES "arm64")
    # -mcpu=apple-a14 is like -march on gcc, I guess
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -ffast-math -mcpu=apple-a14")
elseif(${CMAKE_SYSTEM} MATCHES "^?Windows$")
    # SSE2 for Intel, using -march rocketlake for my machine, because it is cometlake, but only rocketlake (predecessor) is supported for now
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -mfpmath=sse2 -ffast-math -march=rocketlake")
endif()
# / C++ setup ----------------------------------------------------------------------------------------------------------


add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(bench)
add_subdirectory(test)

message(STATUS "Cmake detected CPU arch: " ${CMAKE_HOST_SYSTEM_PROCESSOR})
message(STATUS "Cmake detected operating system: " ${CMAKE_SYSTEM_NAME})

