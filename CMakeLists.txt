project(nns C)
cmake_minimum_required(VERSION 3.16)

# Store in CMAKE_DEB_HOST_ARCH var the current build architecture
execute_process(COMMAND
        arch
        OUTPUT_VARIABLE
        CMAKE_HOST_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

if(${CMAKE_HOST_ARCH} MATCHES "arm64")
    set(CMAKE_SYSTEM_NAME Darwin)
    set(CMAKE_SYSTEM_PROCESSOR arm64)
elseif(${CMAKE_HOST_ARCH} MATCHES "")
    set(CMAKE_SYSTEM_NAME Windows)
    set(CMAKE_SYSTEM_PROCESSOR x86)
endif()

# to get supported archs on macOS: clang --print-supported-cpus

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if(${CMAKE_HOST_ARCH} MATCHES "arm64")
    # -mcpu=apple-a14 is like -march on gcc, I guess
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -ffast-math -mcpu=apple-a14")
elseif(${CMAKE_HOST_ARCH} MATCHES "")
    # SSE for Intel
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -mfpmath=sse -ffast-math")
endif()

set(CMAKE_C_FLAGS "")
set(CMAKE_C_FLAGS_DEBUG "-g")


if(${CMAKE_HOST_ARCH} MATCHES "arm64")
    # -mcpu=apple-a14 is like -march on gcc, I guess
    set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -ffast-math -mcpu=apple-a14")
elseif(${CMAKE_HOST_ARCH} MATCHES "")
    # SSE for Intel
    set(CMAKE_C_FLAGS_RELEASE "-O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -mfpmath=sse -ffast-math")
endif()


add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(bench)
add_subdirectory(test)

