#include <stdlib.h>
#include <stdio.h>
#include "bench_utils.h"

const long long ITERATION_COUNT = 100ll*1000ll;

float scalar_muladd_fp32()
{
    float weight = 2.0f, value = 1.0f;
    float bias = 0.1f;

    float sum = 0;

    for (long long it = 0; it < ITERATION_COUNT; it++ )
    {
        float rand_element = rand() % 1000;
        sum += weight * value + bias + rand_element;
    }

    return sum;
}

double scalar_muladd_fp64()
{
    double weight = 2.0f, value = 1.0f;
    double bias = 0.1f;

    double sum = 0;

    for (long long it = 0; it < ITERATION_COUNT; it++ )
    {
        double rand_element = rand() % 1000;
        sum += weight * value + bias + rand_element;
    }

    return sum;
}

int main()
{
    long long repeat_count = 1000ll * 100ll;

    double float_msecs = print_elapsed_time(scalar_muladd_fp32, "muladd float", repeat_count);
    double double_msecs = print_elapsed_time(scalar_muladd_fp64, "muladd double", repeat_count);

    printf("float over double speed-up factor: %fx\n", (double_msecs / float_msecs));

    // these results are obsole, TODO: write results to separated txt file
    // mingw 64 gcc, windows 10, intel i7 cometlake

    /*
     *   when using (x64): -O3 -finline-functions -m64 -funroll-loops -oFast -funsafe-math-optimizations -mfpmath=sse2 -ffast-math -march=rocketlake

        muladd float: Average time elapsed over 100000 runs: 0.987450 ms
        muladd double: Average time elapsed over 100000 runs: 0.999490 ms
        float over double speed-up factor: 1.012193x

     */

    /*
     * macOS + Apple M1 + ARM64 + clang:

     muladd float: Average time elapsed over 100000 runs: 0.653000 ms
     muladd double: Average time elapsed over 100000 runs: 0.648000 ms
     float over double speed-up factor: 0.992343x

     */
}