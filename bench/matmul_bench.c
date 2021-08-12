#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "bench_utils.h"
#include "../lib/matrix.h"

const long long MILLION = 1000ll * 1000ll;
const long long BILLION = 1000ll * MILLION;
const long long MUL_REPEAT_COUNT = 10ll * MILLION;

long long LOG_PERIOD = 1ll * MILLION;

void double_multiply()
{
    int a_rows = 3;
    int a_cols = 2;
    const double a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *input_a_matrix_gen = generate_matrix_d(a_rows, a_cols, a_mat);

    int b_rows = 2;
    int b_cols = 3;
    const double b_mat[2][3] = {{5,0,3}, {4,1,7}};
    Matrix *input_b_matrix_gen = generate_matrix_d(b_rows, b_cols, b_mat);

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;

    Matrix *result_matrix_gen = generate_matrix_d(res_rows, res_cols, NULL);

    for (long long i = 0; i < MUL_REPEAT_COUNT; i++)
    {
        reset_matrix(result_matrix_gen);

        // DISP_MATRIX_ISET() branches every iteration, not ideal, therefore new version is added after
        // do randomization, invalidate cache
#if 0
        DISP_MATRIX_ISET(input_a_matrix_gen, rand()%a_rows, rand()%a_cols, (double)(rand()%100));
#endif
        matrix_assign_item_d(input_a_matrix_gen, rand()%a_rows, rand()%a_cols, (double)(rand()%100));

        multiply(input_a_matrix_gen, input_b_matrix_gen, result_matrix_gen);

        /*if (i % LOG_PERIOD == 0)
        {
            printf("double completed %.2f%\n", 100 * ((float)i /(float) MUL_REPEAT_COUNT));
        }*/
    }

    // Cleanup
    delete_matrix(input_a_matrix_gen);
    delete_matrix(input_b_matrix_gen);
    delete_matrix(result_matrix_gen);
}

void float_multiply()
{
    int a_rows = 3;
    int a_cols = 2;
    const float a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *input_a_matrix_gen = generate_matrix_f(a_rows, a_cols, a_mat);

    int b_rows = 2;
    int b_cols = 3;
    const float b_mat[2][3] = {{5,0,3}, {4,1,7}};
    Matrix *input_b_matrix_gen = generate_matrix_f(b_rows, b_cols, b_mat);

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;

    Matrix *output_res_matrix_gen = generate_matrix_f(res_rows, res_cols, NULL);

    for (long long i = 0; i < MUL_REPEAT_COUNT; i++)
    {
        reset_matrix(output_res_matrix_gen);

        // DISP_MATRIX_ISET() branches every iteration, not ideal, therefore new version is added after
        // do randomization, invalidate cache
#if 0
        DISP_MATRIX_ISET(input_a_matrix_gen, rand()%a_rows, rand()%a_cols, (float)(rand()%100));
#endif
        matrix_assign_item_f(input_a_matrix_gen, rand()%a_rows, rand()%a_cols, (float )(rand()%100));

        multiply(input_a_matrix_gen, input_b_matrix_gen, output_res_matrix_gen);

        /*if (i % LOG_PERIOD == 0)
        {
            printf("float completed %.2f%\n", 100 * ((float)i /(float) MUL_REPEAT_COUNT));
        }*/
    }

    // Cleanup
    delete_matrix(input_a_matrix_gen);
    delete_matrix(input_b_matrix_gen);
    delete_matrix(output_res_matrix_gen);
}

int main()
{
    printf("sizeof(float) == %ld\n", sizeof(float) );
    printf("sizeof(double) == %ld\n", sizeof(double) );

    assert(sizeof(float) == 4);
    assert(sizeof(double) == 8);

    int repeat_count = 10;

    double float_msecs = print_elapsed_time(float_multiply, "float", repeat_count);
    double double_msecs = print_elapsed_time(double_multiply, "double", repeat_count);

    printf("float over double speed-up factor: %fx\n", (double_msecs / float_msecs));

    // these results are obsolete, TODO: write results to separated txt file
    // mingw 64 gcc, windows 10, intel i7 cometlake
    /*
     *  sizeof(float) == 4
        sizeof(double) == 8
        float: Average time elapsed over 10 runs: 532.500000 ms
        double: Average time elapsed over 10 runs: 449.300000 ms
        float over double speed-up factor: 0.843756x
     *
     */

    /* macOS + Apple M1 + ARM64 + clang:
     * sizeof(float) == 4
     *   sizeof(double) == 8
     *   float: Average time elapsed over 10 runs: 475.628000 ms
     *   double: Average time elapsed over 10 runs: 629.374000 ms
     *   float over double speed-up factor: 1.323248x
     */
    /*
     * macOS + Apple M1 + ARM64 + clang + -mcpu=apple-a14:
     * sizeof(float) == 4
     * sizeof(double) == 8
     * float: Average time elapsed over 10 runs: 467.548000 ms
     * double: Average time elapsed over 10 runs: 619.646000 ms
     * float over double speed-up factor: 1.325310x
     *
     */

    /*
     * branching off:
     * mingw 64 gcc, windows 10, intel i7 cometlake
     * float: Average time elapsed over 10 runs: 349.900000 ms
     * double: Average time elapsed over 10 runs: 251.400000 ms
     *
     * macOS + Apple M1 + ARM64 + clang:
     * double: Average time elapsed over 10 runs: 623.303000 ms
     * float: Average time elapsed over 10 runs: 291.724000 ms
     */
}