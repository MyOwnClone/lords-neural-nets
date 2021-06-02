#include <matrix.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include "bench_utils.h"

const long MILLION = 1000l * 1000l;
const long BILLION = 1000l * MILLION;
const long TEST_REPEAT_COUNT = 500l * MILLION;

void double_multiply()
{
    int a_rows = 3;
    int a_cols = 2;
    const double a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_matrix(a_rows, a_cols, a_mat, NULL, D_DOUBLE);

    int b_rows = 2;
    int b_cols = 3;
    const double b_mat[2][3] = {{5,0,3}, {4,1,7}};
    Matrix *b_matrix = create_matrix(b_rows, b_cols, b_mat, NULL, D_DOUBLE);

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;

    Matrix *res_matrix = create_matrix(res_rows, res_cols, NULL, NULL, D_DOUBLE);

    for (long i = 0; i < TEST_REPEAT_COUNT; i++)
    {
        reset_matrix(res_matrix);
        multiply(a_matrix, b_matrix, res_matrix);
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(res_matrix);
}

void float_multiply()
{
    int a_rows = 3;
    int a_cols = 2;
    const float a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_matrix(a_rows, a_cols, NULL, a_mat, D_FLOAT);

    int b_rows = 2;
    int b_cols = 3;
    const float b_mat[2][3] = {{5,0,3}, {4,1,7}};
    Matrix *b_matrix = create_matrix(b_rows, b_cols, NULL, b_mat, D_FLOAT);

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;

    Matrix *res_matrix = create_matrix(res_rows, res_cols, NULL, NULL, D_FLOAT);

    for (long i = 0; i < TEST_REPEAT_COUNT; i++)
    {
        reset_matrix(res_matrix);
        multiply(a_matrix, b_matrix, res_matrix);
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(res_matrix);
}

int main()
{
    printf("sizeof(float) == %ld\n", sizeof(float) );
    printf("sizeof(double) == %ld\n", sizeof(double) );

    assert(sizeof(float) == 4);
    assert(sizeof(double) == 8);

    long repeat_count = 10;

    double float_msecs = print_elapsed_time(float_multiply, "float", repeat_count);
    double double_msecs = print_elapsed_time(double_multiply, "double", repeat_count);

    printf("float over double speed-up factor: %fx\n", (double_msecs / float_msecs));
}