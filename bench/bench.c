#include <matrix.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "bench_utils.h"

const long MILLION = 1000l * 1000l;
const long BILLION = 1000l * MILLION;
const long TEST_REPEAT_COUNT = 100l * MILLION;

void double_multiply()
{
    int a_rows = 3;
    int a_cols = 2;
    const double a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_matrix(a_rows, a_cols, a_mat);

    int b_rows = 2;
    int b_cols = 3;
    const double b_mat[2][3] = {{5,0,3}, {4,1,7}};
    Matrix *b_matrix = create_matrix(b_rows, b_cols, b_mat);

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;

    Matrix *res_matrix = create_matrix(res_rows, res_cols, NULL);

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
    Matrix *a_matrix = create_matrix_float(a_rows, a_cols, a_mat);

    int b_rows = 2;
    int b_cols = 3;
    const float b_mat[2][3] = {{5,0,3}, {4,1,7}};
    Matrix *b_matrix = create_matrix_float(b_rows, b_cols, b_mat);

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;

    Matrix *res_matrix = create_matrix_float(res_rows, res_cols, NULL);

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
    print_elapsed_time(float_multiply, "float");
    print_elapsed_time(double_multiply, "double");
}