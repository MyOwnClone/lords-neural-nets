#include <stdio.h>
#include <stdbool.h>
#include "../lib/matrix.h"
#include "test.h"
#include "utils.h"

static int test_create_matrix_float()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 3;

    float f_mat[3][3] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};

    Matrix *matrix = create_f_matrix(rows, cols, f_mat);

    // Test
    if (is_null(matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix is NULL");
    }

    if (!IS_EQUAL(matrix, rows, cols, f_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(matrix);

    // Test create empty matrix
    rows = 0;
    cols = 0;

    matrix = create_f_matrix(rows, cols, NULL);

    // Test
    if (matrix == NULL) {
        res+=fail(__func__,  __LINE__, "Matrix struct is NULL");
    }

    if (matrix && MATRIX_GET(matrix) != NULL) {
        res+=fail(__func__,  __LINE__, "Matrix in matrix struct is not NULL");
    }

    return eval_test_result(__func__, res);
}

static int test_create_matrix()
{
    // Setup
    int res = 0;


    int rows = 3;
    int cols = 3;

    double mat[3][3] = {{1,1,1}, {2,2,2}, {3,3,3}};

    Matrix *matrix = create_d_matrix(rows, cols, mat);

    // Test
    if (is_null(matrix))
    {
        res+=fail(__func__, __LINE__, "Matrix is NULL");
    }

    if (!IS_EQUAL(matrix, rows, cols, mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(matrix);

    // Test create empty matrix
    rows = 0;
    cols = 0;

    matrix = create_d_matrix(rows, cols, NULL);

    // Test
    if (matrix == NULL) {
        res+=fail(__func__,  __LINE__, "Matrix struct is NULL");
    }

    if (matrix->matrix != NULL) {
        res+=fail(__func__,  __LINE__, "Matrix in matrix struct is not NULL");
    }
   
    return eval_test_result(__func__, res);
}

static int test_is_null_float()
{
    // Setup
    int res = 0;

    Matrix *matrix = NULL;

    // Test null pointer
    if(!is_null(matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should be null");
    }

    // Test non null matrix
    matrix = create_f_matrix(1,1,NULL);
    if(is_null(matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");

    }

    // Test null array
    float **temp = MATRIX_GET(matrix);
    MATRIX_SET(matrix, NULL);

    if(!is_null(matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should be null");

    }

    // Cleanup
    MATRIX_SET(matrix, temp);
    delete_matrix(matrix);

    return eval_test_result(__func__, res);
}

static int test_is_null()
{
    // Setup
    int res = 0;

    Matrix *matrix = NULL;

    // Test null pointer
    if(!is_null(matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should be null");
    }

    // Test non null matrix
    matrix = create_d_matrix(1,1,NULL);
    if(is_null(matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");

    }

    // Test null array
    double **temp = MATRIX_GET(matrix);
    MATRIX_SET(matrix, NULL);

    if(!is_null(matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should be null");

    }

    // Cleanup
    MATRIX_SET(matrix, temp);
    delete_matrix(matrix);
   
    return eval_test_result(__func__, res);
}

static int test_transpose_float()
{
    // Setup
    int res = 0;

    int rows = 2;
    int cols = 4;

    float mat[2][4] = {
            {1,2,3,4},
            {5,6,6,7}
    };

    float transposed_mat[4][2] = {
            {1,5},
            {2,6},
            {3,6},
            {4,7}
    };

    Matrix *matrix = create_f_matrix(rows, cols, mat);
    Matrix *transposed = create_f_matrix(cols, rows, NULL);

    transpose(matrix, transposed);

    // Test
    if (is_null(transposed))
    {
        res+=fail(__func__,  __LINE__, "Transposed matrix is NULL\n");
    }

    if (!IS_EQUAL(transposed, cols, rows, transposed_mat))
    {
        res+=fail(__func__,  __LINE__, "Transposed matrix is not as expected!\n");
    }

    // Cleanup
    delete_matrix(matrix);
    delete_matrix(transposed);

    return eval_test_result(__func__, res);
}

static int test_transpose()
{
    // Setup
    int res = 0;

    int rows = 2;
    int cols = 4;

    double mat[2][4] = {
        {1,2,3,4},
        {5,6,6,7}
    };

    double transposed_mat[4][2] = {
        {1,5},
        {2,6},
        {3,6},
        {4,7}
    };

    Matrix *matrix = create_d_matrix(rows, cols, mat);
    Matrix *transposed = create_d_matrix(cols, rows, NULL);

    transpose(matrix, transposed);

    // Test
    if (is_null(transposed))
    {
        res+=fail(__func__,  __LINE__, "Transposed matrix is NULL\n");
    }

    if (!IS_EQUAL(transposed, cols, rows, transposed_mat))
    {
        res+=fail(__func__,  __LINE__, "Transposed matrix is not as expected!\n");
    }

    // Cleanup

    delete_matrix(matrix);
    delete_matrix(transposed);
   
    return eval_test_result(__func__, res);
}

static int test_multiply_double()
{
    // Setup
    int res = 0;

    int a_rows = 3;
    int a_cols = 2;
    const double a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_d_matrix(a_rows, a_cols, a_mat);

    int b_rows = 2;
    int b_cols = 3;
    const double b_mat[2][3] = {{5,0,3}, {4,1,7}};
    Matrix *b_matrix = create_d_matrix(b_rows, b_cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_d_matrix(c_rows, c_cols, NULL);

    // Test multiply wrong dimensions
    Matrix *res_wrong_dims_mat = create_d_matrix(a_cols, c_rows, NULL);
    int res_wrong_dims = multiply(a_matrix, c_matrix, res_wrong_dims_mat);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Mismatched dimension should not be multiplied");
    }

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;
    const double res_mat[3][3] = {
        {14.00, 1.00, 13.00 },
        {23.00, 2.00, 23.00 },
        {37.00, 3.00, 36.00 }
    };

    Matrix *res_matrix = create_d_matrix(res_rows, res_cols, NULL);
    multiply(a_matrix, b_matrix, res_matrix);

    if (is_null(res_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(res_matrix, res_rows, res_cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);
    delete_matrix(res_wrong_dims_mat);
    delete_matrix(res_matrix);
   
    return eval_test_result(__func__, res);
}

static int test_multiply_float()
{
    // Setup
    int res = 0;

    int a_rows = 3;
    int a_cols = 2;
    const float a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_f_matrix(a_rows, a_cols, a_mat);

    int b_rows = 2;
    int b_cols = 3;
    const float b_mat[2][3] = {{5,0,3}, {4,1,7}};
    Matrix *b_matrix = create_f_matrix(b_rows, b_cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_f_matrix(c_rows, c_cols, NULL);

    // Test multiply wrong dimensions
    Matrix *res_wrong_dims_mat = create_f_matrix(a_cols, c_rows, NULL);
    int res_wrong_dims = multiply(a_matrix, c_matrix, res_wrong_dims_mat);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Mismatched dimension should not be multiplied");
    }

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;
    const float res_mat[3][3] = {
            {14.00f, 1.00f, 13.00f },
            {23.00f, 2.00f, 23.00f },
            {37.00f, 3.00f, 36.00f }
    };

    Matrix *res_matrix = create_f_matrix(res_rows, res_cols, NULL);
    multiply(a_matrix, b_matrix, res_matrix);   // FIXME!

    if (is_null(res_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(res_matrix, res_rows, res_cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);
    delete_matrix(res_wrong_dims_mat);
    delete_matrix(res_matrix);

    return eval_test_result(__func__, res);
}

static int test_multiply_transposed_float()
{
    // Setup
    int res = 0;

    int a_rows = 3;
    int a_cols = 2;
    const float a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_f_matrix(a_rows, a_cols, a_mat);

    int b_rows = 3;
    int b_cols = 2;
    const float b_mat[3][2] = {{5,4}, {0,1}, {3,7}}; // Transposed: {{5,0,3}, {4,1,7}}
    Matrix *b_matrix = create_f_matrix(b_rows, b_cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_f_matrix(c_rows, c_cols, NULL);

    // Test multiply wrong dimensions
    Matrix *res_wrong_dims_mat = create_f_matrix(a_cols, c_rows, NULL);
    int res_wrong_dims = multiply_transposed(a_matrix, c_matrix, res_wrong_dims_mat);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Mismatched dimension should not be multiplied");
    }

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;
    const float res_mat[3][3] = {
            {14.00f, 1.00f, 13.00f },
            {23.00f, 2.00f, 23.00f },
            {37.00f, 3.00f, 36.00f }
    };

    Matrix *res_matrix = create_f_matrix(res_rows, res_cols, NULL);
    multiply_transposed(a_matrix, b_matrix, res_matrix);

    if (is_null(res_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(res_matrix, res_rows, res_cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);
    delete_matrix(res_wrong_dims_mat);
    delete_matrix(res_matrix);

    return eval_test_result(__func__, res);
}

static int test_multiply_transposed()
{
    // Setup
    int res = 0;

    int a_rows = 3;
    int a_cols = 2;
    const double a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_d_matrix(a_rows, a_cols, a_mat);

    int b_rows = 3;
    int b_cols = 2;
    const double b_mat[3][2] = {{5,4}, {0,1}, {3,7}}; // Transposed: {{5,0,3}, {4,1,7}}
    Matrix *b_matrix = create_d_matrix(b_rows, b_cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_d_matrix(c_rows, c_cols, NULL);

    // Test multiply wrong dimensions
    Matrix *res_wrong_dims_mat = create_d_matrix(a_cols, c_rows, NULL);
    int res_wrong_dims = multiply_transposed(a_matrix, c_matrix, res_wrong_dims_mat);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Mismatched dimension should not be multiplied");
    }

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 3;
    const double res_mat[3][3] = {
        {14.00, 1.00, 13.00 },
        {23.00, 2.00, 23.00 },
        {37.00, 3.00, 36.00 }
    };

    Matrix *res_matrix = create_d_matrix(res_rows, res_cols, NULL);
    multiply_transposed(a_matrix, b_matrix, res_matrix);

    if (is_null(res_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(res_matrix, res_rows, res_cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);
    delete_matrix(res_wrong_dims_mat);
    delete_matrix(res_matrix);
   
    return eval_test_result(__func__, res);
}

static int test_add_float()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 2;
    const float a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_f_matrix(rows, cols, a_mat);

    const float b_mat[3][2] = {{5,0}, {4,3}, {4,1}};
    Matrix *b_matrix = create_f_matrix(rows, cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_f_matrix(c_rows, c_cols, NULL);

    // Test add wrong dimensions
    int res_wrong_dims = add(a_matrix, c_matrix);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Sum of mismatched dimension matrices should not be possible");
    }

    // Test add correct dimensions
    const float res_mat[3][2] = {
            {7.00f, 1.00f },
            {7.00f, 5.00f },
            {9.00f, 4.00f }
    };

    add(a_matrix, b_matrix);

    if (is_null(a_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a_matrix, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);

    return eval_test_result(__func__, res);
}

static int test_add()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 2;
    const double a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_d_matrix(rows, cols, a_mat);

    const double b_mat[3][2] = {{5,0}, {4,3}, {4,1}};
    Matrix *b_matrix = create_d_matrix(rows, cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_d_matrix(c_rows, c_cols, NULL);

    // Test add wrong dimensions
    int res_wrong_dims = add(a_matrix, c_matrix);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Sum of mismatched dimension matrices should not be possible");
    }

    // Test add correct dimensions
    const double res_mat[3][2] = {
        {7.00, 1.00 },
        {7.00, 5.00 },
        {9.00, 4.00 }
    };

    add(a_matrix, b_matrix);

    if (is_null(a_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a_matrix, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);
   
    return eval_test_result(__func__, res);
}

static int test_subtract_float()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 2;
    const float a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_f_matrix(rows, cols, a_mat);

    const float b_mat[3][2] = {{5,0}, {4,3}, {4,1}};
    Matrix *b_matrix = create_f_matrix(rows, cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_f_matrix(c_rows, c_cols, NULL);

    // Test add wrong dimensions
    int res_wrong_dims = subtract(a_matrix, c_matrix);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Subtraction of mismatched dimension matrices should not be possible");
    }

    // Test add correct dimensions
    const float res_mat[3][2] = {
            {-3.00f, 1.00f },
            {-1.00f, -1.00f },
            {1.00f, 2.00f }
    };

    subtract(a_matrix, b_matrix);

    if (is_null(a_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a_matrix, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);

    return eval_test_result(__func__, res);
}

static int test_subtract()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 2;
    const double a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_d_matrix(rows, cols, a_mat);

    const double b_mat[3][2] = {{5,0}, {4,3}, {4,1}};
    Matrix *b_matrix = create_d_matrix(rows, cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_d_matrix(c_rows, c_cols, NULL);

    // Test add wrong dimensions
    int res_wrong_dims = subtract(a_matrix, c_matrix);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Subtraction of mismatched dimension matrices should not be possible");
    }

    // Test add correct dimensions
    const double res_mat[3][2] = {
        {-3.00, 1.00 },
        {-1.00, -1.00 },
        {1.00, 2.00 }
    };

    subtract(a_matrix, b_matrix);

    if (is_null(a_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a_matrix, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);
   
    return eval_test_result(__func__, res);
}

static int test_scalar_multiply_float()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 3;
    const float a_mat[3][3] = {{1,2,3}, {4,5,6}, {7,8,9}};
    Matrix *a = create_f_matrix(rows, cols, a_mat);

    double x = 0.5;
    const float res_mat[3][3] = {{0.5f, 1, 1.5f}, {2, 2.5f, 3}, {3.5f, 4, 4.5f}};
    scalar_multiply(a, x);

    // Test
    if (is_null(a))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a);

    return eval_test_result(__func__, res);
}

static int test_scalar_multiply()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 3;
    const double a_mat[3][3] = {{1,2,3}, {4,5,6}, {7,8,9}};
    Matrix *a = create_d_matrix(rows, cols, a_mat);

    double x = 0.5;
    const double res_mat[3][3] = {{0.5,1,1.5}, {2,2.5,3}, {3.5,4,4.5}};
    scalar_multiply(a, x);

    // Test
    if (is_null(a))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a); 
   
    return eval_test_result(__func__, res);
}

static int test_scalar_add_float()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 3;
    const float a_mat[3][3] = {{1,2,3}, {4,5,6}, {7,8,9}};
    Matrix *a = create_f_matrix(rows, cols, a_mat);

    double x = 10.5;
    const float res_mat[3][3] = {{11.5f,12.5f,13.5f}, {14.5f,15.5f,16.5f}, {17.5f,18.5f,19.5f}};

    scalar_add(a, x);

    // Test
    if (is_null(a))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a);

    return eval_test_result(__func__, res);
}

static int test_scalar_add()
{
    // Setup
    int res = 0;

    int rows = 3;
    int cols = 3;
    const double a_mat[3][3] = {{1,2,3}, {4,5,6}, {7,8,9}};
    Matrix *a = create_d_matrix(rows, cols, a_mat);

    double x = 10.5;
    const double res_mat[3][3] = {{11.5,12.5,13.5}, {14.5,15.5,16.5}, {17.5,18.5,19.5}};

    scalar_add(a, x);

    // Test
    if (is_null(a))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a);   
   
    return eval_test_result(__func__, res);
}

static double square(double num)
{
    return num*num;
}

static float square_f(float num)
{
    return num*num;
}

static int test_apply_float()
{
    // Setup
    int res = 0;

    float (*square_ptr)(float) = &square_f;

    int rows = 2;
    int cols = 2;
    const float a_mat[2][2] = {{1,2}, {3,4}};
    Matrix *a_matrix = create_f_matrix(rows, cols, a_mat);

    const float res_mat[2][2] = {{1,4}, {9, 16}};

    APPLY(a_matrix, NULL, square_ptr);

    if (is_null(a_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a_matrix, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    delete_matrix(a_matrix);
    square_ptr = NULL;

    return eval_test_result(__func__, res);
}

static int test_apply()
{
    // Setup
    int res = 0;

    double (*square_ptr)(double) = &square;

    int rows = 2;
    int cols = 2;
    const double a_mat[2][2] = {{1,2}, {3,4}};
    Matrix *a_matrix = create_d_matrix(rows, cols, a_mat);

    const double res_mat[2][2] = {{1,4}, {9, 16}};

    APPLY(a_matrix, NULL, square_ptr);

    if (is_null(a_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(a_matrix, rows, cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    delete_matrix(a_matrix);
    square_ptr = NULL;
   
    return eval_test_result(__func__, res);
}

static int test_hadamard_float()
{
    // Setup
    int res = 0;

    int a_rows = 3;
    int a_cols = 2;
    const float a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_f_matrix(a_rows, a_cols, a_mat);

    int b_rows = 3;
    int b_cols = 2;
    const float b_mat[3][2] = {{5,0}, {3,4}, {1,7}};
    Matrix *b_matrix = create_f_matrix(b_rows, b_cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_f_matrix(c_rows, c_cols, NULL);

    // Test multiply wrong dimensions
    Matrix *res_wrong_dims_mat = create_f_matrix(a_cols, c_rows, NULL);
    int res_wrong_dims = hadamard(a_matrix, c_matrix, res_wrong_dims_mat);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Mismatched dimension should not be multiplied");
    }

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 2;
    const float res_mat[3][2] = {
            {10.00f, 0.00f},
            {9.00f, 8.00f},
            {5.00f, 21.00f}
    };

    Matrix *res_matrix = create_f_matrix(res_rows, res_cols, NULL);
    hadamard(a_matrix, b_matrix, res_matrix);

    if (is_null(res_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(res_matrix, res_rows, res_cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);
    delete_matrix(res_wrong_dims_mat);
    delete_matrix(res_matrix);

    return eval_test_result(__func__, res);
}

static int test_hadamard()
{
    // Setup
    int res = 0;

    int a_rows = 3;
    int a_cols = 2;
    const double a_mat[3][2] = {{2,1}, {3,2}, {5,3}};
    Matrix *a_matrix = create_d_matrix(a_rows, a_cols, a_mat);

    int b_rows = 3;
    int b_cols = 2;
    const double b_mat[3][2] = {{5,0}, {3,4}, {1,7}};
    Matrix *b_matrix = create_d_matrix(b_rows, b_cols, b_mat);

    int c_rows = 4;
    int c_cols = 5;
    Matrix *c_matrix = create_d_matrix(c_rows, c_cols, NULL);

    // Test multiply wrong dimensions
    Matrix *res_wrong_dims_mat = create_d_matrix(a_cols, c_rows, NULL);
    int res_wrong_dims = hadamard(a_matrix, c_matrix, res_wrong_dims_mat);
    if (res_wrong_dims != -1)
    {
        res+=fail(__func__,  __LINE__, "Mismatched dimension should not be multiplied");
    }

    // Test multiply correct dimensions
    int res_rows = 3;
    int res_cols = 2;
    const double res_mat[3][2] = {
        {10.00, 0.00},
        {9.00, 8.00},
        {5.00, 21.00}
    };

    Matrix *res_matrix = create_d_matrix(res_rows, res_cols, NULL);
    hadamard(a_matrix, b_matrix, res_matrix);

    if (is_null(res_matrix))
    {
        res+=fail(__func__,  __LINE__, "Matrix should not be null");
    }

    if (!IS_EQUAL(res_matrix, res_rows, res_cols, res_mat))
    {
        res+=fail(__func__,  __LINE__, "Wrong matrix dimensions or values");
    }

    // Cleanup
    delete_matrix(a_matrix);
    delete_matrix(b_matrix);
    delete_matrix(c_matrix);
    delete_matrix(res_wrong_dims_mat);
    delete_matrix(res_matrix);
   
    return eval_test_result(__func__, res);
}

static int test_argmax_float()
{
    // Setup
    int res = 0;
    float mat[10][1];
    for (int i = 0; i < 10; i++)
    {
        mat[i][0] = (float) i / 10;
    }

    Matrix *a = create_f_matrix(10, 1, mat);

    // Tests
    int arg = argmax(a);

    if (arg != 9)
    {
        res+=fail(__func__,  __LINE__, "Wrong argmax result");
    }

    int index = 4;
    MATRIX_IADD(a, index, 0, 10);

    arg = argmax(a);

    if (arg != index)
    {
        res+=fail(__func__,  __LINE__, "Wrong argmax result");
    }

    // Cleanup
    delete_matrix(a);
    return eval_test_result(__func__, res);
}

static int test_argmax()
{
    // Setup
    int res = 0;
    double mat[10][1];
    for (int i = 0; i < 10; i++)
    {
        mat[i][0] = (double) i / 10;
    }

    Matrix *a = create_d_matrix(10, 1, mat);

    // Tests
    int arg = argmax(a);

    if (arg != 9)
    {
        res+=fail(__func__,  __LINE__, "Wrong argmax result");
    }

    int index = 4;
    a->matrix[index][0] += 10;

    arg = argmax(a);

    if (arg != index)
    {
        res+=fail(__func__,  __LINE__, "Wrong argmax result");
    }

    // Cleanup
    delete_matrix(a);
    return eval_test_result(__func__, res);
}

int test_matrix()
{
    int res = 0;
    res += test_create_matrix();
    res += test_create_matrix_float();
    res += test_is_null();
    res += test_is_null_float();
    res += test_transpose();
    res += test_transpose_float();
    res += test_multiply_double();
    res += test_multiply_float();
    res += test_multiply_transposed();
    res += test_multiply_transposed_float();
    res += test_add_float();
    res += test_add();
    res += test_subtract();
    res += test_subtract_float();
    res += test_scalar_multiply();
    res += test_scalar_multiply_float();
    res += test_scalar_add();
    res += test_scalar_add_float();
    res += test_apply_float();
    res += test_apply();
    res += test_hadamard();
    res += test_hadamard_float();
    res += test_argmax();
    res += test_argmax_float();

    return res;
}