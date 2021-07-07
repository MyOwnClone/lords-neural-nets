#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

double** matrix_get_d(Matrix* x) { return x->d_matrix; }
float** matrix_get_f(Matrix* x) { return x->f_matrix; }

inline void matrix_item_assign(Matrix *x, Matrix *y, int row1, int col1, int row2, int col2 )
{
    MATRIX_ISET(x, row1, col1, MATRIX_IGET(y, row2, col2))
}

Matrix* create_matrix(int rows, int cols, const double double_mat[][cols], const float float_mat[][cols], MatrixDataType dataType)
{
    Matrix *matrix = (Matrix *) malloc (sizeof (Matrix));

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->matrix = NULL;
    matrix->d_matrix = NULL;
    matrix->f_matrix = NULL;
    matrix->type = dataType;

    if (dataType == D_FLOAT && float_mat == NULL && double_mat != NULL)
    {
        printf("DataType set to a float but a double matrix is set instead!");

        return NULL;
    }

    if (dataType == D_DOUBLE && double_mat == NULL && float_mat != NULL)
    {
        printf("DataType set to a double but a float matrix is set instead!");

        return NULL;
    }

    if (rows > 0 && cols > 0) {
        if (dataType == D_DOUBLE)
        {
            matrix->matrix = (double **) malloc(sizeof(double *) * rows);
            for (int i = 0; i < rows; i++)
            {
                matrix->matrix[i] = (double *) malloc(sizeof(double) * cols);
                for (int j = 0; j < cols; j++)
                {
                    if (double_mat != NULL)
                    {
                        MATRIX_ISET(matrix, i, j, double_mat[i][j])
                    } else
                    {
                        MATRIX_ISET(matrix, i, j, 0)
                    }
                }
            }
        }
        else
        {
            matrix->f_matrix = (float**) malloc (sizeof (float*) *rows);
            for (int row = 0; row < rows; row++)
            {
                matrix->f_matrix[row] = (float*) malloc (sizeof (float) *cols);
                for (int col = 0; col < cols; col++)
                {
                    if (float_mat != NULL)
                    {
                        float f_value = float_mat[row][col];

                        MATRIX_ISET(matrix, row, col, f_value)
                    }
                    else
                    {
                        MATRIX_ISET(matrix, row, col, 0)
                    }
                }
            }
        }
    }

    return matrix;
}

void print_matrix(Matrix *matrix)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        printf("[ ");
        for (int j = 0; j < matrix->cols; j++)
        {
            printf("%.8f ", MATRIX_IGET(matrix, i, j));
        }
        printf("]\n");
    }    
}

bool is_null(Matrix *a)
{
    if (a == NULL || (a->matrix == NULL && a->f_matrix == NULL))
    {
        return true;
    }

    return false;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ArgumentSelectionDefects"
int transpose(Matrix *a, Matrix *result)
{
    if (is_null(a) || is_null(result))
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < a->cols; col++)
        {
            matrix_item_assign(result, a, col, row, row, col);
        }        
    }
    
    return 0;
}
#pragma clang diagnostic pop

int multiply(Matrix *a, Matrix *b, Matrix *result)
{
    if (is_null(a) || is_null(b) || is_null(result))
    {
        return -1;
    }

    if (a->cols != b->rows || a->rows != result->rows || b->cols != result->cols)
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < b->cols; col++)
        {
            MATRIX_ISET(result, row, col, 0)

            for (int k = 0; k < a->cols; k++)
            {
                MATRIX_IADD(result, row, col, MATRIX_IGET(a, row, k) * MATRIX_IGET(b, k, col))
            }            
        }        
    }

    return 0;
}

int multiply_transposed(Matrix *a, Matrix *b_t, Matrix *result)
{
    if (is_null(a) || is_null(b_t) || is_null(result))
    {
        return -1;
    }

    if (a->cols != b_t->cols || a->rows != result->rows || b_t->rows != result->cols)
    {
        return -1;
    }

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < b_t->rows; j++)
        {
            MATRIX_ISET(result, i, j, 0)

            for (int k = 0; k < a->cols; k++)
            {
                MATRIX_IADD(result, i, j, MATRIX_IGET(a, i, k) * MATRIX_IGET(b_t, j, k))
            }            
        }        
    }

    return 0;
}

int add(Matrix *a, Matrix *b)
{
    if (is_null(a) || is_null(b))
    {
        return -1;
    }

    if (a->rows != b->rows || a->cols != b->cols)
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < a->cols; col++)
        {
            MATRIX_IADD(a, row, col, MATRIX_IGET(b, row, col))
        }        
    }

    return 0;
}

int subtract(Matrix *a, Matrix *b)
{
    if (is_null(a) || is_null(b))
    {
        return -1;
    }

    if (a->rows != b->rows || a->cols != b->cols)
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < a->cols; col++)
        {
            MATRIX_ISUB(a, row, col, MATRIX_IGET(b, row, col))
        }        
    }

    return 0;
}

int scalar_multiply(Matrix *a, double x)
{
    if (is_null(a))
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < a->cols; col++)
        {
            MATRIX_IMULS(a, row, col, x)
        }        
    }  

    return 0;  
}

int scalar_add(Matrix *a, double x) {
    if (is_null(a))
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < a->cols; col++)
        {
            MATRIX_IADDS(a, row, col, x)
        }        
    }

    return 0;
}

// we need special _f version, because pointers to functions are incompatible between float and double
int apply_f(Matrix *a, Matrix *result, float (*fn)(float))
{
    if (is_null(result))
    {
        result = a;
    }
    else if (a->rows != result->rows || a->cols != result->cols)
    {
        return -1;
    }

    if (is_null(a))
    {
        return -1;
    }

    if (fn == NULL)
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < a->cols; col++)
        {
            MATRIX_IAPPLY_FN(result, row, col, a, fn)
        }
    }

    return 0;
}

int apply(Matrix *a, Matrix *result, double (*fn)(double))
{
    if (is_null(result))
    {
        result = a;
    }
    else if (a->rows != result->rows || a->cols != result->cols)
    {
        return -1;
    }    
    
    if (is_null(a))
    {
        return -1;
    }

    if (fn == NULL)
    {
        return -1;
    }

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            MATRIX_IAPPLY_FN(result, i, j, a, fn)
        }        
    }

    return 0;
}

int hadamard(Matrix *a, Matrix *b, Matrix *result)
{
    if (is_null(a) || is_null(b))
    {
        return -1;
    }

    if (a->rows != b->rows || a->cols != b->cols || a->rows != result->rows || a->cols != result->cols)
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < a->cols; col++)
        {
            MATRIX_ISET(result, row, col, MATRIX_IGET(a, row, col) * MATRIX_IGET(b, row, col) )
        }        
    }

    return 0;    
}

int argmax(Matrix *a)
{
    int max = 0;

    if (a->rows == 1)
    {
        for (int i = 0; i < a->cols; i++)
        {
            if (MATRIX_IGET(a, 0, i) > MATRIX_IGET(a, 0, max))
            {
                max = i;
            }
        }        
    }
    else if (a->cols == 1)
    {
        for (int i = 0; i < a->rows; i++)
        {
            if (MATRIX_IGET(a, i, 0) > MATRIX_IGET(a, max, 0))
            {
                max = i;
            }
        }   
    }
    else
    {
        max = -1;
    }

    return max;    
}

int reset_matrix(Matrix *a)
{
    if (is_null(a))
    {
        return -1;
    }

    for (int row = 0; row < a->rows; row++)
    {
        for (int col = 0; col < a->cols; col++)
        {
            MATRIX_ISET(a, row, col, 0)
        }        
    }

    return 0;    
}

int delete_matrix(Matrix *a)
{
    if (is_null(a))
    {
        return -1;
    }

    if (a->type == D_FLOAT)
    {
        for (int i = 0; i < a->rows; i++)
        {
            free(a->f_matrix[i]);
        }
    }
    else
    {
        for (int i = 0; i < a->rows; i++)
        {
            free(a->d_matrix[i]);
        }
    }

    if (a->type == D_FLOAT)
    {
        free(a->f_matrix);
    }
    else
    {
        free(a->d_matrix);
    }

    free(a);
    a = NULL;
    
    return 0;
}

bool is_float_matrix(Matrix *a)
{
    return a->type == D_FLOAT;
}

bool is_equal(Matrix *matrix, int rows, int cols, const double d_mat[rows][cols], const float f_mat[rows][cols])
{
    if (matrix->rows != rows || matrix->cols != cols)
    {
        return false;
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (is_float_matrix(matrix))
            {
                float f_value = (float) f_mat[i][j];

                if ((float)MATRIX_IGET(matrix, i, j) != f_value)
                {
                    return false;
                }
            }
            else
            {
                if (MATRIX_IGET(matrix, i, j) != d_mat[i][j])
                {
                    return false;
                }
            }
        }
    }

    return true;
}

void matrix_set_d(Matrix *x, double **mat)
{
    x->d_matrix = mat;
}

void matrix_set_f(Matrix *x, float **mat)
{
    x->f_matrix = mat;
}

Matrix *create_matrix_f(int rows, int cols, const float **float_mat)
{
    return create_matrix(rows, cols, NULL, float_mat, D_FLOAT);
}

Matrix *create_matrix_d(int rows, int cols, const double **double_mat)
{
    return create_matrix(rows, cols, double_mat, NULL, D_DOUBLE);
}

Matrix *create_empty_matrix(int rows, int cols, MatrixDataType dataType)
{
    return create_matrix(rows, cols, NULL,NULL, dataType);
}

