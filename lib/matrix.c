#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

double** matrix_d(Matrix* x) { return x->d_matrix; }
float** MATRIX_F(Matrix* x) { return x->f_matrix; }

inline void matrix_item_assign(Matrix *x, Matrix *y, int col1, int row1, int col2, int row2)
{
    MATRIX_ISET(x, col1, row1, MATRIX_IGET(y, col2, row2));
}

inline void matrix_item_assign_direct(Matrix *x, Matrix *y, int col, int row)
{
    MATRIX_ISET(x, col, row, MATRIX_IGET(y, col, row))
}

Matrix* create_matrix(int rows, int cols, const double mat[][cols])
{
    Matrix *matrix = (Matrix *) malloc (sizeof (Matrix));

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->matrix = NULL;
    matrix->d_matrix = NULL;
    matrix->f_matrix = NULL;
    matrix->type = D_DOUBLE;

    if (rows > 0 && cols > 0) {
        matrix->matrix = (double**) malloc (sizeof (double*) *rows);
        for (int i = 0; i < rows; i++)
        {
            matrix->matrix[i] = (double*) malloc (sizeof (double) *cols);
            for (int j = 0; j < cols; j++)
            {
                if (mat != NULL)
                {
                    MATRIX_ISET(matrix, i, j, mat[i][j]);
                }
                else
                {
                    MATRIX_ISET(matrix, i, j, 0);
                }
            }
        }
    }

    return matrix;
}

Matrix* create_matrix_float(int rows, int cols, const float mat[][cols])
{
    Matrix *matrix = (Matrix *) malloc (sizeof (Matrix));

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->matrix = NULL;
    matrix->d_matrix = NULL;
    matrix->f_matrix = NULL;
    matrix->type = D_FLOAT;

    if (rows > 0 && cols > 0) {
        matrix->f_matrix = (float**) malloc (sizeof (float*) *rows);
        for (int i = 0; i < rows; i++)
        {
            matrix->f_matrix[i] = (float*) malloc (sizeof (float) *cols);
            for (int j = 0; j < cols; j++)
            {
                if (mat != NULL)
                {
                    MATRIX_ISET(matrix, i, j, mat[i][j]);
                }
                else
                {
                    MATRIX_ISET(matrix, i, j, 0);
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
            printf("%.2f ", MATRIX_IGET(matrix, i, j));
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

int transpose(Matrix *a, Matrix *result)
{
    if (is_null(a) || is_null(result))
    {
        return -1;
    }

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            matrix_item_assign(result, a, j, i, i, j);
        }        
    }
    
    return 0;
}

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

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < b->cols; j++)
        {
            MATRIX_ISET(result, i, j, 0);

            for (int k = 0; k < a->cols; k++)
            {
                MATRIX_IADD(result, i, j, MATRIX_IGET(a, i, k) * MATRIX_IGET(b, k, j))
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
            MATRIX_ISET(result, i, j, 0);

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

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            MATRIX_IADD(a, i, j, MATRIX_IGET(b, i, j));
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

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            MATRIX_ISUB(a, i, j, MATRIX_IGET(b, i, j));
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

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            MATRIX_IMULS(a, i, j, x);
        }        
    }  

    return 0;  
}

int scalar_add(Matrix *a, double x) {
    if (is_null(a))
    {
        return -1;
    }

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            MATRIX_IADDS(a, i, j, x);
        }        
    }

    return 0;
}

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

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            MATRIX_IAPPLY_FN(result, i, j, a, fn);
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
            MATRIX_IAPPLY_FN(result, i, j, a, fn);
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

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            MATRIX_ISET(result, i, j, MATRIX_IGET(a, i, j) * MATRIX_IGET(b, i, j) );
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

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            MATRIX_ISET(a, i, j, 0);
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

