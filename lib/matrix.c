#include "matrix.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

double** matrix_get_d(Matrix* x) { return x->d_matrix; }
float** matrix_get_f(Matrix* x) { return x->f_matrix; }

IntrospectionMode g_introspection_mode;

inline void matrix_item_assign(Matrix *x, Matrix *y, int row1, int col1, int row2, int col2 )
{
    DISP_MATRIX_ISET(x, row1, col1, DISP_MATRIX_IGET(y, row2, col2))
}

Matrix* generate_matrix(int rows, int cols, const double double_mat[][cols], const float float_mat[][cols], MatrixDataType dataType)
{
    Matrix *matrix_gen = (Matrix *) malloc (sizeof (Matrix));

    matrix_gen->rows = rows;
    matrix_gen->cols = cols;
    matrix_gen->matrix = NULL;
    matrix_gen->d_matrix = NULL;
    matrix_gen->f_matrix = NULL;
    matrix_gen->type = dataType;

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
            matrix_gen->matrix = (double **) malloc(sizeof(double *) * rows);
            for (int i = 0; i < rows; i++)
            {
                matrix_gen->matrix[i] = (double *) malloc(sizeof(double) * cols);
                for (int j = 0; j < cols; j++)
                {
                    if (double_mat != NULL)
                    {
                        DISP_MATRIX_ISET(matrix_gen, i, j, double_mat[i][j])
                    } else
                    {
                        DISP_MATRIX_ISET(matrix_gen, i, j, 0)
                    }
                }
            }
        }
        else
        {
            matrix_gen->f_matrix = (float**) malloc (sizeof (float*) * rows);
            for (int row = 0; row < rows; row++)
            {
                matrix_gen->f_matrix[row] = (float*) malloc (sizeof (float) * cols);
                for (int col = 0; col < cols; col++)
                {
                    if (float_mat != NULL)
                    {
                        float f_value = float_mat[row][col];

                        DISP_MATRIX_ISET(matrix_gen, row, col, f_value)
                    }
                    else
                    {
                        DISP_MATRIX_ISET(matrix_gen, row, col, 0)
                    }
                }
            }
        }
    }

    return matrix_gen;
}

void print_matrix(Matrix *matrix)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        printf("[ ");
        for (int j = 0; j < matrix->cols; j++)
        {
            printf("%.8f ", DISP_MATRIX_IGET(matrix, i, j));
        }
        printf("]\n");
    }    
}

// data_size == without management data (pointers and row col counts)
long get_matrix_data_size(Matrix *matrix)
{
    if (!matrix)
    {
        return 0L;
    }

    return matrix->rows * matrix->cols * ((matrix->type == D_FLOAT) ? sizeof(float) : sizeof(double));
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

int multiply(Matrix *in_a, Matrix *in_b, Matrix *out_result)
{
    if (is_null(in_a) || is_null(in_b) || is_null(out_result))
    {
        return -1;
    }

    if (in_a->cols != in_b->rows || in_a->rows != out_result->rows || in_b->cols != out_result->cols)
    {
        return -1;
    }

    for (int row = 0; row < in_a->rows; row++)
    {
        for (int col = 0; col < in_b->cols; col++)
        {
            DISP_MATRIX_ISET(out_result, row, col, 0)

            for (int k = 0; k < in_a->cols; k++)
            {
                DISP_MATRIX_IADD(out_result, row, col, DISP_MATRIX_IGET(in_a, row, k) * DISP_MATRIX_IGET(in_b, k, col))
            }            
        }        
    }

    return 0;
}

int multiply_transposed(Matrix *in_a, Matrix *in_b_t, Matrix *out_result)
{
    if (is_null(in_a) || is_null(in_b_t) || is_null(out_result))
    {
        return -1;
    }

    if (in_a->cols != in_b_t->cols || in_a->rows != out_result->rows || in_b_t->rows != out_result->cols)
    {
        return -1;
    }

    for (int i = 0; i < in_a->rows; i++)
    {
        for (int j = 0; j < in_b_t->rows; j++)
        {
            DISP_MATRIX_ISET(out_result, i, j, 0)

            for (int k = 0; k < in_a->cols; k++)
            {
                DISP_MATRIX_IADD(out_result, i, j, DISP_MATRIX_IGET(in_a, i, k) * DISP_MATRIX_IGET(in_b_t, j, k))
            }            
        }        
    }

    return 0;
}

int add(Matrix *in_out_a, Matrix *in_b)
{
    if (is_null(in_out_a) || is_null(in_b))
    {
        return -1;
    }

    if (in_out_a->rows != in_b->rows || in_out_a->cols != in_b->cols)
    {
        return -1;
    }

    for (int row = 0; row < in_out_a->rows; row++)
    {
        for (int col = 0; col < in_out_a->cols; col++)
        {
            DISP_MATRIX_IADD(in_out_a, row, col, DISP_MATRIX_IGET(in_b, row, col))
        }        
    }

    return 0;
}

int subtract(Matrix *in_out_a, Matrix *in_b)
{
    if (is_null(in_out_a) || is_null(in_b))
    {
        return -1;
    }

    if (in_out_a->rows != in_b->rows || in_out_a->cols != in_b->cols)
    {
        return -1;
    }

    for (int row = 0; row < in_out_a->rows; row++)
    {
        for (int col = 0; col < in_out_a->cols; col++)
        {
            DISP_MATRIX_ISUB(in_out_a, row, col, DISP_MATRIX_IGET(in_b, row, col))
        }        
    }

    return 0;
}

int scalar_multiply(Matrix *in_out_a, double in_x)
{
    if (is_null(in_out_a))
    {
        return -1;
    }

    for (int row = 0; row < in_out_a->rows; row++)
    {
        for (int col = 0; col < in_out_a->cols; col++)
        {
            DISP_MATRIX_IMULS(in_out_a, row, col, in_x)
        }        
    }  

    return 0;  
}

int scalar_add(Matrix *in_out_a, double in_x) {
    if (is_null(in_out_a))
    {
        return -1;
    }

    for (int row = 0; row < in_out_a->rows; row++)
    {
        for (int col = 0; col < in_out_a->cols; col++)
        {
            DISP_MATRIX_IADDS(in_out_a, row, col, in_x)
        }        
    }

    return 0;
}

// we need special _f version, because pointers to functions are incompatible between float and double
int apply_f(Matrix *in_a, Matrix *out_result, float (*fn)(float), int layer_idx)
{
    if (is_null(out_result))
    {
        out_result = in_a;
    }
    else if (in_a->rows != out_result->rows || in_a->cols != out_result->cols)
    {
        return -1;
    }

    if (is_null(in_a))
    {
        return -1;
    }

    if (fn == NULL)
    {
        return -1;
    }

    for (int row = 0; row < in_a->rows; row++)
    {
        for (int col = 0; col < in_a->cols; col++)
        {
#ifdef INTROSPECT
            float old_value = DISP_MATRIX_IGET(out_result, row, col);
#endif
            DISP_MATRIX_IAPPLY_FN(out_result, row, col, in_a, fn)

#ifdef INTROSPECT
            float new_value = DISP_MATRIX_IGET(out_result, row, col);

            on_neuron_activation_f(layer_idx, row, col, new_value);
#endif

            // TODO: call OnNeuronBackprop(layer, row, col, value)
        }
    }

    return 0;
}

int apply_d(Matrix *in_a, Matrix *out_result, double (*fn)(double), int layer_idx)    // TODO: add layer index param
{
    if (is_null(out_result))
    {
        out_result = in_a;
    }
    else if (in_a->rows != out_result->rows || in_a->cols != out_result->cols)
    {
        return -1;
    }    
    
    if (is_null(in_a))
    {
        return -1;
    }

    if (fn == NULL)
    {
        return -1;
    }

    for (int row = 0; row < in_a->rows; row++)
    {
        for (int col = 0; col < in_a->cols; col++)
        {
#ifdef INTROSPECT
            double old_value = DISP_MATRIX_IGET(out_result, row, col);
#endif

            DISP_MATRIX_IAPPLY_FN(out_result, row, col, in_a, fn)

#ifdef INTROSPECT
            float new_value = DISP_MATRIX_IGET(out_result, row, col);

            on_neuron_activation_d(layer_idx, row, col, new_value);
#endif

            // TODO: call OnNeuronBackprop(layer, row, col, value)
        }        
    }

    return 0;
}

// https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
/*
 * In mathematics, the Hadamard product (also known as the element-wise product, entrywise product[1][2]:ch. 5 or Schur product[3])
 * is a binary operation that takes two matrices of the same dimensions and produces another matrix of the same dimension as the operands,
 * where each element i, j is the product of elements i, j of the original two matrices.
 * It is to be distinguished from the more common matrix product. It is attributed to, and named after,
 * either French mathematician Jacques Hadamard or German mathematician Issai Schur.
 */
int hadamard(Matrix *in_a, Matrix *in_b, Matrix *out_result)
{
    if (is_null(in_a) || is_null(in_b))
    {
        return -1;
    }

    if (in_a->rows != in_b->rows || in_a->cols != in_b->cols || in_a->rows != out_result->rows || in_a->cols != out_result->cols)
    {
        return -1;
    }

    for (int row = 0; row < in_a->rows; row++)
    {
        for (int col = 0; col < in_a->cols; col++)
        {
            DISP_MATRIX_ISET(out_result, row, col, DISP_MATRIX_IGET(in_a, row, col) * DISP_MATRIX_IGET(in_b, row, col) )
        }        
    }

    return 0;    
}

int argmax(Matrix *in_a)
{
    int max = 0;

    if (in_a->rows == 1)
    {
        for (int i = 0; i < in_a->cols; i++)
        {
            if (DISP_MATRIX_IGET(in_a, 0, i) > DISP_MATRIX_IGET(in_a, 0, max))
            {
                max = i;
            }
        }        
    }
    else if (in_a->cols == 1)
    {
        for (int i = 0; i < in_a->rows; i++)
        {
            if (DISP_MATRIX_IGET(in_a, i, 0) > DISP_MATRIX_IGET(in_a, max, 0))
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

int reset_matrix(Matrix *in_out_a)
{
    if (is_null(in_out_a))
    {
        return -1;
    }

    for (int row = 0; row < in_out_a->rows; row++)
    {
        for (int col = 0; col < in_out_a->cols; col++)
        {
            DISP_MATRIX_ISET(in_out_a, row, col, 0)
        }        
    }

    return 0;    
}

int delete_matrix(Matrix *in_out_a)
{
    if (is_null(in_out_a))
    {
        return -1;
    }

    if (in_out_a->type == D_FLOAT)
    {
        for (int i = 0; i < in_out_a->rows; i++)
        {
            free(in_out_a->f_matrix[i]);
        }
    }
    else
    {
        for (int i = 0; i < in_out_a->rows; i++)
        {
            free(in_out_a->d_matrix[i]);
        }
    }

    if (in_out_a->type == D_FLOAT)
    {
        free(in_out_a->f_matrix);
    }
    else
    {
        free(in_out_a->d_matrix);
    }

    free(in_out_a);
    in_out_a = NULL;
    
    return 0;
}

bool is_float_matrix(Matrix *in_a)
{
    return in_a->type == D_FLOAT;
}

bool is_equal(Matrix *in_matrix, int in_rows, int in_cols, const double in_d_mat[in_rows][in_cols], const float in_f_mat[in_rows][in_cols])
{
    if (in_matrix->rows != in_rows || in_matrix->cols != in_cols)
    {
        return false;
    }

    for (int i = 0; i < in_rows; i++)
    {
        for (int j = 0; j < in_cols; j++)
        {
            if (is_float_matrix(in_matrix))
            {
                float f_value = (float) in_f_mat[i][j];

                if ((float)DISP_MATRIX_IGET(in_matrix, i, j) != f_value)
                {
                    return false;
                }
            }
            else
            {
                if (DISP_MATRIX_IGET(in_matrix, i, j) != in_d_mat[i][j])
                {
                    return false;
                }
            }
        }
    }

    return true;
}

void matrix_set_d(Matrix *out_x, double **in_mat)
{
    out_x->d_matrix = in_mat;
}

void matrix_set_f(Matrix *out_x, float **in_mat)
{
    out_x->f_matrix = in_mat;
}

Matrix *generate_matrix_f(int in_rows, int in_cols, const float in_float_mat[][in_cols])
{
    return generate_matrix(in_rows, in_cols, NULL, in_float_mat, D_FLOAT);
}

Matrix *generate_matrix_d(int in_rows, int in_cols, const double in_double_mat[][in_cols])
{
    return generate_matrix(in_rows, in_cols, in_double_mat, NULL, D_DOUBLE);
}

Matrix *generate_empty_matrix(int in_rows, int in_cols, MatrixDataType in_dataType)
{
    return generate_matrix(in_rows, in_cols, NULL, NULL, in_dataType);
}

void on_neuron_activation_f(int in_layer_idx, int in_row, int in_col, float in_value)
{
#ifdef INTROSPECT
    if (!g_introspection_file_handle)
    {
        //printf("g_introspection_file_handle == null!!!");
        return;
    }

    if (g_introspection_mode == IM_PREDICT)
    {
        fprintf(g_introspection_file_handle, "%ld %ld %ld : %f\n", in_layer_idx, in_row, in_col, in_value);
    }
#endif
}

void on_neuron_activation_d(int in_layer_idx, int in_row, int in_col, double in_value)
{
#ifdef INTROSPECT
    if (!g_introspection_file_handle)
    {
        //printf("g_introspection_file_handle == null!!!");
        return;
    }

    if (g_introspection_mode == IM_PREDICT)
    {
        fprintf(g_introspection_file_handle, "%ld %ld %ld : %f\n", in_layer_idx, in_row, in_col, in_value);
    }
#endif
}

void open_activation_introspection(const char *in_output_data_filename)
{
#ifdef INTROSPECT
    g_introspection_file_handle = fopen(in_output_data_filename, "w");
#else
    printf("Warning! Calling introspection functions, but INTROSPECT is undefined!!!");
#endif
}

void close_activation_introspection()
{
#ifdef INTROSPECT
    if (g_introspection_file_handle)
    {
        fclose(g_introspection_file_handle);
    }
#else
    printf("Warning! Calling introspection functions, but INTROSPECT is undefined!!!");
#endif
}

long get_matrix_arr_data_size(Matrix **in_matrix_arr, int in_len)
{
    long result = 0;

    for (int matrixIdx = 0; matrixIdx < in_len; matrixIdx++)
    {
        Matrix *currentMatrix = in_matrix_arr[matrixIdx];

        result += get_matrix_data_size(currentMatrix);
    }

    return result;
}
