#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

#define INTROSPECT

typedef enum
{
    D_FLOAT = 0,
    D_DOUBLE = 1
} MatrixDataType;

typedef struct 
{
    MatrixDataType type;
    int rows;
    int cols;

    float **f_matrix;

    union {
        double **matrix;    // just a hack to be able to use old name 'matrix' and also new name 'd_matrix' for the same double matrix
        double **d_matrix;
    };
} Matrix;

// TODO: in memory dataset/loader

double** matrix_get_d(Matrix* x);
float** matrix_get_f(Matrix* x);

void matrix_set_d(Matrix* x, double **mat);
void matrix_set_f(Matrix* x, float **mat);

// __ == internal

#define __IS_FLOAT(x) ((x)->type == D_FLOAT)
//#define IS_FLOAT(x) true

// all the DISP_ macros (dispatcher) are basically a way to avoid static typing screaming at us that we are using different types: float or double, macros do not check this
// unfortunately, we do not have anything like templates or generics in pure C
// i.e.: DISP_MATRIX_ISET() tak "val" parameter, user can supply double or float, there is a condition which based on type of matrix select double or float assignment, but we can still have one macro, not 2 functions

// Item GET
#define DISP_MATRIX_IGET(x, row, col) (__IS_FLOAT(x) ? (float)(matrix_get_f(x)[row][col]) : (double)(matrix_get_d(x)[row][col]))

// Item SET
#define DISP_MATRIX_ISET(x, row, col, val) if (__IS_FLOAT(x)) { \
                                (x)->f_matrix[row][col] = val; \
                                } else {\
                                (x)->d_matrix[row][col] = val; };

void matrix_item_assign(Matrix *x, Matrix *y, int row1, int col1, int row2, int col2);

// Item APPLY FuNction
#define DISP_MATRIX_IAPPLY_FN(result, row, col, source, function) DISP_MATRIX_ISET(result, row, col, function(DISP_MATRIX_IGET(source, row, col)))

// Item ADD
#define DISP_MATRIX_IADD(x, row, col, val) DISP_MATRIX_ISET(x, row, col, DISP_MATRIX_IGET(x, row, col) + (val))

// Item SUBtract
#define DISP_MATRIX_ISUB(x, row, col, val) DISP_MATRIX_ISET(x, row, col, DISP_MATRIX_IGET(x, row, col) - (val))

// Item MULtiply by Scalar
#define DISP_MATRIX_IMULS(x, row, col, val) DISP_MATRIX_ISET(x, row, col, DISP_MATRIX_IGET(x, row, col) * (val))
#define DISP_MATRIX_IADDS(x, row, col, val) DISP_MATRIX_ISET(x, row, col, DISP_MATRIX_IGET(x, row, col) + (val))

Matrix* create_matrix(int rows, int cols, const double double_mat[][cols], const float float_mat[][cols], MatrixDataType dataType);

Matrix* create_empty_matrix(int rows, int cols, MatrixDataType dataType);

Matrix* create_matrix_f(int rows, int cols, const float float_mat[][cols]);
Matrix* create_matrix_d(int rows, int cols, const double double_mat[][cols]);

void print_matrix(Matrix *matrix);
bool is_null(Matrix *matrix);
int transpose(Matrix *a, Matrix *result);
int multiply(Matrix *a, Matrix *b, Matrix *result);
int multiply_transposed(Matrix *a, Matrix *b_t, Matrix *result);
int add(Matrix *a, Matrix *b);
int subtract(Matrix *a, Matrix *b);
int scalar_multiply(Matrix *matrix, double a);
int scalar_add(Matrix *matrix, double a);
int apply_d(Matrix *a, Matrix *result, double (*fn)(double), int layer_idx);
int apply_f(Matrix *a, Matrix *result, float (*fn)(float), int layer_idx);
int hadamard(Matrix *a, Matrix *b, Matrix *result);
int argmax(Matrix *a);
int reset_matrix(Matrix *a);
int delete_matrix(Matrix *a);
bool is_float_matrix(Matrix *a);

bool is_equal(Matrix *matrix, int rows, int cols, const double d_mat[rows][cols], const float f_mat[rows][cols]);

#define DISP_IS_EQUAL(mat, rows, cols, other_mat) ((is_float_matrix(mat) == true) && (is_equal(mat, rows, cols, NULL, other_mat) == true) || (is_equal(mat, rows, cols, other_mat, NULL)) == true)

#define DISP_MATRIX_SET(matrix, p_mat) if (is_float_matrix(matrix)) \
    matrix_set_f(matrix, p_mat);                                \
else                                                           \
    matrix_set_d(matrix, p_mat);

#define DISP_MATRIX_GET(matrix) (is_float_matrix(matrix) == true) ? matrix_get_f(matrix) : matrix_get_d(matrix)

#define DISP_APPLY(matrix, matrix_result, fn) if (is_float_matrix(matrix)) \
    apply_f(matrix, matrix_result, fn, -1);                                \
else                                                                  \
    apply_d(matrix, matrix_result, fn, -1);

void on_neuron_activation_f(int layer_idx, int row, int col, float value);
void on_neuron_activation_d(int layer_idx, int row, int col, double value);

void open_activation_introspection(const char *data_filename);
void close_activation_introspection();

#include <stdio.h>
extern FILE* g_introspection_file_handle;

typedef enum {
    IM_NONE,
    IM_PREDICT
} IntrospectionMode;

extern IntrospectionMode g_introspection_mode;

#endif /* MATRIX_H */
