#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>
#include <corecrt.h>

//#define INTROSPECT

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

    // all the *_matrix field should be technically called with suffix _gen, cause memory is malloced, but it would pollute code too much

    float **f_matrix;
    double **d_matrix;
} Matrix;

// TODO: in memory dataset/loader

double** matrix_get_d(Matrix* x);
float** matrix_get_f(Matrix* x);

void matrix_set_d(Matrix* out_x, double **in_mat);
void matrix_set_f(Matrix* out_x, float **in_mat);

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

void matrix_assign_item_d(Matrix *in_out_x, int in_row, int in_col, double value);
void matrix_assign_item_f(Matrix *in_out_x, int in_row, int in_col, float value);

double matrix_get_item_d(Matrix *in_x, int in_row, int in_col);
float matrix_get_item_f(Matrix *in_out_x, int in_row, int in_col);

void matrix_assign_item_from_other(Matrix *in_out_x, Matrix *in_y, int in_row1, int in_col1, int in_row2, int in_col2);

// Item APPLY FuNction
#define DISP_MATRIX_IAPPLY_FN(result, row, col, source, function) DISP_MATRIX_ISET(result, row, col, function(DISP_MATRIX_IGET(source, row, col)))

void matrix_item_apply_fn_f(Matrix *out_result, int in_row, int in_col, Matrix *in_source, float (*fn)(float));
void matrix_item_apply_fn_d(Matrix *out_result, int in_row, int in_col, Matrix *in_source, double (*fn)(double));

// Item ADD
#define DISP_MATRIX_IADD(x, row, col, val) DISP_MATRIX_ISET(x, row, col, DISP_MATRIX_IGET(x, row, col) + (val))

// Item SUBtract
#define DISP_MATRIX_ISUB(x, row, col, val) DISP_MATRIX_ISET(x, row, col, DISP_MATRIX_IGET(x, row, col) - (val))

// Item MULtiply by Scalar
#define DISP_MATRIX_IMULS(x, row, col, val) DISP_MATRIX_ISET(x, row, col, DISP_MATRIX_IGET(x, row, col) * (val))
#define DISP_MATRIX_IADDS(x, row, col, val) DISP_MATRIX_ISET(x, row, col, DISP_MATRIX_IGET(x, row, col) + (val))

Matrix* generate_matrix(int in_rows, int in_cols, const double in_double_mat[][in_cols], const float in_float_mat[][in_cols], MatrixDataType in_dataType);

Matrix* generate_empty_matrix(int in_rows, int in_cols, MatrixDataType in_dataType);

Matrix* generate_matrix_f(int in_rows, int in_cols, const float in_float_mat[][in_cols]);
Matrix* generate_matrix_d(int in_rows, int in_cols, const double in_double_mat[][in_cols]);

void print_matrix(Matrix *in_matrix);
size_t get_matrix_data_size(Matrix *in_matrix);
long get_matrix_arr_data_size(Matrix **in_matrix_arr, int in_len);
bool is_matrix_null(Matrix *in_a);
int transpose(Matrix *in_a, Matrix *out_result);
int multiply(Matrix *in_a, Matrix *in_b, Matrix *out_result);
int multiply_transposed(Matrix *in_a, Matrix *in_b_t, Matrix *out_result);
int add(Matrix *in_out_a, Matrix *in_b);
int subtract(Matrix *in_out_a, Matrix *in_b);
int scalar_multiply(Matrix *in_out_a, double in_x);
int scalar_add(Matrix *in_out_a, double in_x);
int apply_d(Matrix *in_a, Matrix *out_result, double (*fn)(double), int in_layer_idx);
int apply_f(Matrix *in_a, Matrix *out_result, float (*fn)(float), int in_layer_idx);
int hadamard(Matrix *in_a, Matrix *in_b, Matrix *out_result);
int argmax(Matrix *in_a);
int reset_matrix(Matrix *in_out_a);
int delete_matrix(Matrix *in_out_a);
bool is_float_matrix(Matrix *in_a);

bool is_equal(Matrix *in_matrix, int in_rows, int in_cols, const double in_d_mat[in_rows][in_cols], const float in_f_mat[in_rows][in_cols]);

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

void on_neuron_activation_f(int in_layer_idx, int in_row, int in_col, float in_value);
void on_neuron_activation_d(int in_layer_idx, int in_row, int in_col, double in_value);

void open_activation_introspection(const char *in_output_data_filename);
void close_activation_introspection();

#endif /* MATRIX_H */
