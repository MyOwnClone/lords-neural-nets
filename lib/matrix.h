#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

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

// TODO: benchmark performance float vs double
// TODO: in memory dataset/loader

double** matrix_get_d(Matrix* x);
float** matrix_get_f(Matrix* x);

void matrix_set_d(Matrix* x, double **mat);
void matrix_set_f(Matrix* x, float **mat);

#define IS_FLOAT(x) ((x)->type == D_FLOAT)
//#define IS_FLOAT(x) false

// Item GET
#define MATRIX_IGET(x, row, col) (IS_FLOAT(x) ? (float)(matrix_get_f(x)[row][col]) : (double)(matrix_get_d(x)[row][col]))

// Item SET
#define MATRIX_ISET(x, row, col, val) if (IS_FLOAT(x)) { \
                                (x)->f_matrix[row][col] = val; \
                                } else {\
                                (x)->d_matrix[row][col] = val; };

void matrix_item_assign_direct(Matrix *x, Matrix *y, int row, int col);
void matrix_item_assign(Matrix *x, Matrix *y, int row1, int col1, int row2, int col2);

// Item APPLY FuNction
#define MATRIX_IAPPLY_FN(result, row, col, source, function) MATRIX_ISET(result, row, col, function(MATRIX_IGET(source, row, col)))

// Item ADD
#define MATRIX_IADD(x, row, col, val) MATRIX_ISET(x, row, col, MATRIX_IGET(x, row, col) + val)

// Item SUBtract
#define MATRIX_ISUB(x, row, col, val) MATRIX_ISET(x, row, col, MATRIX_IGET(x, row, col) - val)

// Item MULtiply by Scalar
#define MATRIX_IMULS(x, row, col, val) MATRIX_ISET(x, row, col, MATRIX_IGET(x, row, col) * val)
#define MATRIX_IADDS(x, row, col, val) MATRIX_ISET(x, row, col, MATRIX_IGET(x, row, col) + val)

Matrix* create_matrix(int rows, int cols, const double double_mat[][cols], const float float_mat[][cols], MatrixDataType dataType);

Matrix* create_empty_matrix(int rows, int cols, MatrixDataType dataType);

Matrix* create_f_matrix(int rows, int cols, const float float_mat[][cols]);
Matrix* create_d_matrix(int rows, int cols, const double double_mat[][cols]);
Matrix* create_empty_f_matrix(int rows, int cols);
Matrix* create_empty_d_matrix(int rows, int cols);

void convert_matrix_to_float(Matrix *matrix);
void print_matrix(Matrix *matrix);
bool is_null(Matrix *matrix);
int transpose(Matrix *a, Matrix *result);
int multiply(Matrix *a, Matrix *b, Matrix *result);
int multiply_transposed(Matrix *a, Matrix *b_t, Matrix *result);
int add(Matrix *a, Matrix *b);
int subtract(Matrix *a, Matrix *b);
int scalar_multiply(Matrix *matrix, double a);
int scalar_add(Matrix *matrix, double a);
int apply(Matrix *a, Matrix *result, double (*fn)(double));
int apply_f(Matrix *a, Matrix *result, float (*fn)(float));
int hadamard(Matrix *a, Matrix *b, Matrix *result);
int argmax(Matrix *a);
int reset_matrix(Matrix *a);
int delete_matrix(Matrix *a);
bool is_float_matrix(Matrix *a);

bool is_equal(Matrix *matrix, int rows, int cols, const double d_mat[rows][cols], const float f_mat[rows][cols]);

#define IS_EQUAL(mat, rows, cols, other_mat) ((is_float_matrix(mat) == true) && (is_equal(mat, rows, cols, NULL, other_mat) == true) || (is_equal(mat, rows, cols, other_mat, NULL)) == true)

#define MATRIX_SET(matrix, p_mat) if (is_float_matrix(matrix)) \
    matrix_set_f(matrix, p_mat);                                \
else                                                           \
    matrix_set_d(matrix, p_mat);

#define MATRIX_GET(matrix) (is_float_matrix(matrix) == true) ? matrix_get_f(matrix) : matrix_get_d(matrix)

#define APPLY(matrix, matrix_result, fn) if (is_float_matrix(matrix)) \
    apply_f(matrix, matrix_result, fn);                                \
else                                                                  \
    apply(matrix, matrix_result, fn);

#endif /* MATRIX_H */
