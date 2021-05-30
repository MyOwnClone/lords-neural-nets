#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

typedef enum
{
    D_FLOAT,
    D_DOUBLE
} MatrixDataType;

typedef struct 
{
    MatrixDataType type;
    int rows;
    int cols;
    union {
        double **matrix;    // just a hack to be able to use old name 'matrix' and also new name 'd_matrix' for the same double matrix
        float **f_matrix;
        double **d_matrix;
    };
} Matrix;

// TODO: benchmark performance float vs double

#define MATRIX_D(x) (x)->d_matrix
#define MATRIX_F(x) (x)->f_matrix

#define MATRIX_GET(x, col, row) (((x)->type == D_FLOAT) ? MATRIX_F(x)[col][row] : MATRIX_D(x)[col][row])

#define MATRIX_SET(x, col, row, val) if ((x)->type == D_FLOAT) \
                                (x)->f_matrix[col][row] = val; \
                                else \
                                (x)->d_matrix[col][row] = val;

#define MATRIX_ASSIGN(x, y, col, row) MATRIX_SET(x, col, row, MATRIX_GET(y, col, row))
#define MATRIX_ASSIGN_XY(x, y, col1, row1, col2, row2) MATRIX_SET(x, col1, row1, MATRIX_GET(y, col2, row2))
#define MATRIX_ADD(x, col, row, val) MATRIX_SET(x, col, row, MATRIX_GET(x, col, row) + val)

Matrix *create_matrix(int rows, int cols, const double mat[][cols]);
Matrix *create_matrix_float(int rows, int cols, const float mat[][cols]);
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
int hadamard(Matrix *a, Matrix *b, Matrix *result);
int argmax(Matrix *a);
int reset_matrix(Matrix *a);
int delete_matrix(Matrix *a);

#endif /* MATRIX_H */
