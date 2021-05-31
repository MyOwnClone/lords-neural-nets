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

    float **f_matrix;

    union {
        double **matrix;    // just a hack to be able to use old name 'matrix' and also new name 'd_matrix' for the same double matrix
        double **d_matrix;
    };
} Matrix;

// TODO: benchmark performance float vs double
// TODO: in memory dataset/loader

double** matrix_d(Matrix* x);
float** MATRIX_F(Matrix* x);

// Item GET
#define MATRIX_IGET(x, col, row) (((x)->type == D_FLOAT) ? (float)(MATRIX_F(x)[col][row]) : (double)(matrix_d(x)[col][row]))

// Item SET
#define MATRIX_ISET(x, col, row, val) if ((x)->type == D_FLOAT) { \
                                (x)->f_matrix[col][row] = val; \
                                } else {\
                                (x)->d_matrix[col][row] = val; };

void matrix_item_assign_direct(Matrix *x, Matrix *y, int col, int row);
void matrix_item_assign(Matrix *x, Matrix *y, int col1, int row1, int col2, int row2);

// Item APPLY FuNction
#define MATRIX_IAPPLY_FN(result, col, row, source, function) MATRIX_ISET(result, col, row, function(MATRIX_IGET(source, col, row)))

// Item ADD
#define MATRIX_IADD(x, col, row, val) MATRIX_ISET(x, col, row, MATRIX_IGET(x, col, row) + val)

// Item SUBtract
#define MATRIX_ISUB(x, col, row, val) MATRIX_ISET(x, col, row, MATRIX_IGET(x, col, row) - val)

// Item MULtiply by Scalar
#define MATRIX_IMULS(x, col, row, val) MATRIX_ISET(x, col, row, MATRIX_IGET(x, col, row) * val)
#define MATRIX_IADDS(x, col, row, val) MATRIX_ISET(x, col, row, MATRIX_IGET(x, col, row) + val)

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
int apply_f(Matrix *a, Matrix *result, float (*fn)(float));
int hadamard(Matrix *a, Matrix *b, Matrix *result);
int argmax(Matrix *a);
int reset_matrix(Matrix *a);
int delete_matrix(Matrix *a);

#endif /* MATRIX_H */
