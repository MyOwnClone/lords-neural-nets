#include <math.h>
#include "matrix.h"

double cost_mse_d(Matrix *prediction, Matrix *target)
{
    if (prediction->cols != 1 || target->cols != 1 || prediction->rows != target->rows)
    {
        return -1;
    }

    double loss = 0.0;
    for (int i = 0; i < prediction->rows; i++)
    {
        loss += pow(DISP_MATRIX_IGET(prediction, i, 0) - DISP_MATRIX_IGET(target, i, 0), 2.0);
    }

    return loss / (2.0*prediction->rows);
}

float cost_mse_f(Matrix *prediction, Matrix *target)
{
    if (prediction->cols != 1 || target->cols != 1 || prediction->rows != target->rows)
    {
        return -1;
    }

    float loss = 0.0f;
    for (int i = 0; i < prediction->rows; i++)
    {
        loss += powf(DISP_MATRIX_IGET(prediction, i, 0) - DISP_MATRIX_IGET(target, i, 0), 2.0f);
    }

    return loss / (2.0f*(float)(prediction->rows));
}

double cost_cross_entropy_d(Matrix *prediction, Matrix *target)
{
    if (prediction->cols != 1 || target->cols != 1 || prediction->rows != target->rows)
    {
        return -1;
    }

    double loss = 0.0;
    for (int row = 0; row < prediction->rows; row++)
    {
        loss += -1.0 * (DISP_MATRIX_IGET(target, row, 0) * log(DISP_MATRIX_IGET(prediction, row, 0)) + (1.0 - DISP_MATRIX_IGET(target, row, 0)) * log(1.0 - DISP_MATRIX_IGET(prediction, row, 0)));
    }

    return loss;
}

float cost_cross_entropy_f(Matrix *prediction, Matrix *target)
{
    if (prediction->cols != 1 || target->cols != 1 || prediction->rows != target->rows)
    {
        return -1;
    }

    float loss = 0.0f;
    for (int row = 0; row < prediction->rows; row++)
    {
        loss += -1.0f * (DISP_MATRIX_IGET(target, row, 0) * logf(DISP_MATRIX_IGET(prediction, row, 0)) + (1.0f - DISP_MATRIX_IGET(target, row, 0)) * logf(1.0f - DISP_MATRIX_IGET(prediction, row, 0)));
    }

    return loss;
}