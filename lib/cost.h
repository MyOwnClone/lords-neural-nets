#ifndef NNS_COST_H
#define NNS_COST_H

#include "matrix.h"

// Cost functions

typedef enum
{
    MEAN_SQUARED_ERROR,
    CROSS_ENTROPY
} CostType;

double cost_mse_d(Matrix *prediction, Matrix *target);
double cost_cross_entropy_d(Matrix *prediction, Matrix *target);

float cost_mse_f(Matrix *prediction, Matrix *target);
float cost_cross_entropy_f(Matrix *prediction, Matrix *target);

#endif //NNS_COST_H
