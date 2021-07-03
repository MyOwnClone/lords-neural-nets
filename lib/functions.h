#include "matrix.h"

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// Activation functions

double act_sigmoid(double num);
double act_sigmoid_der(double num);
double act_relu(double num);
double act_relu_der(double num);
float act_sigmoid_f(float num);
float act_sigmoid_der_f(float num);
float act_relu_f(float num);
float act_relu_der_f(float num);

typedef enum
{
    SIGMOID,
    RELU,
    SIGMOID_F,
    RELU_F
} ActivationType;


typedef struct
{
    double (*fn)(double);
    double (*fn_der)(double);

    float (*fn_f)(float);
    float (*fn_der_f)(float);

    ActivationType type;
} Activation;

Activation* create_sigmoid_activation();
Activation* create_relu_activation();

int delete_activation(Activation *activation);

// Cost functions

typedef enum
{
    MEAN_SQUARED_ERROR,
    CROSS_ENTROPY
} CostType;


double cost_mse(Matrix *prediction, Matrix *target);
double cost_cross_entropy(Matrix *prediction, Matrix *target);

float cost_mse_f(Matrix *prediction, Matrix *target);
float cost_cross_entropy_f(Matrix *prediction, Matrix *target);

#endif /* FUNCTIONS_H */
