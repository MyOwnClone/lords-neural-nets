#ifndef NNS_ACTIVATIONS_H
#define NNS_ACTIVATIONS_H

// Activation functions
double act_sigmoid_d(double num);
double act_sigmoid_der_d(double num);
double act_relu_d(double num);
double act_relu_der_d(double num);

float act_sigmoid_f(float num);
float act_sigmoid_der_f(float num);
float act_relu_f(float num);
float act_relu_der_f(float num);

typedef enum
{
    SIGMOID,
    RELU
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

#endif //NNS_ACTIVATIONS_H
