#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include "activations.h"

// Activations and derivatives

double act_sigmoid_d(double num)
{
    return 1.0 / (1.0 + exp(-1.0*num));
}

#if 1
float act_sigmoid_f(float num)
{
    return 1.0f / (1.0f + expf(-1.0f*num));
}
#else
float act_sigmoid_f(float num)
{
    static long zero_count = 0;
    static long non_zero_count = 0;
    const double low_threshold = (double)1e-10;

    float tmp = 1.0f / (1.0f + expf(-1.0f*num));

    if (tmp < low_threshold && num > low_threshold)
    {
        zero_count++;
    }
    else
    {
        non_zero_count++;
    }

    if (zero_count > 1 && zero_count % 100000 == 0)
    {
        printf("act_sigmoid_f zero count: %d, non zero: %d\n", zero_count, non_zero_count);
    }

    return tmp;
}
#endif

#if 1
double act_sigmoid_der_d(double num)
{
    return exp(-1.0*num)/pow(1.0 + exp(-1.0*num),2.0);
}
#else
double act_sigmoid_der_d(double num)
{
    static long zero_count = 0;
    static long non_zero_count = 0;
    const double low_threshold = (double)1e-10;

    double tmp = exp(-1.0*num)/pow(1.0 + exp(-1.0*num),2.0);

    if (tmp < low_threshold && num > low_threshold)
    {
        zero_count++;
    }
    else
    {
        non_zero_count++;
    }

    if (zero_count > 1 && zero_count % 100000 == 0)
    {
        printf("act_sigmoid_der zero count: %d, non zero: %d\n", zero_count, non_zero_count);
    }

    return tmp;
}
#endif

#if 1
float act_sigmoid_der_f(float num)
{
    return expf(-1.0f*num)/powf(1.0f + expf(-1.0f*num), 2.0f);
}
#else
float act_sigmoid_der_f(float num)
{
    static long zero_count = 0;
    static long non_zero_count = 0;
    const float low_threshold = (float)1e-10;

    // expf returns 0 starting with expf(-1.0f*104), expf(-1.0f*103) is non zero
    // so basically expf for large numbers is zero
    // see: https://www.quora.com/What-are-the-pros-and-the-cons-of-the-sigmoid-activation-function-in-deep-learning
    // "The gradient for inputs that are far from the origin is near zero, so gradient-based learning is slow for saturated neurons using sigmoid"
    // but I guess for floats/fp32, this is actually / practically too small value, that backprop is not working

    float tmp = expf(-1.0f*num)/powf(1.0f + expf(-1.0f*num),2.0f);

    /*if(errno == ERANGE)
    {
        perror("errno == ERANGE");
    }*/

    // we check if result is zero but input is non zero
    if (tmp < low_threshold && num > low_threshold)
    {
        zero_count++;
    }
    else
    {
        non_zero_count++;
    }

    if (zero_count > 1 && zero_count % 100000 == 0)
    {
        printf("act_sigmoid_der_f zero count: %d, non zero: %d\n", zero_count, non_zero_count);
    }

    return tmp;
}
#endif

double act_relu_d(double num)
{
    return fmax(0.0, num);
}

float act_relu_f(float num)
{
    return fmaxf(0.0f, num);
}

double act_relu_der_d(double num)
{
    return num > 0.0 ? 1.0 : 0.0;
}

#if 1
float act_relu_der_f(float num)
{
    return num > 0.0f ? 1.0f : 0.0f;
}
#else
float act_relu_der_f(float num)
{
    static long zero_count = 0;
    static long non_zero_count = 0;
    const float low_threshold = (float)1e-10;

    float tmp = num > 0.0f ? 1.0f : 0.0f;

    // we check if result is zero but input is non zero
    if (tmp < low_threshold && num > low_threshold)
    {
        zero_count++;
    }
    else
    {
        non_zero_count++;
    }

    if (zero_count > 1 && zero_count % 100000 == 0)
    {
        printf("act_relu_der_f zero count: %d, non zero: %d\n", zero_count, non_zero_count);
    }

    return tmp;
}
#endif

Activation* create_sigmoid_activation()
{
    Activation *activation = (Activation *) malloc ( sizeof (Activation));

    activation->fn = &act_sigmoid_d;
    activation->fn_f = &act_sigmoid_f;

    activation->fn_der = &act_sigmoid_der_d;
    activation->fn_der_f = &act_sigmoid_der_f;

    activation->type = SIGMOID;

    return activation;
}

Activation* create_relu_activation()
{
    Activation *activation = (Activation *) malloc ( sizeof (Activation));
    activation->fn = &act_relu_d;
    activation->fn_der = &act_relu_der_d;

    activation->fn_f = &act_relu_f;
    activation->fn_der_f = &act_relu_der_f;

    activation->type = RELU;

    return activation;
}

int delete_activation(Activation *activation)
{
    free(activation);
    activation = NULL;

    return 0;
}