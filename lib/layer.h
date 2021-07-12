#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activations.h"
#include "activations.h"

typedef struct
{
    Matrix *weights;
    Matrix *bias;
    Matrix *neurons;    // immediate neuron's output, before activation
    Matrix *neurons_act;    // neuron's activation output
    Activation *activation;
    int num_neurons;
} Layer;

Layer *create_layer(int layer_size, int input_size, Activation *activation, MatrixDataType dataType, int seed);
int layer_compute(Layer *layer, Matrix *input, int layer_idx);
int delete_layer(Layer *layer);

#endif /* LAYER_H */
