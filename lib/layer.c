#pragma clang diagnostic push
#pragma ide diagnostic ignored "Simplify"
#pragma ide diagnostic ignored "cert-msc50-cpp"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "layer.h"

#define RAND_INIT true

static int init_layer(Layer *layer, int seed);

Layer *create_layer(int layer_size, int input_size, Activation *activation, MatrixDataType dataType, int seed)
{
    Layer *layer = (Layer *) malloc (sizeof (Layer));

    layer->num_neurons = layer_size;
    layer->activation = activation;

    layer->weights = create_empty_matrix(layer_size, input_size, dataType);
    layer->bias = create_empty_matrix(layer_size, 1, dataType);
    layer->neurons = create_empty_matrix(layer_size, 1, dataType);
    layer->neurons_act = create_empty_matrix(layer_size, 1, dataType);

    init_layer(layer, seed);

    return layer;
}

static int init_layer(Layer *layer, int seed)
{
    Matrix *weights = layer->weights;
    Matrix *bias = layer->bias;

    bool is_float = is_float_matrix(layer->weights);

    if (seed == -1)
    {
        seed = time(NULL);
    }

    if (is_float)
    {
        srand(seed);
        double range = sqrt((double) 6/ (weights->rows + weights->cols));

        for (int row = 0; row < weights->rows; row++)
        {
            for (int col = 0; col < weights->cols; col++)
            {
                if (RAND_INIT)
                {
                    double rand_value = (double) rand() / RAND_MAX * 2 * range - range;

                    MATRIX_ISET(weights, row, col, (float)rand_value);
                }
                else
                {
                    MATRIX_ISET(weights, row, col, 0);
                }
            }
        }

        for (int row = 0; row < bias->rows; row++)
        {
            if (RAND_INIT)
            {
                MATRIX_ISET(bias, row, 0, (float)((double) rand() / (double)RAND_MAX));
            }
            else
            {
                MATRIX_ISET(bias, row, 0, 0);
            }
        }
    }
    else // double
    {
        srand(seed);
        double range = sqrt((double) 6 / (weights->rows + weights->cols));

        for (int row = 0; row < weights->rows; row++)
        {
            for (int col = 0; col < weights->cols; col++)
            {
                if (RAND_INIT)
                {
                    double rand_value = (double) rand() / RAND_MAX * 2 * range - range;
                    MATRIX_ISET(weights, row, col, rand_value);
                }
                else
                {
                    MATRIX_ISET(weights, row, col, 0);
                }
            }
        }

        for (int row = 0; row < bias->rows; row++)
        {
            MATRIX_ISET(bias, row, 0,(double) rand() / RAND_MAX);
        }
    }

    return 0;
}

int layer_compute(Layer *layer, Matrix *input)
{
    multiply(layer->weights, input, layer->neurons);
    add(layer->neurons, layer->bias);

    if (layer->neurons->type == D_FLOAT)
    {
        apply_f(layer->neurons, layer->neurons_act, layer->activation->fn_f);
    }
    else {
        apply_d(layer->neurons, layer->neurons_act, layer->activation->fn);
    }

    return 0;
}

int delete_layer(Layer *layer)
{
    if (layer == NULL) {
        return -1;
    }
    
    delete_matrix(layer->weights);
    delete_matrix(layer->bias);
    delete_matrix(layer->neurons);
    delete_matrix(layer->neurons_act);

    free(layer);
    layer = NULL;
    return 0;
}

#pragma clang diagnostic pop