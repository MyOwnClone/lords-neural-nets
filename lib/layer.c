#include "layer.h"
#include "functions.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define RAND_INIT true

static int init_layer(Layer *layer);

Layer* create_layer(int layer_size, int input_size, Activation *activation, MatrixDataType dataType)
{
    Layer *layer = (Layer *) malloc (sizeof (Layer));

    layer->num_neurons = layer_size;
    layer->activation = activation;

    layer->weights = create_empty_matrix(layer_size, input_size, dataType);
    layer->bias = create_empty_matrix(layer_size, 1, dataType);
    layer->neurons = create_empty_matrix(layer_size, 1, dataType);
    layer->neurons_act = create_empty_matrix(layer_size, 1, dataType);

    init_layer(layer);

    return layer;
}

static int init_layer(Layer *layer)
{
    Matrix *weights = layer->weights;
    Matrix *bias = layer->bias;

    bool is_float = is_float_matrix(layer->weights);

    if (is_float)
    {
        srand(time(NULL));
        float range = sqrtf((float) 6/(float)(weights->rows + weights->cols));

        for (int row = 0; row < weights->rows; row++)
        {
            for (int col = 0; col < weights->cols; col++)
            {
                if (RAND_INIT)
                {
                    MATRIX_ISET(weights, row, col, (float) rand() / (float) (RAND_MAX * 2 * range - range));
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
                MATRIX_ISET(bias, row, 0, (float) rand() / (float)RAND_MAX);
            }
            else
            {
                MATRIX_ISET(bias, row, 0, 0);
            }
        }
    }
    else
    {
        srand(time(NULL));
        double range = sqrt((double) 6 / (weights->rows + weights->cols));

        for (int row = 0; row < weights->rows; row++)
        {
            for (int col = 0; col < weights->cols; col++)
            {
                if (RAND_INIT)
                {
                    MATRIX_ISET(weights, row, col, (double) rand() / RAND_MAX * 2 * range - range);
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
                MATRIX_ISET(bias, row, 0,(double) rand() / RAND_MAX);
            }
            else
            {
                MATRIX_ISET(bias, row, 0, 0);
            }
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
        apply(layer->neurons, layer->neurons_act, layer->activation->fn);
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