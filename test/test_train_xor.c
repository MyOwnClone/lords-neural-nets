#include <network.h>
#include <malloc.h>

#ifdef DEBUG_MODE
    #include <stdio.h>
#endif

#include "test.h"
#include "activations.h"
#include "utils.h"

const int XOR_EPOCH_COUNT = 1 * 1000;
const int XOR_BATCH_SIZE = 0;
const float XOR_LEARNING_RATE = 1.0f;
const float XOR_MOMENTUM = 0.9f;
const float XOR_REG_LAMBDA =  0.0001f;

#define BINARY_OPERAND_COUNT 2
#define XOR_COMBINATION_COUNT (2 * 2)

static int xor_neurons_per_layer[] = {BINARY_OPERAND_COUNT, 1};

static void set_xor_training_options(const CostType *cost_type, TrainingOptions *training_options)
{
    training_options->cost_type = (*cost_type);
    training_options->epochs = XOR_EPOCH_COUNT;
    training_options->batch_size = XOR_BATCH_SIZE;
    training_options->learning_rate = XOR_LEARNING_RATE;
    training_options->momentum = XOR_MOMENTUM;
    training_options->regularization_lambda = XOR_REG_LAMBDA;
}

int test_train_xor_double()
{
    Activation *act_sigmoid = create_sigmoid_activation();

    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    const int seed = 1;
    Network *xor_network = create_network(BINARY_OPERAND_COUNT, 2, xor_neurons_per_layer, act_sigmoid, D_DOUBLE, seed);

    Matrix **inputs = (Matrix**) malloc (sizeof (Matrix*) * XOR_COMBINATION_COUNT);
    double inputs_mat[XOR_COMBINATION_COUNT][BINARY_OPERAND_COUNT][1] = {
            {{1}, {1}},
            {{1}, {0}},
            {{0}, {1}},
            {{0}, {0}}
    };

    Matrix **labels = (Matrix**) malloc (sizeof (Matrix*) * XOR_COMBINATION_COUNT);
    double labels_mat[XOR_COMBINATION_COUNT][1][1] = {
            {{0}},
            {{1}},
            {{1}},
            {{0}}
    };

    for (int i = 0; i < XOR_COMBINATION_COUNT; i++)
    {
        inputs[i] = create_matrix_d(BINARY_OPERAND_COUNT, 1, inputs_mat[i]);
        labels[i] = create_matrix_d(1, 1, labels_mat[i]);
    }

    Metrics monitor;
    Dataset *dataset = create_dataset(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    set_xor_training_options(&cost_type, training_options);

    train(xor_network, dataset, &monitor, training_options, NULL);

    delete_train_test_data(act_sigmoid, xor_network, dataset, training_options, NULL);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

#ifdef DEBUG_MODE
    printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);
#endif

    return eval_test_result(__func__, res);
}

int test_train_xor_float()
{
    Activation *act_sigmoid = create_sigmoid_activation();

    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    const int seed = 1;
    Network *xor_network = create_network(BINARY_OPERAND_COUNT, 2, xor_neurons_per_layer, act_sigmoid, D_FLOAT, seed);

    Matrix **inputs = (Matrix**) malloc (sizeof (Matrix*) * XOR_COMBINATION_COUNT);
    float inputs_mat[XOR_COMBINATION_COUNT][BINARY_OPERAND_COUNT][1] = {
            {{1}, {1}},
            {{1}, {0}},
            {{0}, {1}},
            {{0}, {0}}
    };

    Matrix **labels = (Matrix**) malloc (sizeof (Matrix*) * XOR_COMBINATION_COUNT);
    float labels_mat[XOR_COMBINATION_COUNT][1][1] = {
            {{0}},
            {{1}},
            {{1}},
            {{0}}
    };

    for (int i = 0; i < XOR_COMBINATION_COUNT; i++)
    {
        inputs[i] = create_matrix_f(BINARY_OPERAND_COUNT, 1, inputs_mat[i]);
        labels[i] = create_matrix_f(1, 1, labels_mat[i]);
    }

    Metrics monitor;
    Dataset *dataset = create_dataset(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    set_xor_training_options(&cost_type, training_options);

    train(xor_network, dataset, &monitor, training_options, NULL);

    delete_train_test_data(act_sigmoid, xor_network, dataset, training_options, NULL);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

#if DEBUG_MODE
    printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);
#endif

    return eval_test_result(__func__, res);
}