#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "bench_utils.h"
#include "../lib/network.h"

static int xor_neurons_per_layer[] = {2, 1};

#define BINARY_OPERAND_COUNT 2
#define XOR_COMBINATION_COUNT (2 * 2)

// TODO: refactor

static const int XOR_EPOCH_COUNT = 20000;   // totally overkill
static const int XOR_BATCH_SIZE = 0;
static const float XOR_LEARNING_RATE = 1;
static const double XOR_MOMENTUM = 0.9;

static const double XOR_REG_LAMBDA = 0.0001;

static void delete_train_data(Activation *act_sigmoid, Network *xor_network, Dataset *dataset, TrainingOptions *training_options, TrainingLoggingOptions *training_logging_options)
{
    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

static void set_training_options(CostType cost_type, TrainingOptions *training_options)
{
    training_options->cost_type = cost_type;
    training_options->epochs = XOR_EPOCH_COUNT;
    training_options->batch_size = XOR_BATCH_SIZE;
    training_options->learning_rate = XOR_LEARNING_RATE;
    training_options->momentum = XOR_MOMENTUM;
    training_options->regularization_lambda = XOR_REG_LAMBDA;
}

void xor_float()
{
    Activation *act_sigmoid = create_sigmoid_activation();
    Network *xor_network = create_network(BINARY_OPERAND_COUNT, 2, xor_neurons_per_layer, act_sigmoid, D_FLOAT, -1);

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

    Metrics metrics;
    Dataset *dataset = create_dataset(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    set_training_options(cost_type, training_options);

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
#ifdef DEBUG_MODE
    training_logging_options->LogEachNThEpoch = 1;
#else
    training_logging_options->log_each_nth_epoch = NO_LOGGING;
#endif

    train_f(xor_network, dataset, &metrics, training_options, training_logging_options);

#ifdef DEBUG_MODE
    printf("acc %f, loss: %f\n", metrics.acc, metrics.loss);
#endif

    assert(metrics.acc >= 0.9 && metrics.loss < 0.1);

    delete_train_data(act_sigmoid, xor_network, dataset, training_options, training_logging_options);
}

void xor_double()
{
    Activation *act_sigmoid = create_sigmoid_activation();
    Network *xor_network = create_network(BINARY_OPERAND_COUNT, 2, xor_neurons_per_layer, act_sigmoid, D_DOUBLE, -1);

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

    Metrics metrics;
    Dataset *dataset = create_dataset(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    set_training_options(cost_type, training_options);

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
#ifdef DEBUG_MODE
    training_logging_options->LogEachNThEpoch = 1;
#else
    training_logging_options->log_each_nth_epoch = NO_LOGGING;
#endif

    train_d(xor_network, dataset, &metrics, training_options, training_logging_options);

#ifdef DEBUG_MODE
    printf("acc %f, loss: %f\n", metrics.acc, metrics.loss);
#endif

    assert(metrics.acc >= 0.9 && metrics.loss < 0.1);

    delete_train_data(act_sigmoid, xor_network, dataset, training_options, training_logging_options);
}

int main()
{
    int repeat_count = 10000;

#ifdef DEBUG_MODE
    repeat_count = 1;
#endif

    printf("sizeof(float) == %ld\n", sizeof(float) );
    printf("sizeof(double) == %ld\n", sizeof(double) );

    double float_msecs = print_elapsed_time(xor_float, "xor float", repeat_count);
    double double_msecs = print_elapsed_time(xor_double, "xor double", repeat_count);

    printf("float over double speed-up factor: %fx\n", (double_msecs / float_msecs));
}