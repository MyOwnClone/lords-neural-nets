#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "bench_utils.h"
#include "../lib/network.h"

#define BINARY_OPERAND_COUNT 2
#define XOR_COMBINATION_COUNT (2 * 2)

static int xor_neurons_per_layer[] = {BINARY_OPERAND_COUNT, 1};

static const int XOR_EPOCH_COUNT = 20000;   // totally overkill
static const int XOR_BATCH_SIZE = 0;
static const float XOR_LEARNING_RATE = 1;
static const double XOR_MOMENTUM = 0.9;

static const double XOR_REG_LAMBDA = 0.0001;

#ifdef __APPLE__
    const int SEED = 1; // macOS / Unix uses different stdlib and windows SEED does not provide weights that converge, BUT WHY? :-(
#else
    const int SEED = -1;
#endif

static void delete_train_data(Activation *act_sigmoid, Network *xor_network, Dataset *dataset, TrainingOptions *training_options, TrainingLoggingOptions *training_logging_options)
{
    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

static void set_training_options(TrainingOptions *training_options)
{
    training_options->cost_type = CROSS_ENTROPY;
    training_options->epochs = XOR_EPOCH_COUNT;
    training_options->batch_size = XOR_BATCH_SIZE;
    training_options->learning_rate = XOR_LEARNING_RATE;
    training_options->momentum = XOR_MOMENTUM;
    training_options->regularization_lambda = XOR_REG_LAMBDA;
}

void xor_float()
{
    Activation *act_sigmoid_gen = generate_sigmoid_activation();
    Network *xor_network_gen = generate_network(BINARY_OPERAND_COUNT, 2, xor_neurons_per_layer, act_sigmoid_gen, D_FLOAT, SEED);

    Matrix **inputs_gen = (Matrix**) malloc (sizeof (Matrix*) * XOR_COMBINATION_COUNT);
    float inputs_mat[XOR_COMBINATION_COUNT][BINARY_OPERAND_COUNT][1] = {
            {{1}, {1}},
            {{1}, {0}},
            {{0}, {1}},
            {{0}, {0}}
    };

    Matrix **labels_gen = (Matrix**) malloc (sizeof (Matrix*) * XOR_COMBINATION_COUNT);
    float labels_mat[XOR_COMBINATION_COUNT][1][1] = {
            {{0}},
            {{1}},
            {{1}},
            {{0}}
    };

    for (int i = 0; i < XOR_COMBINATION_COUNT; i++)
    {
        inputs_gen[i] = generate_matrix_f(BINARY_OPERAND_COUNT, 1, inputs_mat[i]);
        labels_gen[i] = generate_matrix_f(1, 1, labels_mat[i]);
    }

    Metrics metrics;
    Dataset *dataset_gen = generate_dataset_structures(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs_gen, labels_gen, NULL, NULL);

    TrainingOptions *training_options = init_training_options();
    set_training_options(training_options);

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
#ifdef DEBUG_MODE
    training_logging_options->LogEachNThEpoch = 1;
#else
    training_logging_options->log_each_nth_epoch = NO_LOGGING;
#endif

    train(xor_network_gen, dataset_gen, &metrics, training_options, training_logging_options);

#ifdef DEBUG_MODE
    printf("acc %f, loss: %f\n", metrics.acc, metrics.loss);
#endif

    float expected_acc_threshold = 0.9;
    float expected_loss_threshold = 0.2;

    if (metrics.acc < expected_acc_threshold || metrics.loss >= expected_loss_threshold)
    {
        printf("assert failed with acc %f and loss %f\n", metrics.acc, metrics.loss);
    }

    assert(metrics.acc >= expected_acc_threshold && metrics.loss < expected_loss_threshold);

    delete_train_data(act_sigmoid_gen, xor_network_gen, dataset_gen, training_options, training_logging_options);

    free(inputs_gen);
    free(labels_gen);
}

void xor_double()
{
    Activation *act_sigmoid_gen = generate_sigmoid_activation();
    Network *xor_network_gen = generate_network(BINARY_OPERAND_COUNT, 2, xor_neurons_per_layer, act_sigmoid_gen, D_DOUBLE, SEED);

    Matrix **inputs_gen = (Matrix**) malloc (sizeof (Matrix*) * XOR_COMBINATION_COUNT);
    double inputs_mat[XOR_COMBINATION_COUNT][BINARY_OPERAND_COUNT][1] = {
            {{1}, {1}},
            {{1}, {0}},
            {{0}, {1}},
            {{0}, {0}}
    };

    Matrix **labels_gen = (Matrix**) malloc (sizeof (Matrix*) * XOR_COMBINATION_COUNT);
    double labels_mat[XOR_COMBINATION_COUNT][1][1] = {
            {{0}},
            {{1}},
            {{1}},
            {{0}}
    };

    for (int i = 0; i < XOR_COMBINATION_COUNT; i++)
    {
        inputs_gen[i] = generate_matrix_d(BINARY_OPERAND_COUNT, 1, inputs_mat[i]);
        labels_gen[i] = generate_matrix_d(1, 1, labels_mat[i]);
    }

    Metrics metrics;
    Dataset *dataset_gen = generate_dataset_structures(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs_gen, labels_gen, NULL, NULL);

    TrainingOptions *training_options = init_training_options();
    set_training_options(training_options);

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
#ifdef DEBUG_MODE
    training_logging_options->LogEachNThEpoch = 1;
#else
    training_logging_options->log_each_nth_epoch = NO_LOGGING;
#endif

    train(xor_network_gen, dataset_gen, &metrics, training_options, training_logging_options);

#ifdef DEBUG_MODE
    printf("acc %f, loss: %f\n", metrics.acc, metrics.loss);
#endif

    float expected_acc_threshold = 0.9;
    float expected_loss_threshold = 0.2;

    if (metrics.acc < expected_acc_threshold || metrics.loss >= expected_loss_threshold)
    {
        printf("assert failed with acc %f and loss %f\n", metrics.acc, metrics.loss);
    }

    assert(metrics.acc >= expected_acc_threshold && metrics.loss < expected_loss_threshold);

    delete_train_data(act_sigmoid_gen, xor_network_gen, dataset_gen, training_options, training_logging_options);
    free(inputs_gen);
    free(labels_gen);
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