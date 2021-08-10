#include "../lib/network.h"
#include <stdlib.h>

#define BINARY_OPERAND_COUNT 2
#define XOR_COMBINATION_COUNT (2 * 2)

static int xor_neurons_per_layer[] = {BINARY_OPERAND_COUNT, 1};

static const int XOR_EPOCH_COUNT = 2000;   // totally overkill
static const int XOR_BATCH_SIZE = 0;
static const float XOR_LEARNING_RATE = 1;
static const double XOR_MOMENTUM = 0.9;

static const double XOR_REG_LAMBDA = 0.0001;

int main() {
    Activation *act_sigmoid_gen = generate_sigmoid_activation();
    Network *xor_network_gen = generate_network(BINARY_OPERAND_COUNT, 2, xor_neurons_per_layer, act_sigmoid_gen, D_DOUBLE, 1);

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

    Metrics monitor;
    Dataset *dataset_gen = generate_dataset_structures(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs_gen, labels_gen, NULL, NULL);

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = CROSS_ENTROPY;
    training_options->epochs = XOR_EPOCH_COUNT;
    training_options->batch_size = XOR_BATCH_SIZE;
    training_options->learning_rate = XOR_LEARNING_RATE;
    training_options->momentum = XOR_MOMENTUM;
    training_options->regularization_lambda = XOR_REG_LAMBDA;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->log_each_nth_epoch = 1;

    printf("network size: %ld B\n", get_network_data_size(xor_network_gen));

    train(xor_network_gen, dataset_gen, &monitor, training_options, training_logging_options);

    delete_network(xor_network_gen);
    delete_activation(act_sigmoid_gen);
    delete_dataset(dataset_gen);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}