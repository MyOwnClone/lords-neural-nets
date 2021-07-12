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
    int layer_count = 2;

    Activation *act_sigmoid = create_sigmoid_activation();
    Network *xor_network = create_network(BINARY_OPERAND_COUNT, layer_count, xor_neurons_per_layer, act_sigmoid, D_FLOAT, -1);

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

    open_activation_introspection("xor_f.acts");

    Metrics monitor;
    Dataset *dataset = create_dataset(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs, labels, NULL, NULL);

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = CROSS_ENTROPY;
    training_options->epochs = XOR_EPOCH_COUNT;
    training_options->batch_size = XOR_BATCH_SIZE;
    training_options->learning_rate = XOR_LEARNING_RATE;
    training_options->momentum = XOR_MOMENTUM;
    training_options->regularization_lambda = XOR_REG_LAMBDA;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->log_each_nth_epoch = 1;

    write_network_introspection_params(xor_network);

    train(xor_network, dataset, &monitor, training_options, training_logging_options);

    close_activation_introspection();

    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}