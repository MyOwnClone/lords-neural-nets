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

    Activation *act_sigmoid_gen = generate_sigmoid_activation();
    Network *xor_network_gen = generate_network(BINARY_OPERAND_COUNT, layer_count, xor_neurons_per_layer, act_sigmoid_gen, D_FLOAT, 1);

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

    open_activation_introspection("xor_f.acts");

    Metrics monitor;
    Dataset *dataset_gen = generate_dataset_structures(XOR_COMBINATION_COUNT, XOR_COMBINATION_COUNT, inputs_gen, labels_gen, NULL, NULL);

    TrainingOptions *training_options_gen = generate_training_options();
    training_options_gen->cost_type = CROSS_ENTROPY;
    training_options_gen->epochs = XOR_EPOCH_COUNT;
    training_options_gen->batch_size = XOR_BATCH_SIZE;
    training_options_gen->learning_rate = XOR_LEARNING_RATE;
    training_options_gen->momentum = XOR_MOMENTUM;
    training_options_gen->regularization_lambda = XOR_REG_LAMBDA;

    TrainingLoggingOptions *training_logging_options_gen = generate_training_logging_options();
    training_logging_options_gen->log_each_nth_epoch = 1;

    write_network_introspection_params(xor_network_gen);

    printf("network size: %ld B\n", get_network_data_size(xor_network_gen));

    train(xor_network_gen, dataset_gen, &monitor, training_options_gen, training_logging_options_gen);

    close_activation_introspection();

    delete_network(xor_network_gen);
    delete_activation(act_sigmoid_gen);
    delete_dataset(dataset_gen);
    delete_training_options(training_options_gen);
    delete_training_logging_options(training_logging_options_gen);
}