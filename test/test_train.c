#include <functions.h>
#include <network.h>
#include <malloc.h>
#include <stdio.h>
#include "test.h"

const int epoch_count = 1 * 1000;

int test_train_xor_double()
{
    int layers[] = {2,1};

    Activation *act_sigmoid = create_sigmoid_activation();

    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    const int seed = 1;
    Network *xor_network = create_network(2, 2, layers, act_sigmoid, D_DOUBLE, seed);

    Matrix **inputs = (Matrix**) malloc (sizeof (Matrix*) * 4);
    double inputs_mat[4][2][1] = {
            {{1}, {1}},
            {{1}, {0}},
            {{0}, {1}},
            {{0}, {0}}
    };

    Matrix **labels = (Matrix**) malloc (sizeof (Matrix*) * 4);
    double labels_mat[4][1][1] = {
            {{0}},
            {{1}},
            {{1}},
            {{0}}
    };

    for (int i = 0; i < 4; i++)
    {
        inputs[i] = create_d_matrix(2, 1, inputs_mat[i]);
        labels[i] = create_d_matrix(1, 1, labels_mat[i]);
    }

    Monitor monitor;
    Dataset *dataset = create_dataset(4,2,1,4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = epoch_count;
    training_options->batch_size = 0;
    training_options->learning_rate = 1;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.0001;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->LogEachNThEpoch = -1; // no logging

    train(xor_network, dataset, &monitor, training_options, training_logging_options);

    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

    //printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);

    return eval_test_result(__func__, res);
}

int test_train_xor_float()
{
    int layers[] = {2,1};

    Activation *act_sigmoid = create_sigmoid_activation();

    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    const int seed = 1;
    Network *xor_network = create_network(2, 2, layers, act_sigmoid, D_FLOAT, seed);

    Matrix **inputs = (Matrix**) malloc (sizeof (Matrix*) * 4);
    float inputs_mat[4][2][1] = {
            {{1}, {1}},
            {{1}, {0}},
            {{0}, {1}},
            {{0}, {0}}
    };

    Matrix **labels = (Matrix**) malloc (sizeof (Matrix*) * 4);
    float labels_mat[4][1][1] = {
            {{0}},
            {{1}},
            {{1}},
            {{0}}
    };

    for (int i = 0; i < 4; i++)
    {
        inputs[i] = create_f_matrix(2, 1, inputs_mat[i]);
        labels[i] = create_f_matrix(1, 1, labels_mat[i]);
    }

    Monitor monitor;
    Dataset *dataset = create_dataset(4,2,1,4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = epoch_count;
    training_options->batch_size = 0;
    training_options->learning_rate = 1;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.0001;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->LogEachNThEpoch = -1;

    train_f(xor_network, dataset, &monitor, training_options, training_logging_options);

    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

    //printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);

    return eval_test_result(__func__, res);
}