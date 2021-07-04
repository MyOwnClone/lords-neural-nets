#include <functions.h>
#include <network.h>
#include <malloc.h>
#include "test.h"

const int XOR_EPOCH_COUNT = 1 * 1000;

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

    Metrics monitor;
    Dataset *dataset = create_dataset(4,2,1,4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = XOR_EPOCH_COUNT;
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

    Metrics monitor;
    Dataset *dataset = create_dataset(4,2,1,4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = XOR_EPOCH_COUNT;
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

long MNIST_EPOCH_COUNT = 10;

int test_train_mnist_double()
{
    int num_train = 6000;
    int num_test = 1000;

    char *train_inputs_fn = "./resources/mnist_train_vectors.csv";
    Matrix **train_inputs = load_csv(train_inputs_fn, num_train, 28*28, D_DOUBLE);
    normalize(train_inputs, num_train, 255);
    //logger(INFO, __func__, "Created training dataset");

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels = load_csv(train_labels_fn, num_train, 1, D_DOUBLE);
    vectorize(train_labels, num_train, 10);
    //logger(INFO, __func__, "Created training labels dataset");

    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs = load_csv(test_inputs_fn, num_test, 28*28, D_DOUBLE);
    normalize(test_inputs, num_test, 255);
    //logger(INFO, __func__, "Created test dataset");

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels = load_csv(test_labels_fn, num_test, 1, D_DOUBLE);
    vectorize(test_labels, num_test, 10);
    //logger(INFO, __func__, "Created test labels dataset");


    Dataset *dataset = create_dataset(num_train, 28*28, 10, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Metrics monitor;

    int layers[] = {100,10};

    Activation *act_sigmoid = create_sigmoid_activation();
    CostType cost_type = CROSS_ENTROPY;

    int seed = 1;
    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    Network *mnist_network = create_network(28 * 28, 2, layers, act_sigmoid, D_DOUBLE, seed);

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = MNIST_EPOCH_COUNT;
    training_options->batch_size = 10;
    training_options->learning_rate = 0.5;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.09;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->LogEachNThEpoch = -1;

    train(mnist_network, dataset, &monitor, training_options, training_logging_options);

    delete_network(mnist_network);
    delete_dataset(dataset);
    delete_activation(act_sigmoid);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

    //printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);

    return eval_test_result(__func__, res);
}

int test_train_mnist_float()
{
    int num_train = 6000;
    int num_test = 1000;

    char *train_inputs_fn = "./resources/mnist_train_vectors.csv";
    Matrix **train_inputs = load_csv(train_inputs_fn, num_train, 28*28, D_FLOAT);

    normalize(train_inputs, num_train, 255);

    //logger(INFO, __func__, "Created training dataset");

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels = load_csv(train_labels_fn, num_train, 1, D_FLOAT);

    vectorize(train_labels, num_train, 10);
    //logger(INFO, __func__, "Created training labels dataset");

    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs = load_csv(test_inputs_fn, num_test, 28*28, D_FLOAT);

    normalize(test_inputs, num_test, 255);
    //logger(INFO, __func__, "Created test dataset");

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels = load_csv(test_labels_fn, num_test, 1, D_FLOAT);

    vectorize(test_labels, num_test, 10);
    //logger(INFO, __func__, "Created test labels dataset");

    Dataset *dataset = create_dataset(num_train, 28*28, 10, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Metrics monitor;

    int layers[] = {100,10};

    Activation *act_sigmoid = create_sigmoid_activation();
    CostType cost_type = CROSS_ENTROPY;

    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    int seed = 1;
    Network *mnist_network = create_network(28 * 28, 2, layers, act_sigmoid, D_FLOAT, seed);

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = MNIST_EPOCH_COUNT;
    training_options->batch_size = 10;
    training_options->learning_rate = 0.5;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.9;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->LogEachNThEpoch = -1;

    train_f(mnist_network, dataset, &monitor, training_options, training_logging_options);

    delete_network(mnist_network);
    delete_dataset(dataset);
    delete_activation(act_sigmoid);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);

    int res = (monitor.loss < 0.2 && monitor.acc > 0.9) ? 0 : -1;

    //printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);

    return eval_test_result(__func__, res);
}