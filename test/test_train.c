#include <network.h>
#include <malloc.h>

#ifdef DEBUG
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
const long MNIST_EPOCH_COUNT = 10;

static int xor_layers[] = {2, 1};

const int MNIST_NUM_TRAIN = 6000;
const int MNIST_NUM_TEST = 1000;

static int mnist_layers[] = {100, 10};

const int MNIST_BATCH_SIZE = 10;
const float MNIST_LEARNING_RATE = 0.5f;
const float MNIST_MOMENTUM = 0.9f;
const float MNIST_FLOAT_REG_LAMBDA = 0.9f;
const float MNIST_DOUBLE_REG_LAMBDA = 0.09f;

//TODO: refactor

static void delete_test_data(Activation *act_sigmoid, Network *xor_network, Dataset *dataset, TrainingOptions *training_options, TrainingLoggingOptions *training_logging_options)
{
    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

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
    Network *xor_network = create_network(2, 2, xor_layers, act_sigmoid, D_DOUBLE, seed);

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
    Dataset *dataset = create_dataset(4, 4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    set_xor_training_options(&cost_type, training_options);

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->log_each_nth_epoch = -1; // no logging

    train(xor_network, dataset, &monitor, training_options, training_logging_options);

    delete_test_data(act_sigmoid, xor_network, dataset, training_options, training_logging_options);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

#if 0
    printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);
#endif

    return eval_test_result(__func__, res);
}

int test_train_xor_float()
{
    Activation *act_sigmoid = create_sigmoid_activation();

    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    const int seed = 1;
    Network *xor_network = create_network(2, 2, xor_layers, act_sigmoid, D_FLOAT, seed);

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
    Dataset *dataset = create_dataset(4, 4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    set_xor_training_options(&cost_type, training_options);

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->log_each_nth_epoch = -1;

    train_f(xor_network, dataset, &monitor, training_options, training_logging_options);

    delete_test_data(act_sigmoid, xor_network, dataset, training_options, training_logging_options);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

#if 0
    printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);
#endif

    return eval_test_result(__func__, res);
}

void set_mnist_training_options(CostType cost_type, TrainingOptions *training_options, float reg_lambda)
{
    training_options->cost_type = cost_type;
    training_options->epochs = MNIST_EPOCH_COUNT;
    training_options->batch_size = MNIST_BATCH_SIZE;
    training_options->learning_rate = MNIST_LEARNING_RATE;
    training_options->momentum = MNIST_MOMENTUM;
    training_options->regularization_lambda = reg_lambda;
}

int test_train_mnist_double()
{
    int num_train = MNIST_NUM_TRAIN;
    int num_test = MNIST_NUM_TEST;

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


    Dataset *dataset = create_dataset(num_train, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Metrics monitor;

    Activation *act_sigmoid = create_sigmoid_activation();
    CostType cost_type = CROSS_ENTROPY;

    int seed = 1;
    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    Network *mnist_network = create_network(28 * 28, 2, mnist_layers, act_sigmoid, D_DOUBLE, seed);

    TrainingOptions *training_options = init_training_options();

    float reg_lambda = MNIST_DOUBLE_REG_LAMBDA;  // yeah, this is different from float version. Why? Dunno :-(
    set_mnist_training_options(cost_type, training_options, reg_lambda);

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->log_each_nth_epoch = -1;

    train(mnist_network, dataset, &monitor, training_options, training_logging_options);

    delete_test_data(act_sigmoid, mnist_network, dataset, training_options, training_logging_options);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

#if 0
    printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);
#endif

    return eval_test_result(__func__, res);
}

int test_train_mnist_float()
{
    int num_train = MNIST_NUM_TRAIN;
    int num_test = MNIST_NUM_TEST;

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

    Dataset *dataset = create_dataset(num_train, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Metrics monitor;

    Activation *act_sigmoid = create_sigmoid_activation();
    CostType cost_type = CROSS_ENTROPY;

    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    int seed = 1;
    Network *mnist_network = create_network(28 * 28, 2, mnist_layers, act_sigmoid, D_FLOAT, seed);

    TrainingOptions *training_options = init_training_options();
    set_mnist_training_options(cost_type, training_options, MNIST_FLOAT_REG_LAMBDA);

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->log_each_nth_epoch = -1;

    train_f(mnist_network, dataset, &monitor, training_options, training_logging_options);

    delete_test_data(act_sigmoid, mnist_network, dataset, training_options, training_logging_options);

    int res = (monitor.loss < 0.2 && monitor.acc > 0.9) ? 0 : -1;

#if 0
    printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);
#endif

    return eval_test_result(__func__, res);
}