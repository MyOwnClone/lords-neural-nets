#include "../lib/network.h"

#ifdef DEBUG_MODE
    #include <stdio.h>
#endif

#include "test.h"
#include "utils.h"

const int MNIST_NUM_TRAIN = 6000;
const int MNIST_NUM_TEST = 1000;

const long MNIST_EPOCH_COUNT = 10;
const int MNIST_BATCH_SIZE = 10;
const float MNIST_LEARNING_RATE = 0.5f;
const float MNIST_MOMENTUM = 0.9f;
const float MNIST_FLOAT_REG_LAMBDA = 0.9f;
const float MNIST_DOUBLE_REG_LAMBDA = 0.09f;
const int MNIST_CHAR_RES = 28;
const int MNIST_CHAR_COUNT = 10;    // 10 possible digits 0-9

static int mnist_neurons_per_layer[] = {100, MNIST_CHAR_COUNT};

static void set_mnist_training_options(TrainingOptions *training_options, float reg_lambda)
{
    training_options->cost_type = 1;
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
    Matrix **train_inputs = load_csv_to_generated_matrix(train_inputs_fn, num_train, MNIST_CHAR_RES * MNIST_CHAR_RES, D_DOUBLE);
    normalize(train_inputs, num_train, 255);
#if DEBUG_MODE
    logger(INFO, __func__, "Created training dataset");
#endif

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels = load_csv_to_generated_matrix(train_labels_fn, num_train, 1, D_DOUBLE);
    vectorize(train_labels, num_train, MNIST_CHAR_COUNT);
#if DEBUG_MODE
    logger(INFO, __func__, "Created training labels dataset");
#endif

    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs = load_csv_to_generated_matrix(test_inputs_fn, num_test, MNIST_CHAR_RES * MNIST_CHAR_RES, D_DOUBLE);
    normalize(test_inputs, num_test, 255);
#if DEBUG_MODE
    logger(INFO, __func__, "Created test dataset");
#endif

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels = load_csv_to_generated_matrix(test_labels_fn, num_test, 1, D_DOUBLE);
    vectorize(test_labels, num_test, MNIST_CHAR_COUNT);
#if DEBUG_MODE
    logger(INFO, __func__, "Created test labels dataset");
#endif


    Dataset *dataset = generate_dataset_structures(num_train, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Metrics monitor;

    Activation *act_sigmoid = generate_sigmoid_activation();

    int seed = 1;
    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    Network *mnist_network = generate_network(MNIST_CHAR_RES * MNIST_CHAR_RES, 2, mnist_neurons_per_layer, act_sigmoid, D_DOUBLE, seed);

    TrainingOptions *training_options = init_training_options();

    float reg_lambda = MNIST_DOUBLE_REG_LAMBDA;  // yeah, this is different from float version. Why? Dunno :-(
    set_mnist_training_options(training_options, reg_lambda);

    train(mnist_network, dataset, &monitor, training_options, NULL);

    delete_train_test_data(act_sigmoid, mnist_network, dataset, training_options, NULL);

    int res = (monitor.loss < 0.1 && monitor.acc > 0.9) ? 0 : -1;

#if DEBUG_MODE
    printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);
#endif

    return eval_test_result(__func__, res);
}

int test_train_mnist_float()
{
    int num_train = MNIST_NUM_TRAIN;
    int num_test = MNIST_NUM_TEST;

    char *train_inputs_fn = "./resources/mnist_train_vectors.csv";
    Matrix **train_inputs = load_csv_to_generated_matrix(train_inputs_fn, num_train, MNIST_CHAR_RES * MNIST_CHAR_RES, D_FLOAT);

    normalize(train_inputs, num_train, 255);

#if DEBUG_MODE
    logger(INFO, __func__, "Created training dataset");
#endif

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels = load_csv_to_generated_matrix(train_labels_fn, num_train, 1, D_FLOAT);

    vectorize(train_labels, num_train, MNIST_CHAR_COUNT);
#if DEBUG_MODE
    logger(INFO, __func__, "Created training labels dataset");
#endif

    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs = load_csv_to_generated_matrix(test_inputs_fn, num_test, MNIST_CHAR_RES * MNIST_CHAR_RES, D_FLOAT);

    normalize(test_inputs, num_test, 255);
#if DEBUG_MODE
    logger(INFO, __func__, "Created test dataset");
#endif

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels = load_csv_to_generated_matrix(test_labels_fn, num_test, 1, D_FLOAT);

    vectorize(test_labels, num_test, MNIST_CHAR_COUNT);
#if DEBUG_MODE
    logger(INFO, __func__, "Created test labels dataset");
#endif

    Dataset *dataset = generate_dataset_structures(num_train, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Metrics monitor;

    Activation *act_sigmoid = generate_sigmoid_activation();

    // FIXME: when using other seed, test may fail, training may diverge and not reach tested condition
    int seed = 1;
    Network *mnist_network = generate_network(MNIST_CHAR_RES * MNIST_CHAR_RES, 2, mnist_neurons_per_layer, act_sigmoid, D_FLOAT, seed);

    TrainingOptions *training_options = init_training_options();
    set_mnist_training_options(training_options, MNIST_FLOAT_REG_LAMBDA);

    train(mnist_network, dataset, &monitor, training_options, NULL);

    delete_train_test_data(act_sigmoid, mnist_network, dataset, training_options, NULL);

    int res = (monitor.loss < 0.2 && monitor.acc > 0.9) ? 0 : -1;

#if DEBUG_MODE
    printf("loss: %f, acc: %f\n", monitor.loss, monitor.acc);
#endif

    return eval_test_result(__func__, res);
}