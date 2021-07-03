#include "../lib/utils.h"
#include "../lib/functions.h"
#include "../lib/network.h"
#include "bench_utils.h"
#include <stdio.h>
#include <float.h>

long EPOCH_COUNT = 10;

void mnist_double()
{
    int num_train = 60000;
    int num_test = 10000;

    char *train_inputs_fn = "./resources/mnist_train_vectors.csv";
    Matrix **train_inputs = load_csv(train_inputs_fn, num_train, 28*28, D_DOUBLE);
    normalize(train_inputs, num_train, 255);
    logger(INFO, __func__, "Created training dataset");

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels = load_csv(train_labels_fn, num_train, 1, D_DOUBLE);
    vectorize(train_labels, num_train, 10);
    logger(INFO, __func__, "Created training labels dataset");

    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs = load_csv(test_inputs_fn, num_test, 28*28, D_DOUBLE);
    normalize(test_inputs, num_test, 255);
    logger(INFO, __func__, "Created test dataset");

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels = load_csv(test_labels_fn, num_test, 1, D_DOUBLE);
    vectorize(test_labels, num_test, 10);
    logger(INFO, __func__, "Created test labels dataset");


    Dataset *dataset = create_dataset(num_train, 28*28, 10, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Monitor monitor[] = {acc, loss};

    int layers[] = {100,10};

    Activation *act_sigmoid = create_sigmoid_activation();
    CostType cost_type = CROSS_ENTROPY;
    Network *mnist_network = create_network(28*28, 2, layers, act_sigmoid, D_DOUBLE);

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = EPOCH_COUNT;
    training_options->batch_size = 10;
    training_options->learning_rate = 0.1;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.09;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->LogEachNThEpoch = 1;

    train(mnist_network, dataset, monitor, training_options, training_logging_options);

    delete_network(mnist_network);
    delete_dataset(dataset);
    delete_activation(act_sigmoid);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

void mnist_float()
{
    int num_train = 60000;
    int num_test = 10000;

    char *train_inputs_fn = "./resources/mnist_train_vectors.csv";
    Matrix **train_inputs = load_csv(train_inputs_fn, num_train, 28*28, D_FLOAT);

    normalize(train_inputs, num_train, 255);

    logger(INFO, __func__, "Created training dataset");

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels = load_csv(train_labels_fn, num_train, 1, D_FLOAT);

    vectorize(train_labels, num_train, 10);
    logger(INFO, __func__, "Created training labels dataset");

    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs = load_csv(test_inputs_fn, num_test, 28*28, D_FLOAT);

    normalize(test_inputs, num_test, 255);
    logger(INFO, __func__, "Created test dataset");

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels = load_csv(test_labels_fn, num_test, 1, D_FLOAT);

    vectorize(test_labels, num_test, 10);
    logger(INFO, __func__, "Created test labels dataset");

    Dataset *dataset = create_dataset(num_train, 28*28, 10, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Monitor monitor[] = {acc, loss};

    int layer_count = 2;
    int layers[] = {100,10};    // layer 0: 100 neurons, layer 1: 10 neurons

    Activation *act_sigmoid = create_sigmoid_activation();
    CostType cost_type = CROSS_ENTROPY;
    Network *mnist_network = create_network(28*28, layer_count, layers, act_sigmoid, D_FLOAT);

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = EPOCH_COUNT;
    training_options->batch_size = 10;
    training_options->learning_rate = 0.1;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.9;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->LogEachNThEpoch = 1;

    train_f(mnist_network, dataset, monitor, training_options, training_logging_options);

    delete_network(mnist_network);
    delete_dataset(dataset);
    delete_activation(act_sigmoid);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

int main()
{
    printf("The minimum value of float = %.10e\n", FLT_MIN);

    // FIXME: mnist has vanishing gradient problem with float backend
    double float_msecs = print_elapsed_time(mnist_float, "mnist float", 1);
    double double_msecs = print_elapsed_time(mnist_double, "mnist double", 1);

    printf("float over double speed-up factor: %fx\n", (double_msecs / float_msecs));

    // mingw 64 gcc, windows 10, intel i7 cometlake
    /*
     *  mnist float: Average time elapsed over 1 runs: 296_109.000000 ms :-( :-(     *
     *  mnist double: Average time elapsed over 1 runs: 188_733.000000 ms
     */

    /*
     * macOS + Apple M1 + ARM64 + clang: (but for float it does not converge :-( )
     *
     * mnist float: Average time elapsed over 1 runs: 232_673.735000 ms
     * mnist double: Average time elapsed over 1 runs: 256_663.657000 ms
     * float over double speed-up factor: 1.103105x
     */
}
