#include "../lib/utils.h"
#include "../lib/activations.h"
#include "../lib/network.h"
#include "bench_utils.h"
#include <stdio.h>
#include <assert.h>

static const int TRAIN_SAMPLE_COUNT = 6000;
static const int TEST_SAMPLE_COUNT = 1000;
long MNIST_EPOCH_COUNT = 10;
const int MNIST_CHAR_RES = 28;
const int MNIST_CHAR_COUNT = 10;    // 10 possible digits 0-9
const int MNIST_BATCH_SIZE = 10;
const float MNIST_LEARNING_RATE = 0.5f;
const float MNIST_MOMENTUM = 0.9f;
const float MNIST_FLOAT_REG_LAMBDA = 0.9f;
const float MNIST_DOUBLE_REG_LAMBDA = 0.09f;

const int SEED = 1;

static int neurons_per_layer[] = {100, MNIST_CHAR_COUNT};

static void delete_training_data(Dataset *dataset, Activation *act_sigmoid, Network *mnist_network, TrainingOptions *training_options, TrainingLoggingOptions *training_logging_options)
{
    delete_network(mnist_network);
    delete_dataset(dataset);
    delete_activation(act_sigmoid);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

static void set_mnist_bench_training_options(TrainingOptions *training_options, float reg_lambda)
{
    training_options->cost_type = CROSS_ENTROPY;
    training_options->epochs = MNIST_EPOCH_COUNT;
    training_options->batch_size = MNIST_BATCH_SIZE;
    training_options->learning_rate = MNIST_LEARNING_RATE;
    training_options->momentum = MNIST_MOMENTUM;
    training_options->regularization_lambda = reg_lambda;
}

void mnist_double()
{
    int num_train = TRAIN_SAMPLE_COUNT;
    int num_test = TEST_SAMPLE_COUNT;

    char *train_inputs_fn = "./resources/mnist_train_vectors.csv";
    Matrix **train_inputs_gen = load_csv_to_generated_matrix(train_inputs_fn, num_train, MNIST_CHAR_RES * MNIST_CHAR_RES, D_DOUBLE);
    normalize(train_inputs_gen, num_train, 255);

#ifdef DEBUG_MODE
    logger(LOG_INFO, __func__, "Created training dataset");
#endif

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels_gen = load_csv_to_generated_matrix(train_labels_fn, num_train, 1, D_DOUBLE);
    vectorize(train_labels_gen, num_train, MNIST_CHAR_COUNT);

#ifdef DEBUG_MODE
    logger(LOG_INFO, __func__, "Created training labels dataset");
#endif

    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs_gen = load_csv_to_generated_matrix(test_inputs_fn, num_test, MNIST_CHAR_RES * MNIST_CHAR_RES, D_DOUBLE);
    normalize(test_inputs_gen, num_test, 255);

#ifdef DEBUG_MODE
    logger(LOG_INFO, __func__, "Created test dataset");
#endif

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels_gen = load_csv_to_generated_matrix(test_labels_fn, num_test, 1, D_DOUBLE);
    vectorize(test_labels_gen, num_test, MNIST_CHAR_COUNT);

#ifdef DEBUG_MODE
    logger(LOG_INFO, __func__, "Created test labels dataset");
#endif

    Dataset *dataset_gen = generate_dataset_structures(num_train, num_test, train_inputs_gen, train_labels_gen, test_inputs_gen, test_labels_gen);
    Metrics metrics;

    Activation *act_sigmoid_gen = generate_sigmoid_activation();
    Network *mnist_network_gen = generate_network(MNIST_CHAR_RES * MNIST_CHAR_RES, 2, neurons_per_layer, act_sigmoid_gen, D_DOUBLE, SEED);

    TrainingOptions *training_options = generate_training_options();
    set_mnist_bench_training_options(training_options, MNIST_DOUBLE_REG_LAMBDA);

    TrainingLoggingOptions *training_logging_options = generate_training_logging_options();
    
#ifdef DEBUG_MODE
    training_logging_options->log_each_nth_epoch = 1;
#else
    training_logging_options->log_each_nth_epoch = NO_LOGGING;
#endif

    train(mnist_network_gen, dataset_gen, &metrics, training_options, training_logging_options);

#ifdef DEBUG_MODE
    printf("acc %f, loss: %f\n", metrics.acc, metrics.loss);
#endif

    assert(metrics.acc >= 0.9 && metrics.loss < 0.1);

    delete_training_data(dataset_gen, act_sigmoid_gen, mnist_network_gen, training_options, training_logging_options);

    // these are freed in delete_dataset() called from delete_training_data()
    /*free(test_inputs_gen);
    free(test_labels_gen);
    free(train_inputs_gen);
    free(train_labels_gen);*/
}

void mnist_float()
{
    int num_train = TRAIN_SAMPLE_COUNT;
    int num_test = TEST_SAMPLE_COUNT;

    char *train_inputs_fn = "./resources/mnist_train_vectors.csv";
    Matrix **train_inputs_gen = load_csv_to_generated_matrix(train_inputs_fn, num_train, MNIST_CHAR_RES * MNIST_CHAR_RES, D_FLOAT);

    normalize(train_inputs_gen, num_train, 255);

#ifdef DEBUG_MODE
    logger(LOG_INFO, __func__, "Created training dataset");
#endif

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels_gen = load_csv_to_generated_matrix(train_labels_fn, num_train, 1, D_FLOAT);

    vectorize(train_labels_gen, num_train, MNIST_CHAR_COUNT);
#ifdef DEBUG_MODE
    logger(LOG_INFO, __func__, "Created training labels dataset");
#endif

    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs_gen = load_csv_to_generated_matrix(test_inputs_fn, num_test, 28 * 28, D_FLOAT);

    normalize(test_inputs_gen, num_test, 255);
#ifdef DEBUG_MODE
    logger(LOG_INFO, __func__, "Created test dataset");
#endif

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels_gen = load_csv_to_generated_matrix(test_labels_fn, num_test, 1, D_FLOAT);

    vectorize(test_labels_gen, num_test, MNIST_CHAR_COUNT);
#ifdef DEBUG_MODE
    logger(LOG_INFO, __func__, "Created test labels dataset");
#endif

    Dataset *dataset_gen = generate_dataset_structures(num_train, num_test, train_inputs_gen, train_labels_gen, test_inputs_gen, test_labels_gen);
    Metrics metrics;

    Activation *act_sigmoid_gen = generate_sigmoid_activation();
    Network *mnist_network_gen = generate_network(MNIST_CHAR_RES * MNIST_CHAR_RES, 2, neurons_per_layer, act_sigmoid_gen, D_FLOAT, SEED);

    TrainingOptions *training_options = generate_training_options();
    set_mnist_bench_training_options(training_options, MNIST_FLOAT_REG_LAMBDA);

    TrainingLoggingOptions *training_logging_options = generate_training_logging_options();
#ifdef DEBUG_MODE
    training_logging_options->log_each_nth_epoch = 1;
#else
    training_logging_options->log_each_nth_epoch = NO_LOGGING;
#endif

    train(mnist_network_gen, dataset_gen, &metrics, training_options, training_logging_options);

#ifdef DEBUG_MODE
    printf("acc %f, loss: %f\n", metrics.acc, metrics.loss);
#endif

    float expected_acc_threshold = 0.9f;
    float expected_loss_threshold = 0.2f;

    if (metrics.acc < expected_acc_threshold || metrics.loss >= expected_loss_threshold)
    {
        printf("assert failed with acc %f and loss %f\n", metrics.acc, metrics.loss);
    }

    assert(metrics.acc >= expected_acc_threshold && metrics.loss < expected_loss_threshold);

    delete_training_data(dataset_gen, act_sigmoid_gen, mnist_network_gen, training_options, training_logging_options);

    // these are freed in delete_dataset() called from delete_training_data()

    /*delete_matrix(*train_labels_gen);
    delete_matrix(*train_inputs_gen);
    delete_matrix(*test_inputs_gen);
    delete_matrix(*test_labels_gen);*/

}

int main()
{
    int repeat_count = 10;

#ifdef DEBUG_MODE
    repeat_count = 1;
#endif

    printf("sizeof(float) == %ld\n", sizeof(float) );
    printf("sizeof(double) == %ld\n", sizeof(double) );

    double float_msecs = print_elapsed_time(mnist_float, "mnist float", repeat_count);
    double double_msecs = print_elapsed_time(mnist_double, "mnist double", repeat_count);

    printf("float over double speed-up factor: %fx\n", (double_msecs / float_msecs));
}
