#include "../lib/utils.h"
#include "../lib/activations.h"
#include "../lib/network.h"

static const int TRAIN_SAMPLE_COUNT = 60;
static const int TEST_SAMPLE_COUNT = 10;
long MNIST_EPOCH_COUNT = 2000;
const int MNIST_CHAR_RES = 28;
const int MNIST_CHAR_COUNT = 10;    // 10 possible digits 0-9
const int MNIST_BATCH_SIZE = 10;
const float MNIST_LEARNING_RATE = 0.1f;
const float MNIST_MOMENTUM = 0.9f;
const float MNIST_FLOAT_REG_LAMBDA = 0.9f;
const float MNIST_DOUBLE_REG_LAMBDA = 0.09f;

int main()
{
    int num_train = TRAIN_SAMPLE_COUNT;
    int num_test = TEST_SAMPLE_COUNT;

    char *train_inputs_fn = "./resources/mnist_train_vectors.csv";
    Matrix **train_inputs = load_csv_to_generated_matrix(train_inputs_fn, num_train, MNIST_CHAR_RES * MNIST_CHAR_RES, D_DOUBLE);
    normalize(train_inputs, num_train, 255);
    logger(LOG_INFO, __func__, "Created training dataset");

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels = load_csv_to_generated_matrix(train_labels_fn, num_train, 1, D_DOUBLE);
    vectorize(train_labels, num_train, MNIST_CHAR_COUNT);
    logger(LOG_INFO, __func__, "Created training labels dataset");
 
    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs = load_csv_to_generated_matrix(test_inputs_fn, num_test, MNIST_CHAR_RES * MNIST_CHAR_RES, D_DOUBLE);
    normalize(test_inputs, num_test, 255);
    logger(LOG_INFO, __func__, "Created test dataset");

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels = load_csv_to_generated_matrix(test_labels_fn, num_test, 1, D_DOUBLE);
    vectorize(test_labels, num_test, MNIST_CHAR_COUNT);
    logger(LOG_INFO, __func__, "Created test labels dataset");

    open_activation_introspection("mnist_d.acts");

    Dataset *dataset = generate_dataset_structures(num_train, num_test, train_inputs, train_labels, test_inputs, test_labels);
    Metrics metrics;

    int neurons_per_layer[] = {100, MNIST_CHAR_COUNT};

    Activation *act_sigmoid = generate_sigmoid_activation();
    Network *mnist_network = generate_network(MNIST_CHAR_RES * MNIST_CHAR_RES, 2, neurons_per_layer, act_sigmoid, D_DOUBLE, TIME_SEED);

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = CROSS_ENTROPY;
    training_options->epochs = MNIST_EPOCH_COUNT;
    training_options->batch_size = MNIST_BATCH_SIZE;
    training_options->learning_rate = MNIST_LEARNING_RATE;
    training_options->momentum = MNIST_MOMENTUM;
    training_options->regularization_lambda = MNIST_DOUBLE_REG_LAMBDA;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->log_each_nth_epoch = 1000;

    write_network_introspection_params(mnist_network);

    train(mnist_network, dataset, &metrics, training_options, training_logging_options);

    close_activation_introspection();

    delete_network(mnist_network);
    delete_dataset(dataset);
    delete_activation(act_sigmoid);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}