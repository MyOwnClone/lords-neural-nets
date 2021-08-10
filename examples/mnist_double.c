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
    Matrix **train_inputs_gen = load_csv_to_generated_matrix(train_inputs_fn, num_train, MNIST_CHAR_RES * MNIST_CHAR_RES, D_DOUBLE);
    normalize(train_inputs_gen, num_train, 255);
    logger(LOG_INFO, __func__, "Created training dataset");

    char *train_labels_fn = "./resources/mnist_train_labels.csv";
    Matrix **train_labels_gen = load_csv_to_generated_matrix(train_labels_fn, num_train, 1, D_DOUBLE);
    vectorize(train_labels_gen, num_train, MNIST_CHAR_COUNT);
    logger(LOG_INFO, __func__, "Created training labels dataset");
 
    char *test_inputs_fn = "./resources/mnist_test_vectors.csv";
    Matrix **test_inputs_gen = load_csv_to_generated_matrix(test_inputs_fn, num_test, MNIST_CHAR_RES * MNIST_CHAR_RES, D_DOUBLE);
    normalize(test_inputs_gen, num_test, 255);
    logger(LOG_INFO, __func__, "Created test dataset");

    char *test_labels_fn = "./resources/mnist_test_labels.csv";
    Matrix **test_labels_gen = load_csv_to_generated_matrix(test_labels_fn, num_test, 1, D_DOUBLE);
    vectorize(test_labels_gen, num_test, MNIST_CHAR_COUNT);
    logger(LOG_INFO, __func__, "Created test labels dataset");

    open_activation_introspection("mnist_d.acts");

    Dataset *dataset_gen = generate_dataset_structures(num_train, num_test, train_inputs_gen, train_labels_gen, test_inputs_gen, test_labels_gen);
    Metrics metrics;

    int neurons_per_layer[] = {100, MNIST_CHAR_COUNT};

    Activation *act_sigmoid_gen = generate_sigmoid_activation();
    Network *mnist_network_gen = generate_network(MNIST_CHAR_RES * MNIST_CHAR_RES, 2, neurons_per_layer, act_sigmoid_gen, D_DOUBLE, TIME_SEED);

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = CROSS_ENTROPY;
    training_options->epochs = MNIST_EPOCH_COUNT;
    training_options->batch_size = MNIST_BATCH_SIZE;
    training_options->learning_rate = MNIST_LEARNING_RATE;
    training_options->momentum = MNIST_MOMENTUM;
    training_options->regularization_lambda = MNIST_DOUBLE_REG_LAMBDA;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->log_each_nth_epoch = 1000;

    write_network_introspection_params(mnist_network_gen);

    long network_data_size = get_network_data_size(mnist_network_gen);

    long dataset_size = get_matrix_arr_data_size(train_inputs_gen, num_train);
    dataset_size += get_matrix_arr_data_size(train_labels_gen, num_train);
    dataset_size += get_matrix_arr_data_size(test_inputs_gen, num_test);
    dataset_size += get_matrix_arr_data_size(test_labels_gen, num_test);

    printf("network size: %f MB, dataset size: %f MB\n", convert_bytes_to_Mbytes(network_data_size), convert_bytes_to_Mbytes(dataset_size));

    train(mnist_network_gen, dataset_gen, &metrics, training_options, training_logging_options);

    close_activation_introspection();

    delete_network(mnist_network_gen);
    delete_dataset(dataset_gen);
    delete_activation(act_sigmoid_gen);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}