#include "../lib/network.h"
#include "../lib/functions.h"
#include <stdlib.h>
#include <stdio.h>
#include "bench_utils.h"

void xor_float()
{
    int layers[] = {2,1};

    Activation *act_sigmoid = create_sigmoid_activation();
    Network *xor_network = create_network(2, 2, layers, act_sigmoid, D_FLOAT, -1);

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

    Metrics metrics;
    Dataset *dataset = create_dataset(4,2,1,4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = 20000;
    training_options->batch_size = 0;
    training_options->learning_rate = 1;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.0001;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
#ifdef DEBUG_MODE
    training_logging_options->LogEachNThEpoch = 1; // no logging
#else
    training_logging_options->LogEachNThEpoch = 1000; // no logging
#endif

    train_f(xor_network, dataset, &metrics, training_options, training_logging_options);

    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

void xor_double()
{
    int layers[] = {2,1};

    Activation *act_sigmoid = create_sigmoid_activation();
    Network *xor_network = create_network(2, 2, layers, act_sigmoid, D_DOUBLE, -1);

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

    Metrics metrics;
    Dataset *dataset = create_dataset(4,2,1,4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = 20000;
    training_options->batch_size = 0;
    training_options->learning_rate = 1;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.0001;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
#ifdef DEBUG_MODE
    training_logging_options->LogEachNThEpoch = 1; // no logging
#else
    training_logging_options->LogEachNThEpoch = 1000; // no logging
#endif

    train(xor_network, dataset, &metrics, training_options, training_logging_options);

    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

int main()
{
    int repeat_count = 1;

#ifdef DEBUG_MODE
    repeat_count = 1;
#endif

    double float_msecs = print_elapsed_time(xor_float, "xor float", repeat_count);
    double double_msecs = print_elapsed_time(xor_double, "xor double", repeat_count);

    printf("float over double speed-up factor: %fx\n", (double_msecs / float_msecs));

    /* mingw 64 mingw 64 gcc, windows 10, intel i7 cometlake:
        xor float: Average time elapsed over 10000 runs: 3.823900 ms
        xor double: Average time elapsed over 10000 runs: 3.393700 ms
        float over double speed-up factor: 0.887497x
     */

    /* macOS + Apple M1 + ARM64 + clang:

     xor float: Average time elapsed over 10000 runs: 1.343000 ms
     xor double: Average time elapsed over 10000 runs: 1.361000 ms
     float over double speed-up factor: 1.013403x

     (turning off -funsafe-math-optimizations does not change the times here)
     */

    /*
     HOWEVER!!! on M1, both float and double versions do not converge:

     train_f: Validation accuracy: 0.500
     train_f: Training accuracy: 0.500
     train_f: Training loss: nan
     xor float: Average time elapsed over 1 runs: 14.140000 ms

     train: Validation accuracy: 0.500
     train: Training accuracy: 0.500
     train: Training loss: 0.35071
     xor double: Average time elapsed over 1 runs: 12.397000 ms
     float over double speed-up factor: 0.876733x
     */

    /*
    HOWEVER 2: if I remove the -funsafe-math-optimizations parameter from cmake, double version converges even on M1, float version still NOPE
    train: Validation accuracy: 1.000
    train: Training accuracy: 1.000
    train: Training loss: 0.01189
    xor double: Average time elapsed over 1 runs: 12.924000 ms
    float over double speed-up factor: 0.873538x
     */
}