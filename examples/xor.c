#include "../lib/network.h"
#include "../lib/functions.h"
#include <stdlib.h>
#include <stdio.h>

int main() {
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

    Metrics monitor[] = {acc, loss};
    Dataset *dataset = create_dataset(4,2,1,4, inputs, labels, NULL, NULL);

    CostType cost_type = CROSS_ENTROPY;

    TrainingOptions *training_options = init_training_options();
    training_options->cost_type = cost_type;
    training_options->epochs = 2000;
    training_options->batch_size = 0;
    training_options->learning_rate = 1;
    training_options->momentum = 0.9;
    training_options->regularization_lambda = 0.0001;

    TrainingLoggingOptions *training_logging_options = init_training_logging_options();
    training_logging_options->LogEachNThEpoch = 1;

    train(xor_network, dataset, monitor, training_options, training_logging_options);

    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);

    /*
    train: Epoch: 2000/2000
    train: Validation accuracy: 1.000
    train: Training accuracy: 1.000
    train: Training loss: 0.00654
     */

    /*
     * FIXME: on Intel, it sometimes converges to acc 1 and training loss ~0.00654, but sometimes both acc are around 0.5
     */
}