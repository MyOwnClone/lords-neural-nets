#ifndef NN_H
#define NN_H

#include "layer.h"
#include "matrix.h"
#include "activations.h"
#include "utils.h"
#include "metrics.h"

typedef struct 
{
    Layer **layers;
    int num_layers;
} Network;

#define TIME_SEED (-1)

Network *generate_network(int input_size, int num_layers, int neurons_per_layer[], Activation *activation, MatrixDataType dataType, int seed);
void print_network(Network *network);
long get_network_data_size(Network *network);
int delete_network(Network *network);
Matrix* predict(Network *network, Matrix *input);
double accuracy_d(Network *network, Matrix **inputs, Matrix **targets, int input_length);
float accuracy_f(Network *network, Matrix **inputs, Matrix **targets, int input_length);
int train(Network *network, Dataset *dataset, Metrics *metrics, TrainingOptions *training_options, TrainingLoggingOptions * training_logging_options);
void write_network_introspection_params(Network *network);
void on_new_epoch_start(int epoch_idx);

#endif