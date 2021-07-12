#include "layer.h"
#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

/*
 * seed == -1, function internally uses time(NULL) for seed, seed != -1 -> used as is
 */
Network *create_network(int input_size, int num_layers, int neurons_per_layer[], Activation *activation, MatrixDataType dataType, int seed)
{
    Network *network = (Network *) malloc(sizeof(Network));
    network->num_layers = num_layers;

    network->layers = (Layer **) malloc(sizeof(Layer *) * num_layers);
    int prev_layer_size = input_size;
    for (int i = 0; i < num_layers; i++)
    {
        network->layers[i] = create_layer(neurons_per_layer[i], prev_layer_size, activation, dataType, seed);
        prev_layer_size = neurons_per_layer[i];
    }

    return network;
}

void print_network(Network *network)
{
    for (int i = 0; i < network->num_layers; i++)
    {
        printf("Layer %d\n", i);
        print_matrix(network->layers[i]->weights);
    }
}

int delete_network(Network *network)
{
    for (int i = 0; i < network->num_layers; i++)
    {
        delete_layer(network->layers[i]);
    }

    free(network->layers);
    free(network);
    network = NULL;

    return 0;
}

Matrix *predict(Network *network, Matrix *input)
{
    Matrix *layer_input = input;

    layer_input->type = input->type;

    for (int layer_idx = 0; layer_idx < network->num_layers; layer_idx++)
    {
        Layer *layer = network->layers[layer_idx];
        int res = layer_compute(layer, layer_input, layer_idx);
        if (res < 0)
        {
            logger(LOG_EXCEPTION, __func__, "Exception during prediction");
        }
        layer_input = layer->neurons_act;
    }

    return layer_input;
}

float accuracy_f(Network *network, Matrix **inputs, Matrix **targets, int input_length)
{
    int correct = 0;

    for (int i = 0; i < input_length; i++)
    {
        Matrix *prediction = predict(network, inputs[i]);
        if (targets[i]->cols == 1 && targets[i]->rows == 1)
        {
            float pred_value = DISP_MATRIX_IGET(prediction, 0, 0) < 0.5 ? 0 : 1;

            if (pred_value == DISP_MATRIX_IGET(targets[i], 0, 0))
            {
                correct++;
            }
        }
        else
        {
            int predicted_class = argmax(prediction);
            int real_class = argmax(targets[i]);
            if (predicted_class == real_class) correct++;
        }
    }

    return (float) correct / (float )input_length;
}

double accuracy_d(Network *network, Matrix **inputs, Matrix **targets, int input_length)
{
    int correct = 0;

    for (int i = 0; i < input_length; i++)
    {
        Matrix *prediction = predict(network, inputs[i]);
        if (targets[i]->cols == 1 && targets[i]->rows == 1)
        {
            double pred_value = DISP_MATRIX_IGET(prediction, 0, 0) < 0.5 ? 0 : 1;

            if (pred_value == DISP_MATRIX_IGET(targets[i], 0, 0))
            {
                correct++;
            }
        }
        else
        {
            int predicted_class = argmax(prediction);
            int real_class = argmax(targets[i]);
            if (predicted_class == real_class) correct++;
        }
    }

    return (double) correct / input_length;
}

// exp in a name means this is an expression - r-value
#define ACCURACY_EXP(net, inputs, targets, len) is_float_matrix(*(inputs)) ? accuracy_f(net, inputs, targets, len) : accuracy_d(net, inputs, targets, len)

static int init_training(
        Network *network,
        Matrix **deltas,
        Matrix **temp_deltas,
        Matrix **delta_weights,
        Matrix **temp_delta_weights,
        Matrix **transposed_weights,
        Matrix **delta_bias,
        Matrix **momentums)
{
    for (int i = 0; i < network->num_layers; i++)
    {
        MatrixDataType type = network->layers[i]->weights->type;

        int rows = network->layers[i]->weights->rows;
        int cols = network->layers[i]->weights->cols;

        delta_weights[i] = create_empty_matrix(rows, cols, type);
        momentums[i] = create_empty_matrix(rows, cols, type);
        temp_delta_weights[i] = create_empty_matrix(rows, cols, type);

        if (i > 0)
        {
            transposed_weights[i - 1] = create_empty_matrix(cols, rows, type);
            transpose(network->layers[i]->weights, transposed_weights[i - 1]);
        }

        int bias_rows = network->layers[i]->bias->rows;
        int bias_cols = network->layers[i]->bias->cols;

        delta_bias[i] = create_empty_matrix(bias_rows, bias_cols, type);
        deltas[i] = create_empty_matrix(bias_rows, bias_cols, type);

        if (i > 0)
        {
            temp_deltas[i - 1] = create_empty_matrix(cols, bias_cols, type);
        }
    }

    return 0;
}

static int backpropagate(
        Network *network,
        Matrix *input,
        Matrix **deltas,
        Matrix **temp_deltas,
        Matrix **delta_weights,
        Matrix **temp_delta_weights,
        Matrix **transposed_weights,
        Matrix **delta_bias)
{
    int res;
    Layer *layer;
    Matrix *prev_act;

    for (int layer_idx = network->num_layers - 2; layer_idx >= 0; layer_idx--)
    {
        layer = network->layers[layer_idx];

        if (layer_idx == 0)
        {
            prev_act = input;
        }
        else
        {
            prev_act = network->layers[layer_idx - 1]->neurons_act;
        }

        // Compute new delta
        res = 0;

        if (layer->neurons->type == D_FLOAT)
        {
            res += apply_f(layer->neurons, NULL, layer->activation->fn_der_f, layer_idx);
        }
        else
        {
            res += apply_d(layer->neurons, NULL, layer->activation->fn_der, layer_idx);
        }

        res += multiply(transposed_weights[layer_idx], deltas[layer_idx + 1], temp_deltas[layer_idx]); // Transposed weights array is 1 shorter than matrix length
        res += hadamard(temp_deltas[layer_idx], layer->neurons, deltas[layer_idx]);
        if (res < 0)
        {
            logger(LOG_EXCEPTION, __func__, "Exception during delta calculation");
            return res;
        }

        // Compute delta weights
        res = 0;
        res += multiply_transposed(deltas[layer_idx], prev_act, temp_delta_weights[layer_idx]);
        res += add(delta_weights[layer_idx], temp_delta_weights[layer_idx]);
        if (res < 0)
        {
            logger(LOG_EXCEPTION, __func__, "Exception during delta weights calculation");
            return res;
        }

        // Compute delta bias
        res = add(delta_bias[layer_idx], deltas[layer_idx]);
        if (res < 0)
        {
            logger(LOG_EXCEPTION, __func__, "Exception during delta bias calculation");
            return res;
        }
    }

    return 0;
}

static void cleanup(
        int network_length,
        Matrix **deltas,
        Matrix **temp_deltas,
        Matrix **delta_weights,
        Matrix **temp_delta_weights,
        Matrix **transposed_weights,
        Matrix **delta_bias)
{
    // Cleanup
    for (int i = 0; i < network_length; i++)
    {
        delete_matrix(deltas[i]);
        delete_matrix(delta_weights[i]);
        delete_matrix(temp_delta_weights[i]);
        delete_matrix(delta_bias[i]);

        if (i != network_length - 1)
        {
            delete_matrix(transposed_weights[i]);
            delete_matrix(temp_deltas[i]);
        }
    }
    free(deltas);
    free(temp_deltas);
    free(delta_weights);
    free(temp_delta_weights);
    free(transposed_weights);
    free(delta_bias);
}

static int reset(
        Network *network,
        Matrix **deltas,
        Matrix **temp_deltas,
        Matrix **delta_weights,
        Matrix **temp_delta_weights,
        Matrix **transposed_weights,
        Matrix **delta_bias,
        Matrix **momentums)
{
    int res = 0;
    for (int i = 0; i < network->num_layers; i++)
    {
        res += reset_matrix(deltas[i]);
        res += reset_matrix(delta_weights[i]);
        res += reset_matrix(temp_delta_weights[i]);
        res += reset_matrix(delta_bias[i]);
        res += reset_matrix(momentums[i]);

        if (i != network->num_layers - 1)
        {
            res += reset_matrix(transposed_weights[i]);
            res += reset_matrix(temp_deltas[i]);
        }

        if (i != 0)
        {
            res += transpose(network->layers[i]->weights, transposed_weights[i - 1]);
        }

        if (res < 0)
        {
            logger(LOG_EXCEPTION, __func__, "Exception during training temp objects reset");
            return res;
        }
    }

    return 0;
}

static int
get_initial_delta(CostType cost_type, Activation *activation, Matrix *prediction, Matrix *target, Matrix *layer_output,
                  Matrix *delta)
{
    int res = 0;
    if (cost_type == MEAN_SQUARED_ERROR)
    {

        res = 0;
        res += subtract(prediction, target);
        res += apply_d(layer_output, NULL, activation->fn_der, 0); // TODO: distinction between float and double is missing
        res += hadamard(prediction, layer_output, delta);
        if (res < 0)
        {
            logger(LOG_EXCEPTION, __func__, "Exception during output delta calculation");
            return res;
        }


    } else if (cost_type == CROSS_ENTROPY)
    {
        if (activation->type == SIGMOID)
        {
            res = 0;
            res += reset_matrix(delta);
            res += add(delta, prediction);
            res += subtract(delta, target);


            if (res < 0)
            {
                logger(LOG_EXCEPTION, __func__, "Exception during output delta calculation");
                return res;
            }
        }
    }

    return res;

}

static double get_loss(CostType cost_type, Matrix *prediction, Matrix *target)
{
    if (prediction->type != target->type)
    {
        logger(LOG_EXCEPTION, __func__, "Prediction and target matrices are not the same type!!!");

        return INT_MAX;
    }

    if (cost_type == MEAN_SQUARED_ERROR)
    {
        return cost_mse_d(prediction, target);
    }
    else if (cost_type == CROSS_ENTROPY)
    {
        return cost_cross_entropy_d(prediction, target);
    }
    else
    {
        logger(LOG_EXCEPTION, __func__, "Unknown cost_type!");

        return INT_MAX;
    }
}

static float get_loss_f(CostType cost_type, Matrix *prediction, Matrix *target)
{
    if (prediction->type != target->type)
    {
        logger(LOG_EXCEPTION, __func__, "Prediction and target matrices are not the same type!!!");

        return (float) INT_MAX;
    }

    if (cost_type == MEAN_SQUARED_ERROR)
    {
        return cost_mse_f(prediction, target);
    }
    else if (cost_type == CROSS_ENTROPY)
    {
        return cost_cross_entropy_f(prediction, target);
    }
    else
    {
        logger(LOG_EXCEPTION, __func__, "Unknown cost_type!");

#ifdef WIN32
        return (float) INT_MAX;
#else
        return (float)  INT_FAST32_MAX;
#endif
    }
}

// exp in a name means this is an expression - r-value
#define GET_LOSS_EXP(cost_type, model_prediction, target) (is_float_matrix(model_prediction)) ? get_loss_f(cost_type, model_prediction, target) : get_loss(cost_type, model_prediction, target)

int train(Network *network, Dataset *dataset, Metrics *metrics, TrainingOptions *training_options, TrainingLoggingOptions * training_logging_options)
{
    // Allocate all the memory
    Matrix **delta_weights = (Matrix **) malloc(sizeof(Matrix *) * network->num_layers);
    Matrix **temp_delta_weights = (Matrix **) malloc(sizeof(Matrix *) * network->num_layers);

    Matrix **momentums = (Matrix **) malloc(sizeof(Matrix *) * network->num_layers);

    Matrix **delta_bias = (Matrix **) malloc(sizeof(Matrix *) * network->num_layers);

    Matrix **deltas = (Matrix **) malloc(sizeof(Matrix *) * network->num_layers);

    Matrix **temp_deltas = (Matrix **) malloc(sizeof(Matrix *) * network->num_layers - 1);
    Matrix **transposed_weights = (Matrix **) malloc(sizeof(Matrix *) * network->num_layers - 1);

    CostType cost_type = training_options->cost_type;
    int batch_size = training_options->batch_size;
    int epochs = training_options->epochs;
    double learning_rate = training_options->learning_rate;
    double momentum = training_options->momentum;
    double reg_lambda = training_options->regularization_lambda;

    init_training(
            network,
            deltas,
            temp_deltas,
            delta_weights,
            temp_delta_weights,
            transposed_weights,
            delta_bias,
            momentums
    );

    Matrix *prediction;
    Matrix *target;

    int L = network->num_layers - 1;
    Layer *last_layer = network->layers[L];

    int res = 0;
    int epoch = 0;

    double epoch_accuracy;
    double epoch_loss;

    if (batch_size == 0)
    {
        batch_size = dataset->train_size;
    }

    while (epoch < epochs)
    {
        if (training_logging_options != NULL && (training_logging_options->log_each_nth_epoch > 0 && (epoch + 1) % training_logging_options->log_each_nth_epoch == 0))
        {
            char buffer[10 + (epoch % 10) + (epochs % 10)];
            sprintf(buffer, "Epoch: %d/%d", epoch + 1, epochs);
            logger(LOG_INFO, __func__, buffer);
        }

        epoch_loss = 0;

        int i = 0;

        double final_epoch_loss = epoch_loss;

        while (i < dataset->train_size)
        {
            int batch_start = i;
            int batch_end = batch_start + batch_size;

            if (batch_end > dataset->train_size)
            {
                batch_end = dataset->train_size;
            }

            for (int j = batch_start; j < batch_end; j++)
            {
                prediction = predict(network, dataset->train_inputs[j]);
                target = dataset->train_labels[j];

                if (prediction == NULL)
                {
                    logger(LOG_EXCEPTION, __func__, "Exception during prediction");
                    return -1;
                }

                epoch_loss += GET_LOSS_EXP(cost_type, prediction, target);

                // Calculate initial delta
                get_initial_delta(cost_type, last_layer->activation, prediction, target, last_layer->neurons, deltas[L]);

                // Update delta weights
                res = 0;
                res += multiply_transposed(deltas[L], network->layers[L - 1]->neurons_act, temp_delta_weights[L]);
                res += add(delta_weights[L], temp_delta_weights[L]);
                if (res < 0)
                {
                    logger(LOG_EXCEPTION, __func__, "Exception during output delta weights calculation");
                    return res;
                }

                // Update delta biases
                res = add(delta_bias[L], deltas[L]);
                if (res < 0)
                {
                    logger(LOG_EXCEPTION, __func__, "Exception during output delta bias calculation");
                    return res;
                }

                backpropagate(
                        network,
                        dataset->train_inputs[j],
                        deltas,
                        temp_deltas,
                        delta_weights,
                        temp_delta_weights,
                        transposed_weights,
                        delta_bias
                );
            }

            // Adjust weights
            double eta = -1 * (learning_rate / batch_size);
            for (int j = 0; j < network->num_layers; j++)
            {
                // Get momentum
                scalar_multiply(momentums[j], momentum);

                // Get weights adjustment
                scalar_multiply(delta_weights[j], eta);

                // Add momentum
                add(delta_weights[j], momentums[j]);

                // L2 Regularization
                scalar_multiply(network->layers[j]->weights, 1 - ((learning_rate * reg_lambda) / dataset->train_size));

                // TODO: call OnWeightUpdate(layer, index, delta)

                // Set new weights
                add(network->layers[j]->weights, delta_weights[j]);

                // Set bias
                scalar_multiply(delta_bias[j], eta);

                // TODO: call OnBiasUpdate(layer, index, delta)
                add(network->layers[j]->bias, delta_bias[j]);
            }

            reset(
                    network,
                    deltas,
                    temp_deltas,
                    delta_weights,
                    temp_delta_weights,
                    transposed_weights,
                    delta_bias,
                    momentums
            );

            i += batch_size;
        }

        if (training_logging_options != NULL && (training_logging_options->log_each_nth_epoch > 0 && (epoch + 1) % training_logging_options->log_each_nth_epoch == 0))
        {
            if (training_logging_options != NULL && training_logging_options->log_accuracy)
            {
                epoch_accuracy = ACCURACY_EXP(network, dataset->val_inputs, dataset->val_labels, dataset->val_size);
                char acc_buffer[27];
                sprintf(acc_buffer, "Validation accuracy: %.3f", epoch_accuracy);
                logger(LOG_INFO, __func__, acc_buffer);

                epoch_accuracy = ACCURACY_EXP(network, dataset->train_inputs, dataset->train_labels,
                                              dataset->train_size);
                char acc_train_buffer[27];
                sprintf(acc_train_buffer, "Training accuracy: %.3f", epoch_accuracy);
                logger(LOG_INFO, __func__, acc_train_buffer);
            }

            if (training_logging_options != NULL && training_logging_options->log_loss)
            {
                epoch_loss = (double) epoch_loss / dataset->train_size;
                char loss_buffer[23];
                sprintf(loss_buffer, "Training loss: %.5f", epoch_loss);
                logger(LOG_INFO, __func__, loss_buffer);
            }
        }
        else
        {
            epoch_accuracy = (float) ACCURACY_EXP(network, dataset->val_inputs, dataset->val_labels, dataset->val_size);
            final_epoch_loss = (float) epoch_loss / (float) dataset->train_size;
        }

        metrics->acc = epoch_accuracy;
        metrics->loss = final_epoch_loss;

        epoch++;
    }

    cleanup(
            network->num_layers,
            deltas,
            temp_deltas,
            delta_weights,
            temp_delta_weights,
            transposed_weights,
            delta_bias
    );

    return 0;
}

void write_network_introspection_params(Network *network)
{
#ifdef INTROSPECT
    if (!g_introspection_file_handle)
    {
        printf("g_introspection_file_handle == null!!!");
        return;
    }

    if (!network)
    {
        printf("Network == NULL !!!");
    }

    fprintf(g_introspection_file_handle, "%ld\n", network->num_layers);

    for (int layer_idx = 0; layer_idx < network->num_layers; layer_idx++)
    {
        Layer* layer = network->layers[layer_idx];

        fprintf(g_introspection_file_handle, "%ld\n", layer->num_neurons);
    }

#else
    printf("Warning! Calling introspection functions, but INTROSPECT is undefined!!!");
#endif
}