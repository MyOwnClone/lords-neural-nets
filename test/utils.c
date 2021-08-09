#include <stdbool.h>
#include <stdio.h>
#include "utils.h"
#include "..\lib\utils.h"

int fail(const char *test_name, int line, const char *message)
{
    RED_COLOR;
    printf("%s:%d %s\n", test_name, line, message);
    RESET_COLOR;
    return -1;
}

int eval_test_result(const char *test_name, int result)
{
    if (result<0) {
        RED_COLOR;
        printf("%s\n", test_name);
        RESET_COLOR;
    }
    else
    {
        GREEN_COLOR;
        printf("%s\n", test_name);
        RESET_COLOR;
    }

    return result;
}

bool is_non_zero(Matrix *matrix)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            if (DISP_MATRIX_IGET(matrix, i, j) != 0)
            {
                return true;
            }
        }
    }
    return false;
}

void delete_train_test_data(Activation *act_sigmoid, Network *xor_network, Dataset *dataset, TrainingOptions *training_options, TrainingLoggingOptions *training_logging_options)
{
    delete_network(xor_network);
    delete_activation(act_sigmoid);
    delete_dataset(dataset);
    delete_training_options(training_options);
    delete_training_logging_options(training_logging_options);
}

