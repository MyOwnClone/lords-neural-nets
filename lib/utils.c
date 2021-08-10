#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

FILE* g_introspection_file_handle = NULL;

void logger(int log_level, const char *function_name, const char *message)
{
    if (log_level >= LOG_LEVEL)
    {
        if (log_level == LOG_EXCEPTION)
        {
#ifndef __MINGW64__
            RED_COLOR;
#endif
        }
        printf("%s: %s\n", function_name, message);
#ifndef __MINGW64__
        RESET_COLOR;
#endif
    }
}


Matrix** load_csv_to_generated_matrix(char *filename, int lines, int line_length, MatrixDataType matrixDataType)
{
    FILE* fp = fopen(filename, "r");

    if (!fp) 
    {
        logger(LOG_EXCEPTION, __func__, "Failed to open csv file");
        return NULL;
    }

    Matrix **result_gen = (Matrix**) malloc (sizeof (Matrix*) * lines);

    int buffer_length = line_length*4;
    char buffer[buffer_length];

    int line_idx = 0;
    while(fgets(buffer, buffer_length, fp)) {        
        char *token = strtok(buffer, ",");

        if (matrixDataType == D_DOUBLE)
        {
            double mat[line_length][1];

            int i = 0;
            while (token != NULL)
            {
                mat[i++][0] = strtod(token, NULL);
                token = strtok(NULL, ",");
            }

            result_gen[line_idx++] = generate_matrix_d(line_length, 1, mat);
        }
        else if (matrixDataType == D_FLOAT)
        {
            float mat[line_length][1];

            int i = 0;
            while (token != NULL)
            {
                mat[i++][0] = strtof(token, NULL);
                token = strtok(NULL, ",");
            }

            result_gen[line_idx++] = generate_matrix_f(line_length, 1, mat);
        }

        if (line_idx >= lines)
        {
            break;
        }
    }

    fclose(fp);
    return result_gen;
}

// creates one hot encoding?
int vectorize(Matrix **a, int length, int num_classes)
{
    bool float_matrix = is_float_matrix(a[0]);

    for (int i = 0; i < length; i++)
    {
        int index = (int) DISP_MATRIX_IGET(a[i], 0, 0);

        if (index >= num_classes)
        {
            return -1;
        }

        if (float_matrix)
        {
            float mat[num_classes][1];
            for (int j = 0; j < num_classes; j++) {
                mat[j][0] = 0;
            }

            mat[index][0] = 1;

            delete_matrix(a[i]);
            a[i] = generate_matrix_f(num_classes, 1, mat);
        }
        else {
            double mat[num_classes][1];
            for (int j = 0; j < num_classes; j++) {
                mat[j][0] = 0;
            }

            mat[index][0] = 1;

            delete_matrix(a[i]);
            a[i] = generate_matrix_d(num_classes, 1, mat);
        }
    }

    return 0;    
}

int normalize(Matrix **a, int length, int max_num)
{
    for (int i = 0; i < length; i++)
    {
        Matrix *matrix = a[i];
        if (is_null(matrix))
        {
            return -1;
        }

        for (int j = 0; j < matrix->rows; j++)
        {
            for (int k = 0; k < matrix->cols; k++)
            {
                DISP_MATRIX_ISET(matrix, j, k, DISP_MATRIX_IGET(matrix, j, k) / max_num);
            }            
        }        
    }
    
    return 0;
}

TrainingOptions* generate_training_options()
{
    TrainingOptions *training_options = (TrainingOptions *) malloc (sizeof (TrainingOptions));
    training_options->cost_type = CROSS_ENTROPY;
    training_options->epochs = 0;
    training_options->batch_size = 0;
    training_options->learning_rate = 0;
    training_options->momentum = 0;
    training_options->regularization_lambda = 0;

    return training_options;
}

TrainingLoggingOptions* generate_training_logging_options()
{
    TrainingLoggingOptions *training_logging_options = (TrainingLoggingOptions *) malloc (sizeof (TrainingLoggingOptions));
    training_logging_options->log_loss = true;
    training_logging_options->log_accuracy = true;
    training_logging_options->log_each_nth_epoch = 1;

    return training_logging_options;
}

int delete_training_options(TrainingOptions *training_options)
{
    free(training_options);
    training_options = NULL;

    return 0;
}

int delete_training_logging_options(TrainingLoggingOptions *training_logging_options)
{
    if (!training_logging_options)
    {
        return 0;
    }

    free(training_logging_options);
    training_logging_options = NULL;

    return 0;
}

float convert_bytes_to_kbytes(long bytes)
{
    return (float)bytes / 1024.0f;
}

float convert_bytes_to_Mbytes(long bytes)
{
    return (float)bytes / 1024.0f / 1024.0f;
}
