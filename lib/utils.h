#ifndef UTILS_H
#define UTILS_H

#include "matrix.h"
#include "functions.h"
#include "dataset.h"
#include "train_opts.h"
#include "train_log_opts.h"

#define RED_COLOR printf("\033[0;31m")
#define GREEN_COLOR printf("\033[0;32m")
#define RESET_COLOR printf("\033[0m" )

#define DEBUG 0
#define INFO 1
#define EXCEPTION 2

#ifndef LOG_LEVEL
#define LOG_LEVEL 1
#endif

TrainingOptions* init_training_options();
int delete_training_options(TrainingOptions *training_options);

TrainingLoggingOptions* init_training_logging_options();
int delete_training_logging_options(TrainingLoggingOptions *training_logging_options);

void logger(int log_level, const char *function_name, const char *message);

Matrix** load_csv(char *filename, int lines, int line_length, MatrixDataType matrixDataType);
int vectorize(Matrix **a, int length, int num_classes);
int normalize(Matrix **a, int length, int max_num);

#endif
