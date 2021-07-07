#ifndef NNS_UTILS_H
#define NNS_UTILS_H

#include <stdlib.h>
#include "../lib/activations.h"
#include "../lib/network.h"
#include "matrix.h"

int fail(const char *test_name, int line, const char *message);
int eval_test_result(const char *test_name, int result);
bool is_non_zero(Matrix *matrix);

void delete_train_test_data(Activation *act_sigmoid, Network *xor_network, Dataset *dataset, TrainingOptions *training_options, TrainingLoggingOptions *training_logging_options);

#endif //NNS_UTILS_H
