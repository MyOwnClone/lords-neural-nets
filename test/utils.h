#ifndef NNS_UTILS_H
#define NNS_UTILS_H

#include <stdlib.h>
#include "matrix.h"

int fail(const char *test_name, int line, const char *message);
int eval_test_result(const char *test_name, int result);
bool is_non_zero(Matrix *matrix);

#endif //NNS_UTILS_H
