#include "test.h"
#include "utils.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    int res = 0;
    res += test_matrix();
    res += test_layer();
    res += test_network();
    res += test_functions();
    res += test_utils();
    res += test_train_xor_float();
    res += test_train_xor_double();
    res += test_train_mnist_double();
    res += test_train_mnist_float();

    if (res < 0)
    {
#ifndef __MINGW64__
        RED_COLOR;
#endif
    }

    printf("%d tests failed!\n", abs(res));

#ifndef __MINGW64__
    RESET_COLOR;
#endif

    eval_test_result("All tests finished", res);

    return res < 0 ? 1 : 0; // different exit code for passing and failing
}