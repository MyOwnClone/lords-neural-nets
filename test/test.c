#include "test.h"
#include "../lib/matrix.h"
#include "../lib/utils.h"
#include <stdbool.h>
#include <stdio.h>

int fail(const char *test_name, int line, const char *message)
{
#ifndef __MINGW64__
    RED_COLOR;
#endif
    printf("%s:%d %s\n", test_name, line, message);
#ifndef __MINGW64__
    RESET_COLOR;
#endif
    return -1;
}

int eval_test_result(const char *test_name, int result)
{
    if (result<0) {
#ifndef __MINGW64__
        RED_COLOR;
#else
        printf("X ");
#endif
        printf("%s\n", test_name);
#ifndef __MINGW64__
        RESET_COLOR;
#endif
    }
    else
    {
#ifndef __MINGW64__
        GREEN_COLOR;
#else
        printf("* ");
#endif
        printf("%s\n", test_name);
#ifndef __MINGW64__
        RESET_COLOR;
#endif
    }

    return result;
}

bool is_non_zero(Matrix *matrix)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        for (int j = 0; j < matrix->cols; j++)
        {
            if (MATRIX_IGET(matrix, i, j) != 0)
            {
                return true;
            }
        }        
    }
    return false;    
}

int main()
{
    int res = 0;
    res += test_matrix();
    res += test_layer();
    res += test_network();
    res += test_functions();
    res += test_utils();

    eval_test_result("All tests finished!", res);

    return res < 0 ? 1 : 0; // different exit code for passing and failing
}