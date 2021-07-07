#include <stdbool.h>
#include "test.h"
#include "utils.h"
#include "../lib/activations.h"

static bool is_approx_equal(double a, double b)
{
    int a_int = (int) a * 1000000;
    int b_int = (int) b * 1000000;

    return a_int == b_int;
}

static int test_act_sigmoid_float()
{
    int res = 0;

    if (act_sigmoid_f(0) != 0.5 ||
        !is_approx_equal(act_sigmoid_f(-1), 0.2689414213699951207488) ||
        !is_approx_equal(act_sigmoid_f(1), 0.7310585786300048792512))
    {
        res += fail(__func__,  __LINE__, "Unexpected sigmoid value");
    }

    return eval_test_result(__func__, res);
}


static int test_act_sigmoid()
{
    int res = 0;

    if (act_sigmoid_d(0) != 0.5 ||
        !is_approx_equal(act_sigmoid_d(-1), 0.2689414213699951207488) ||
        !is_approx_equal(act_sigmoid_d(1), 0.7310585786300048792512))
    {
        res += fail(__func__,  __LINE__, "Unexpected sigmoid value");
    }

    return eval_test_result(__func__, res);
}

static int test_act_sigmoid_der()
{
    int res = 0;

    if (act_sigmoid_der_d(0) != 0.25 ||
        !is_approx_equal(act_sigmoid_der_d(-1), 0.1966119332414818525374) ||
        !is_approx_equal(act_sigmoid_der_d(1), 0.1966119332414818525374))
    {
        res += fail(__func__,  __LINE__, "Unexpected sigmoid derivation value");
    }

    return eval_test_result(__func__, res);
}

static int test_act_sigmoid_der_float()
{
    int res = 0;

    if (act_sigmoid_der_f(0) != 0.25 ||
        !is_approx_equal(act_sigmoid_der_f(-1), 0.1966119332414818525374) ||
        !is_approx_equal(act_sigmoid_der_f(1), 0.1966119332414818525374))
    {
        res += fail(__func__,  __LINE__, "Unexpected sigmoid derivation value");
    }

    return eval_test_result(__func__, res);
}

static int test_act_relu_float()
{
    int res = 0;

    if (act_relu_f(-1) != 0 || act_relu_f(0) != 0 || act_relu_f(1) != 1 || act_relu_f(2) != 2)
    {
        res += fail(__func__,  __LINE__, "Unexpected relu value");
    }
    return eval_test_result(__func__, res);
}

static int test_act_relu()
{
    int res = 0;

    if (act_relu_d(-1) != 0 || act_relu_d(0) != 0 || act_relu_d(1) != 1 || act_relu_d(2) != 2)
    {
        res += fail(__func__,  __LINE__, "Unexpected relu value");
    }
    return eval_test_result(__func__, res);
}

static int test_act_relu_der_float()
{
    int res = 0;

    if (act_relu_der_f(-1) != 0 || act_relu_der_f(0) != 0 || act_relu_der_f(1) != 1 || act_relu_der_f(2) != 1)
    {
        res += fail(__func__,  __LINE__, "Unexpected relu derivation value");
    }
    return eval_test_result(__func__, res);
}

static int test_act_relu_der()
{
    int res = 0;

    if (act_relu_der_d(-1) != 0 || act_relu_der_d(0) != 0 || act_relu_der_d(1) != 1 || act_relu_der_d(2) != 1)
    {
        res += fail(__func__,  __LINE__, "Unexpected relu derivation value");
    }
    return eval_test_result(__func__, res);
}

int test_functions()
{
    int res = 0;
    res += test_act_sigmoid();
    res += test_act_sigmoid_float();
    res += test_act_sigmoid_der();
    res += test_act_sigmoid_der_float();
    res += test_act_relu();
    res += test_act_relu_float();
    res += test_act_relu_der();
    res += test_act_relu_der_float();
    return res;
}