#include <stdio.h>
#include <sys/time.h>
#include "bench_utils.h"

void print_elapsed_time(fn func, char* desc)
{
    struct timeval tval_before, tval_after, tval_result;

    gettimeofday(&tval_before, NULL);

    func();

    gettimeofday(&tval_after, NULL);

    timersub(&tval_after, &tval_before, &tval_result);
    printf("%s: Time elapsed: %ld.%06ld\n", (desc), (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}