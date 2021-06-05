#include <stdio.h>
#include "bench_utils.h"

#ifdef __MINGW64__
    #include <sys\timeb.h>

    double print_elapsed_time(fn func, char* desc, long long repeat_count)
    {
        struct timeb start, end;
        long diff = 0;

        for (long long repeat = 0; repeat < repeat_count; repeat++)
        {
            ftime(&start);

            func();

            ftime(&end);
            diff += (int) (1000.0 * (double)(end.time - start.time) + (end.millitm - start.millitm));
        }

        double average_msecs = (double)diff / repeat_count;

        printf("%s: Average time elapsed over %ld runs: %f ms\n", (desc), repeat_count, average_msecs);

        return average_msecs;
    }
#else
    #include <sys/time.h>

    double print_elapsed_time(fn func, char* desc, long long repeat_count)
    {
        struct timeval tval_before, tval_after, tval_result, tval_overall;

        timerclear(&tval_overall);

        for (long long repeat = 0; repeat < repeat_count; repeat++)
        {
            gettimeofday(&tval_before, NULL);

            func();

            gettimeofday(&tval_after, NULL);

            timersub(&tval_after, &tval_before, &tval_result);

            timeradd(&tval_result, &tval_overall, &tval_overall);
        }

        long seconds_as_usecs = (long int)tval_overall.tv_sec * 1000 * 1000;
        long usecs = (long int)tval_overall.tv_usec;

        long average_usecs = (seconds_as_usecs + usecs) / repeat_count;
        double average_msecs = (double)average_usecs / 1000.0f;

        printf("%s: Average time elapsed over %ld runs: %f ms\n", (desc), repeat_count, average_msecs);

        return average_msecs;
    }
#endif