#ifndef UTILS_H
#define UTILS_H

#include "matrix.h"
#include "dataset.h"
#include "train_opts.h"
#include "train_log_opts.h"

#ifdef __WIN32__
    #include <windows.h>

    // i can't properly name few colors, so take this with a grain of salt :-)
    typedef enum win32_text_color {
        BLACK = 0,
        LIGHT_GREY,
        GREEN,
        DARK_GREEN,
        RED,
        PURPLE,
        ORANGE,
        WHITE,
        GREY,
        BLUE,
        DARKER_GREEN,
        CYAN,
        LIGHT_RED,
        PINK,
        YELLOW
    } win32_text_color_t;

    #define DEFAULT_WIN32_COLOR WHITE

    // https://stackoverflow.com/questions/3274824/color-console-in-ansi-c
    #define set_color(color_index) \
    { \
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE); \
        SetConsoleTextAttribute(hConsole, color_index);    \
        hConsole = GetStdHandle(STD_ERROR_HANDLE);         \
        SetConsoleTextAttribute(hConsole, color_index);    \
    }

    #define RED_COLOR set_color(RED)
    #define RESET_COLOR set_color(WHITE)
    #define GREEN_COLOR set_color(GREEN)
#else

    #define RED_COLOR fprintf(stderr, "\033[0;31m"); printf("\033[0;31m")
    #define GREEN_COLOR fprintf(stderr, "\033[0;32m"); printf("\033[0;32m")
    #define RESET_COLOR fprintf(stderr, "\033[0m"); printf("\033[0m" )

#endif

#define LOG_DEBUG 0
#define LOG_INFO 1
#define LOG_EXCEPTION 2

#ifndef LOG_LEVEL
#define LOG_LEVEL 1
#endif

TrainingOptions* init_training_options();
int delete_training_options(TrainingOptions *training_options);

TrainingLoggingOptions* init_training_logging_options();
int delete_training_logging_options(TrainingLoggingOptions *training_logging_options);

void logger(int log_level, const char *function_name, const char *message);

Matrix** load_csv_to_generated_matrix(char *filename, int lines, int line_length, MatrixDataType matrixDataType);
int vectorize(Matrix **a, int length, int num_classes);
int normalize(Matrix **a, int length, int max_num);

#include <stdio.h>
extern FILE* g_introspection_file_handle;

typedef enum {
    IM_NONE,
    IM_PREDICT
} IntrospectionMode;

extern IntrospectionMode g_introspection_mode;

#endif
