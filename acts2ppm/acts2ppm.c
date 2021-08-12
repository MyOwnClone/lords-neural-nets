#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <limits.h>
#include "../lib/utils.h"

#define PPM_HEADER "P3"
#define PPM_COMMENT "# lnn activations from one epoch"
#define APP_NAME "acts2ppm"

#define MAX_PPM_INT_VALUE 65536

// source: https://stackoverflow.com/questions/122616/how-do-i-trim-leading-trailing-whitespace-in-a-standard-way
// Stores the trimmed input string into the given output buffer, which must be
// large enough to store the result.  If it is too small, the output is
// truncated.
size_t trim_whitespace(char *out, size_t len, const char *str)
{
    if(len == 0)
        return 0;

    const char *end;
    size_t out_size;

    // Trim leading space
    while(isspace((unsigned char)*str)) str++;

    if(*str == 0)  // All spaces?
    {
        *out = 0;
        return 1;
    }

    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    end++;

    // Set output size to minimum of trimmed string length and buffer size minus 1
    out_size = (end - str) < len-1 ? (end - str) : len-1;

    // Copy trimmed string and add null terminator
    memcpy(out, str, out_size);
    out[out_size] = 0;

    return out_size;
}

// example params: mnist_d.acts
int main(int argc, char **argv)
{
    if (argc == 1)
    {
        printf("usage: %s file.acts", APP_NAME);

        return -1;
    }

    FILE *act_file = fopen(argv[1], "r");

    if (!act_file)
    {
        RED_COLOR;
        printf("File %s cannot be open!", argv[1]);
        perror("error while opening acts file:");
        RESET_COLOR;

        return -1;
    }

    const int MAX_LINE_LEN = 1024;

    char line_buff[MAX_LINE_LEN];
    char trimmed_line_buff[MAX_LINE_LEN];

    int line_counter = 0;
    int layer_count = 0;

    int epoch_count = 0;

    int* neuron_count_arr = NULL;

    FILE * out_file = NULL;

    int max_neuron_count = INT_MIN;

    int last_col = 0;

    char output_filename[MAX_LINE_LEN];

    while (fgets(line_buff, MAX_LINE_LEN, act_file) != NULL)
    {
        trim_whitespace(trimmed_line_buff, MAX_LINE_LEN, line_buff);

        if (line_counter == 0)
        {
            layer_count = atoi(line_buff);

            neuron_count_arr = (int*) malloc( layer_count * sizeof(int) );

            if (neuron_count_arr == NULL)
            {
                RED_COLOR;
                printf("Malloc failed!");
                perror("error while allocating memory:");
                RESET_COLOR;
                exit(1);
            }

            printf("network has %d layers\n", layer_count);
        }

        if (line_counter >= 1 && line_counter <= layer_count)
        {
            int layer_idx = line_counter-1;

            int neuron_count = atoi(line_buff);

            neuron_count_arr[layer_idx] = neuron_count;

            if (neuron_count > max_neuron_count)
            {
                max_neuron_count = neuron_count;
            }

            printf("layer %d has %d neurons\n", layer_idx, neuron_count_arr[layer_idx]);
        }

        if (line_counter == 1 + layer_count)
        {
            if (strncmp(trimmed_line_buff, "==", MAX_LINE_LEN) != 0)
            {
                RED_COLOR;
                printf("Expected == on line %d!!!\n", line_counter);
                RESET_COLOR;
                exit(1);
            }
        }

        if (strncmp(trimmed_line_buff, "==", MAX_LINE_LEN) == 0)
        {
            if (out_file != NULL)
            {
                fclose(out_file);
            }

            sprintf(output_filename, "ppm_out\\\\%d.ppm", epoch_count++);

            out_file = fopen(output_filename, "w");

            if (out_file == NULL)
            {
                RED_COLOR;
                printf("cannot open output file for writing!\n");
                perror("error when opening output file!");
                RESET_COLOR;
                exit(1);
            }

            fprintf(out_file, "%s\n", PPM_HEADER);
            fprintf(out_file, "%d %d\n", max_neuron_count, layer_count);
            fprintf(out_file, "%d\n", MAX_PPM_INT_VALUE);
        }

        if (line_counter > 1 + layer_count)
        {
            int layer_idx =-1, row = -1, col =-1;

            float value;

            sscanf(trimmed_line_buff, "%d %d %d : %f", &layer_idx, &row, &col, &value);

            bool new_line = (col < last_col);

            if (new_line)
            {
                fprintf(out_file, "\n");
            }

            last_col = col;

            int int_value = value * MAX_PPM_INT_VALUE;

            // commented line gives nice images :-D
            // fprintf(out_file, "%d %d %d ", int_value);

            fprintf(out_file, "%d %d %d ", int_value, int_value, int_value);
        }

        line_counter++;
    }

    fclose(act_file);

    if (neuron_count_arr)
    {
        free(neuron_count_arr);
    }

    if (out_file)
    {
        fclose(out_file);
    }

    GREEN_COLOR;
    printf("Operation completed successfully.\n");
    RESET_COLOR;

    return 0;
}
