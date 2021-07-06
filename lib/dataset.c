#include <stdlib.h>
#include "dataset.h"

Dataset *create_dataset(int train_size, int input_size, int output_size, int val_size, Matrix **train_inputs, Matrix **train_labels, Matrix **val_inputs, Matrix **val_labels)
{
    Dataset *dataset = (Dataset *) malloc(sizeof(Dataset));
    dataset->train_size = train_size;
    dataset->val_size = val_size;

    if (train_inputs == NULL || train_labels == NULL)
    {
        return NULL;
    }

    dataset->train_inputs = train_inputs;
    dataset->train_labels = train_labels;
    dataset->val_inputs = val_inputs;
    dataset->val_labels = val_labels;

    if (val_inputs == NULL)
    {
        dataset->val_inputs = train_inputs;
        dataset->val_labels = train_labels;
    }

    return dataset;
}

int delete_dataset(Dataset *dataset)
{
    if (dataset == NULL)
    {
        return -1;
    }

    bool train_and_val_are_same = false;

    // if train and validation matrices are the same (see line 48) we would have double free() call
    if (dataset->train_size > 0 && dataset->val_size > 0)
    {
        train_and_val_are_same = dataset->train_inputs[0] == dataset->val_inputs[0];
    }

    for (int i = 0; i < dataset->train_size; i++)
    {
        delete_matrix(dataset->train_inputs[i]);
        delete_matrix(dataset->train_labels[i]);
    }

    if (!train_and_val_are_same)
    {
        for (int i = 0; i < dataset->val_size; i++)
        {
            delete_matrix(dataset->val_inputs[i]);
            delete_matrix(dataset->val_labels[i]);
        }
    }

    free(dataset);
    dataset = NULL;

    return 0;
}

