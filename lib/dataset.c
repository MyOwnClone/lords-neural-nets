#include <stdlib.h>
#include "dataset.h"

// renamed from create_dataset() to reflect the fact that it allocates memory (generate_* prefix), also _structure suffix added, since we do not generate actual dataset sample pairs (they are loaded), just a c structures in memory
Dataset *generate_dataset_structures(int train_size, int val_size, Matrix **train_inputs, Matrix **train_labels, Matrix **val_inputs, Matrix **val_labels)
{
    Dataset *dataset_gen = (Dataset *) malloc(sizeof(Dataset));
    dataset_gen->train_size = train_size;
    dataset_gen->val_size = val_size;

    if (train_inputs == NULL || train_labels == NULL)
    {
        return NULL;
    }

    dataset_gen->train_inputs = train_inputs;
    dataset_gen->train_labels = train_labels;
    dataset_gen->val_inputs = val_inputs;
    dataset_gen->val_labels = val_labels;

    if (val_inputs == NULL)
    {
        dataset_gen->val_inputs = train_inputs;
        dataset_gen->val_labels = train_labels;
    }

    return dataset_gen;
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