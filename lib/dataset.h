#ifndef NNS_DATASET_H
#define NNS_DATASET_H

#include "matrix.h"

typedef struct
{
    int train_size;
    Matrix **train_inputs;
    Matrix **train_labels;
    int val_size;
    Matrix **val_inputs;
    Matrix **val_labels;
} Dataset;

Dataset *create_dataset(int train_size, int val_size, Matrix **train_inputs, Matrix **train_labels, Matrix **val_inputs, Matrix **val_labels);
int delete_dataset(Dataset *dataset);

#endif //NNS_DATASET_H
