#ifndef NNS_TRAIN_OPTS_H
#define NNS_TRAIN_OPTS_H

typedef struct
{
    CostType cost_type;
    int batch_size;
    int epochs;
    double learning_rate;
    double momentum;
    double regularization_lambda;
} TrainingOptions;

#endif //NNS_TRAIN_OPTS_H
