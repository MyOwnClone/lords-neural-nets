#ifndef NNS_TRAIN_LOG_OPTS_H
#define NNS_TRAIN_LOG_OPTS_H

typedef struct
{
    bool log_accuracy;
    bool log_loss;
    int log_each_nth_epoch;    // -1 means no logging
} TrainingLoggingOptions;

#define NO_LOGGING -1

#endif //NNS_TRAIN_LOG_OPTS_H
