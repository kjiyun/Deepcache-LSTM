#ifndef HR_MODEL_H
#define HR_MODEL_H

#include "requests.h"
#include <LightGBM/c_api.h>
#include <thread>
#include <mutex>

struct HR_Model {
    double **data;
    int row_count;
    bool full;
    int features_length;
    int max_boost_round;
    int max_train_set_count;
    bool available;
    std::mutex mtx;
    DatasetHandle* dataset_handle;
    BoosterHandle* booster_handle;
    DatasetHandle* new_dataset_handle;
    BoosterHandle* new_booster_handle;
};

HR_Model* create_hr_model(int cache_size, int features_length, int max_boost_round);
void update_hr_model(HR_Model* model, HR_Request** requests, int requests_count, bool verbose);
void train_hr_model(HR_Model* model, bool verbose=false);
double predict_hr_label(HR_Model* model, double* features);
void predict_requests(HR_Model* model, HR_Request** requests, int requests_count);

void destroy_hr_model(HR_Model* model);

#endif // HR_MODEL_H