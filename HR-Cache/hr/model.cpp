#include "model.h"
#include "requests.h"
#include <thread>
#include <mutex>
#include <LightGBM/c_api.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

const int MIN_DATA_SET_COUNT = 100 * 1000;
const int MAX_DATA_SET_COUNT = 1000 * 1000;

HR_Model* create_hr_model(int capacity, int features_length, int max_boost_round) {
    HR_Model* hr_model = new HR_Model;
    hr_model->row_count = 0;
    hr_model->full = false;
    hr_model->available = false;
    hr_model->max_boost_round = max_boost_round;
    int metadata_size = (features_length + 1) * sizeof(double);
    hr_model->max_train_set_count = std::min(capacity / metadata_size, MAX_DATA_SET_COUNT);
    hr_model->max_train_set_count = std::max(hr_model->max_train_set_count, MIN_DATA_SET_COUNT);
    hr_model->data = new double*[hr_model->max_train_set_count];
    for (int i = 0; i < hr_model->max_train_set_count; i++) {
        hr_model->data[i] = new double[features_length + 1];
    }
    hr_model->features_length = features_length;

    return hr_model;
}

void update_hr_model(HR_Model* model, HR_Request** requests, int requests_count, bool verbose) {
    std::lock_guard<std::mutex> lock(model->mtx);

    if (model->available) {
        LGBM_DatasetFree(*(model->dataset_handle));
        LGBM_BoosterFree(*(model->booster_handle));
        delete model->dataset_handle;
        delete model->booster_handle;
    }

    model->new_dataset_handle = new DatasetHandle;
    model->new_booster_handle = new BoosterHandle;

    if (model->full) {
        shuffle(model->data, model->data + model->max_train_set_count, std::mt19937{std::random_device{}()});
    }

    for (int i = model->row_count; i < model->row_count + requests_count; i++) {
        int row = i % model->max_train_set_count;
        int request_index = i - model->row_count;

        // Shuffle the data when we reach the end of the array
        if (row == 0 && i != 0) {
            model->full = true;
            shuffle(model->data, model->data + model->max_train_set_count, std::mt19937{std::random_device{}()});
        }

        for (int j = 0; j < model->features_length; j++) {
            model->data[row][j + 1] = requests[request_index]->features[j];
        }
        model->data[row][0] = static_cast<double>((*requests[request_index]).label);
    }
    model->row_count += requests_count;
    model->row_count %= model->max_train_set_count;

    int actual_row_count = model->full ? model->max_train_set_count : model->row_count;

    double *data = new double[actual_row_count * model->features_length];
    float *labels = new float[actual_row_count];
    for (int i = 0; i < actual_row_count; i++) {
        for (int j = 0; j < model->features_length; j++) {
            data[i * model->features_length + j] = model->data[i][j + 1];
        }
        labels[i] = static_cast<float>(model->data[i][0]);
    }

    LGBM_DatasetCreateFromMat(
        data,
        C_API_DTYPE_FLOAT64,
        actual_row_count,
        model->features_length,
        1,
        "max_bin=255",
        nullptr,
        model->new_dataset_handle
    );
    LGBM_DatasetSetField(*(model->new_dataset_handle), "label", labels, actual_row_count, C_API_DTYPE_FLOAT32);

    delete[] data;
    delete[] labels;

    train_hr_model(model, verbose);
}

void apply_new_model(HR_Model* model) {
    DatasetHandle* temp_dataset_handle = model->dataset_handle;
    BoosterHandle* temp_booster_handle = model->booster_handle;
    model->dataset_handle = model->new_dataset_handle;
    model->booster_handle = model->new_booster_handle;
    model->available = true;
}

void train_hr_model(HR_Model* model, bool verbose) {
    std::string parameters = "force_row_wise=true boosting_type=gbdt objective=binary learning_rate=0.1 num_leaves=32 max_depth=50 min_data_in_leaf=0";
    if (!verbose) {
        parameters += " verbosity=-1";
    }
    char* cparameters = new char[parameters.length() + 1];
    strcpy(cparameters, parameters.c_str());

    LGBM_BoosterCreate(*(model->new_dataset_handle), cparameters, model->new_booster_handle);

    for (int i = 0; i < model->max_boost_round; ++i) {
        int is_finished = 0;
        LGBM_BoosterUpdateOneIter(*(model->new_booster_handle), &is_finished);
        if (is_finished) break;
    }
    delete[] cparameters;

    if (verbose) {
        std::cout << "[LightGBM] [Info] Training finished" << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    apply_new_model(model);
}

double predict_hr_label(HR_Model* model, double* features) {
    std::lock_guard<std::mutex> lock(model->mtx);

    double result;
    int64_t out_len;
    int status = LGBM_BoosterPredictForMat(
        *(model->booster_handle),
        features,
        C_API_DTYPE_FLOAT64,
        1,
        model->features_length,
        1,
        C_API_PREDICT_NORMAL,
        0,
        -1,
        "",
        &out_len,
        &result
    );

    if (status != 0) {
        // Handle the error. For example, you can print the status code
        fprintf(stderr, "[LightGBM] [Error] Prediction failed with error code: %d\n", status);
    }

    return result;
}

void predict_requests(HR_Model* model, HR_Request** requests, int requests_count) {
    double* data = new double[requests_count * model->features_length];
    for (int i = 0; i < requests_count; i++) {
        for (int j = 0; j < model->features_length; j++) {
            data[i * model->features_length + j] = requests[i]->features[j];
        }
    }

    double* result = new double[requests_count];
    int64_t out_len;
    int status = LGBM_BoosterPredictForMat(
        *(model->booster_handle),
        data,
        C_API_DTYPE_FLOAT64,
        requests_count,
        model->features_length,
        1,
        C_API_PREDICT_NORMAL,
        0,
        -1,
        "",
        &out_len,
        result
    );

    if (status != 0) {
        // Handle the error. For example, you can print the status code
        fprintf(stderr, "[LightGBM] [Error] Prediction failed with error code: %d\n", status);
    }

    for (int i = 0; i < requests_count; i++) {
        requests[i]->admit_probability = result[i];
    }
}

void destroy_hr_model(HR_Model* model) {
    if (model->row_count > 0 || model->full) {
        LGBM_DatasetFree(*(model->dataset_handle));
        LGBM_BoosterFree(*(model->booster_handle));
        delete model->dataset_handle;
        delete model->booster_handle;
    }

    for (int i = 0; i < model->max_train_set_count; i++) {
        delete[] model->data[i];
    }
    delete[] model->data;
    delete model;
}
