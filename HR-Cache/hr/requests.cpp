#include "requests.h"
#include "utils.h"
#include <thread>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>
#include <random>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <set>

const int num_threads = std::thread::hardware_concurrency();
const long long MAXIMUM_PROCESS_COUNT = 10 * 1000 * 1000 * num_threads;
const int MINIMUM_WINDOW_SIZE = 10 * 1000;
const int MAXIMUM_WINDOW_SIZE = 10 * 1000 * 1000;
const double MAX_SAMPLE_RATE = 1;

// double sum_h = 0;

HR_RequestWindow* create_request_window(
    int *size,
    long long cache_size,
    int features_length,
    std::unordered_map<HR_FEATURE, bool> features,
    HR_ObjectsMetadata* objects_metadata
) {
    std::srand(std::time(NULL));
    HR_RequestWindow *rw = new HR_RequestWindow;
    rw->size = size;
    rw->cache_size = cache_size;
    rw->features_length = features_length;
    rw->request = NULL;
    rw->requests_count = 0;
    rw->sampled_requests_count = 0;
    rw->objects_count = 0;
    rw->objects_size = 0;
    rw->objects_metadata = objects_metadata;

    int custom_features_count = 0;
    for (const auto& pair : features) {
        if (pair.second) {
            custom_features_count++;
        }
    }
    rw->features = features;
    rw->custom_features_count = custom_features_count;
    rw->sample_rate = 0;
    rw->avg_req_size = 0;
    rw->sampled_requests = NULL;
    return rw;
}

void update_default_features(HR_RequestWindow* request_window) {    
    for (const auto& pair : request_window->objects) {
        Object* object = pair.second;
        request_window->objects_metadata->update_features(object->id, object->request->prev->features);
    }
}

Object* create_object(HR_RequestWindow* request_window, int object_id, int size) {
    Object* object = new Object;
    object->id = object_id;
    object->request = NULL;
    object->requests_count = 0;
    object->size = size;
    object->sampled = false;
    object->hazard_bandwidth = 3;

    object->timestamps = NULL;
    object->timestamps_diffs = NULL;
    object->cumulative_hazards_diffs = NULL;
    object->diffs_count = 0;

    return object;
}

Object* get_object(HR_RequestWindow* request_window, const int object_id, int size) {
    Object* object = request_window->objects[object_id];
    if (object) {
        return object;
    }

    object = create_object(request_window, object_id, size);
    object->idx = request_window->objects_count;
    request_window->objects[object_id] = object;
    request_window->objects_count++;
    request_window->objects_size += size;
    return object;
}

void set_custom_features(HR_RequestWindow* request_window, HR_Request *request, Object *object) {
    int custom_features_count = request_window->custom_features_count;

    if (request_window->features[FEAT_SIZE]) {
        request->features[request_window->features_length - custom_features_count--] = request->size;
    }
    if (request_window->features[FEAT_FREQUENCY]) {
        request->features[request_window->features_length - custom_features_count--] = 
            static_cast<double>(object->requests_count) / request_window->requests_count;
    }
    if (request_window->features[FEAT_DECAYED_FREQUENCY]) {
        request->features[request_window->features_length - custom_features_count--] = 
            request_window->objects_metadata->get_decayed_frequency(object->id);
    }
}

HR_Request* add_request(HR_RequestWindow* request_window, int object_id, double timestamp, int size) {
    request_window->objects_metadata->seen(object_id, timestamp);

    double casted_size = static_cast<double>(size);
    if (request_window->requests_count == 1) {
        request_window->avg_req_size = casted_size;
    } else {
        request_window->avg_req_size = request_window->avg_req_size * (request_window->requests_count - 1) / request_window->requests_count + (casted_size / request_window->requests_count);
    }

    request_window->requests_count++;
    HR_Request* request = new HR_Request;
    if (!request_window->request) {
        request_window->request = request;
        request->next_in_time = request;
        request->prev_in_time = request;
    } else {
        HR_Request* end = request_window->request->prev_in_time;
        end->next_in_time = request;
        request->prev_in_time = end;
        request->next_in_time = request_window->request;
        request_window->request->prev_in_time = request;
    }

    request->object_id = object_id;
    request->timestamp = timestamp;
    request->size = size;
    request->admit_probability = 0;
    request->label = 0;
    request->features = new double[request_window->features_length];

    Object* object = get_object(request_window, request->object_id, size);
    object->requests_count++;

    if (!object->request) {
        object->request = request;
        request->prev = request;
        request->next = request;
        
        memcpy(
            request->features,
            request_window->objects_metadata->get_features(object->id),
            sizeof(double) * request_window->features_length
        );
    } else {
        HR_Request* end = object->request->prev;
        end->next = request;
        request->prev = end;
        request->next = object->request;
        object->request->prev = request;

        if (request_window->features_length - request_window->custom_features_count > 0) {
            // Copy features from the previous request and shifting them to the left
            memcpy(request->features, end->features + 1, sizeof(double) * (request_window->features_length - 1));
            // Add the new timestamp diff feature to the end before the last two features
            request->features[request_window->features_length - 1 - request_window->custom_features_count] = request->timestamp - end->timestamp;
        } else {
            memcpy(request->features, end->features, sizeof(double) * request_window->features_length);
        }
    }

    set_custom_features(request_window, request, object);

    return request;
}

double calculate_object_hazard(Object* object, double timestamp_diff) {
    return calculate_hazard(
        timestamp_diff,
        object->timestamps_diffs,
        object->cumulative_hazards_diffs,
        object->diffs_count,
        object->hazard_bandwidth
    );
}

void prepare_object_samples(Object* object, double last_timestamp, bool discrete) {
    HR_Request* request = object->request;
    for (int i = 0; i < object->requests_count; i++) {
        object->timestamps[i] = request->timestamp;
        request = request->next;
    }
    object->timestamps[object->requests_count] = last_timestamp;

    calculate_diffs(object->timestamps, object->requests_count + 1, object->timestamps_diffs, &object->diffs_count);
    nelson_aalen_fitter(object->timestamps_diffs, object->cumulative_hazards_diffs, &object->diffs_count, discrete);

    // Calculate standard deviation
    double mean = std::accumulate(object->timestamps_diffs, object->timestamps_diffs + object->diffs_count, 0.0) / object->diffs_count;
    double sq_sum = std::inner_product(object->timestamps_diffs, object->timestamps_diffs + object->diffs_count, object->timestamps_diffs, 0.0);
    double std_dev = std::sqrt(sq_sum / object->diffs_count - mean * mean);

    // Scott's rule for univariate data
    object->hazard_bandwidth = 3.49 * std_dev / std::cbrt(object->diffs_count);
    // sum_h += object->hazard_bandwidth;
}

void label_request(HR_RequestWindow* request_window, std::vector<Object*> *objects, HR_Request* request, double* last_timestamps) {
    Object* current_object = request_window->objects[request->object_id];
    if (current_object->requests_count <= 1) {
        request->label = 0;
        return;
    }

    double current_hazard = calculate_object_hazard(
        current_object,
        request->timestamp - last_timestamps[current_object->idx]
    );
    long long cache_size = static_cast<long long>(request_window->cache_size * request_window->sample_rate);
    double current_size = 0;

    for (int i = 0; i < objects->size(); ++i) {
        Object* object = (*objects)[i];

        if (object->id == current_object->id) {
            continue;
        }

        double hazard = calculate_object_hazard(
            object,
            request->timestamp - last_timestamps[object->idx]
        );
        if (hazard >= current_hazard) {
            current_size += object->size;
        }
    }

    if (current_size + request->size <= cache_size) {
        request->label = 1;
    } else if (current_size < cache_size) {
        double remained_fraction = (cache_size - current_size) / request->size;
        if ((double)rand() / RAND_MAX < remained_fraction) {
            request->label = 1;
        } else {
            request->label = 0;
        }
    } else {
        request->label = 0;
    }
}

void prepare_objects(HR_RequestWindow* request_window, std::vector<Object*> *objects, bool discrete, bool verbose) {
    std::thread threads[num_threads];

    double last_timestamp = request_window->request->prev_in_time->timestamp;
    int requests_count = request_window->requests_count;
    int objects_count = objects->size();
    int chunk_size = (objects_count + num_threads - 1) / num_threads;  // Round up division

    for (int i = 0; i < objects_count; ++i) {
        Object* object = (*objects)[i];
        object->timestamps = new double[object->requests_count + 2];
        object->timestamps_diffs = new double[object->requests_count + 2];
        object->cumulative_hazards_diffs = new double[object->requests_count + 2];
    }

    for (int i = 0; i < num_threads; ++i) {
        int chunk_start = i * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, objects_count);

        threads[i] = std::thread([objects, chunk_start, chunk_end, last_timestamp, discrete]() {
            for (int j = chunk_start; j < chunk_end; ++j) {
                prepare_object_samples((*objects)[j], last_timestamp, discrete);
            }
        });
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
    if (verbose) {
        std::cout << "Number of objects prepared: " << objects_count << std::endl;
    }
    // std::cout << "sum_h: " << sum_h / objects_count << std::endl;
    // sum_h = 0;
}

void prepare_requests(HR_RequestWindow* request_window, std::vector<Object*> *objects, 
        bool future_labeling, bool verbose) {
    std::thread threads[num_threads];

    int requests_count = request_window->sampled_requests_count;
    int chunk_size = (requests_count + num_threads - 1) / num_threads;  // Round up division
    
    double** last_timestamps_by_thread = new double*[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        last_timestamps_by_thread[i] = new double[request_window->objects_count];
    }

    for (int i = 0; i < num_threads; ++i) {
        int chunk_start = i * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, requests_count);
        double* last_timestamps = last_timestamps_by_thread[i];

        threads[i] = std::thread([request_window, objects, chunk_start, chunk_end, last_timestamps]() {
            for (int j = 0; j < chunk_start; ++j) {
                last_timestamps[request_window->objects[request_window->sampled_requests[j]->object_id]->idx] = 
                    request_window->sampled_requests[j]->timestamp;
            }
            for (int j = chunk_start; j < chunk_end; ++j) {
                label_request(request_window, objects, request_window->sampled_requests[j], last_timestamps);
                last_timestamps[request_window->objects[request_window->sampled_requests[j]->object_id]->idx] = 
                    request_window->sampled_requests[j]->timestamp;
            }
        });
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
    if (verbose) {
        std::cout << "Number of requests labeled: " << requests_count << std::endl;
    }

    if (future_labeling) {
        for (int i = 0; i < objects->size(); ++i) {
            Object* object = (*objects)[i];
            HR_Request* request = object->request;
            while (request) {
                if (request->next) {
                    request->label = request->next->label;
                }
                request = request->next;

                if (request == object->request) {
                    break;
                }
            }
        }
        if (verbose) {
            std::cout << "Future labeling done" << std::endl;
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        delete[] last_timestamps_by_thread[i];
    }
    delete[] last_timestamps_by_thread;
}

bool window_is_ready(HR_RequestWindow* request_window, double weight) {
    if (request_window->size) {
        return request_window->requests_count >= *request_window->size;
    }

    if (request_window->requests_count < MINIMUM_WINDOW_SIZE) {
        return false;
    }

    if (request_window->requests_count >= MAXIMUM_WINDOW_SIZE) {
        return true;
    }

    return request_window->objects_size >= (1 / weight) * request_window->cache_size;

    // double objects_ln = log(request_window->objects_count);
    // return request_window->requests_count >= weight * (objects_ln * request_window->cache_size / request_window->avg_req_size);
}

void sample_objects(std::vector<Object*> *objects, HR_RequestWindow* request_window, int limit, bool verbose) {
    // First we create an array of objects indexes 0..objects_count-1 and shuffle it
    // Then we iterate over objects and add them to the samples until we hit the limits
    // We have two limits:
    // 1. sampled objects * their total requests count should be lower than MAXIMUM_PROCESS_COUNT
    // 2. total requests count of sampled requests should be lower than limit
    // TODO: Shuffle or something like that is taking too much time for huge number of objects
    if (verbose) {
        std::cout << "Sampling objects & requests ..." << std::endl;
    }

    const int max_requests_num = std::min(limit, static_cast<int>(MAX_SAMPLE_RATE * request_window->requests_count));
    const double estimated_sample_rate = static_cast<double>(max_requests_num) / static_cast<double>(request_window->requests_count);

    int* object_ids = new int[request_window->objects_count];
    for (const auto& pair : request_window->objects) {
        object_ids[pair.second->idx] = pair.second->id;
    }
    shuffle(object_ids, object_ids + request_window->objects_count, std::mt19937{std::random_device{}()});

    double total_objects_size = 0;
    double objects_size = 0;
    request_window->sampled_requests_count = 0;
    for (int i = 0; i < request_window->objects_count; ++i) {
        Object* object = request_window->objects[object_ids[i]];
        total_objects_size += object->size;

        long long potential_requests_count = request_window->sampled_requests_count + object->requests_count;

        if (potential_requests_count > max_requests_num) {
            continue;
        }

        if (potential_requests_count * (objects->size() + 1) > MAXIMUM_PROCESS_COUNT) {
            continue;
        }

        objects_size += object->size;
        object->sampled = true;
        objects->push_back(object);
        request_window->sampled_requests_count += object->requests_count;
    }

    request_window->sample_rate = objects_size / total_objects_size;
    request_window->sampled_requests = new HR_Request*[request_window->sampled_requests_count];
    request_window->sampled_requests_count = 0;
    HR_Request* request = request_window->request;
    for (int i = 0; i < request_window->requests_count; ++i) {
        if (request_window->objects[request->object_id]->sampled) {
            request_window->sampled_requests[request_window->sampled_requests_count++] = request;
        }
        request = request->next_in_time;
    }

    if (verbose) {
        std::cout << std::setprecision(5);
        std::cout << "Number of threads: " << num_threads << std::endl;
        std::cout << "Average request size: " << request_window->avg_req_size << std::endl;
        std::cout << "HR_Requests count: " << request_window->requests_count << ", Objects count: " << request_window->objects_count << std::endl;
        std::cout << "Sampled objects: " << objects->size() << ", Sampled requests: " << request_window->sampled_requests_count << std::endl;
    }

    delete[] object_ids;
}

void prepare_request_window(HR_RequestWindow* request_window, int max_requests_count, double bandwidth, 
        bool discrete, bool future_labeling, bool verbose) {
    std::vector<Object*> objects;
    sample_objects(&objects, request_window, max_requests_count, verbose);
    prepare_objects(request_window, &objects, discrete, verbose);
    prepare_requests(request_window, &objects, future_labeling, verbose);

    if (verbose) {
        double hr_bound = 0;
        for (int i = 0; i < request_window->sampled_requests_count; i++) {
            hr_bound += request_window->sampled_requests[i]->label;
        }
        std::cout << "HR Bound: " << hr_bound / request_window->sampled_requests_count << std::endl;
    }
}

void destroy_request_window(HR_RequestWindow* request_window) {
    HR_Request* request = request_window->request;
    for (int i = 0; i < request_window->requests_count; i++) {
        HR_Request* temp = request;
        request = request->next_in_time;
        delete[] temp->features;
        delete temp;
    }

    for (const auto& pair : request_window->objects) {
        Object* object = pair.second;
        object->hazards.clear();
        if (object->timestamps) {
            delete[] object->timestamps;
        }
        if (object->timestamps_diffs) {
            delete[] object->timestamps_diffs;
        }
        if (object->cumulative_hazards_diffs) {
            delete[] object->cumulative_hazards_diffs;
        }
        delete object;
    }

    if (request_window->sampled_requests) {
        delete[] request_window->sampled_requests;
    }

    request_window->objects.clear();
    delete request_window;
}