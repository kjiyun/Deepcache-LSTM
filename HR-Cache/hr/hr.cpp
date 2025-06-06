#include "metadata.h"
#include "hr.h"
#include "requests.h"
#include "cache.h"
#include "model.h"
#include "utils.h"
#include <thread>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <ctime>

const int CONCURRENCY = 100;

const long long CACHE_SIZE = 3941722;
const double CACHE_HOT_LOWER_BOUND = 0.5;
const double CACHE_COLD_LOWER_BOUND = 0.0;
const bool CACHE_EVICT_HOT_FOR_COLD = true;

const int FEATURES_LENGTH = 32;
const double HAZARD_BANDWIDTH = 3;
const bool HAZARD_DISCRETE = true;
const bool FUTURE_LABELING = true;
const bool ONE_TIME_TRAINING = false;
const int MAX_BOOST_ROUNDS = 100;
const std::unordered_map<HR_FEATURE, bool> FEATURES = {
    {FEAT_FREQUENCY, false},
    {FEAT_SIZE, true},
    {FEAT_DECAYED_FREQUENCY, true}
};
const double DECAY_FACTOR = 0.9;
const double DEFAULT_LEARNING_RATE = 3;

const int REPORT_INTERVAL = 1000 * 1000;  // 기존 1000 * 1000

HRCache* create_hr(
    std::string key,
    std::optional<int> concurrency,
    std::optional<bool> verbose,
    std::optional<long long> cache_size,
    std::optional<double> cache_hot_lower_bound,
    std::optional<double> cache_cold_lower_bound,
    std::optional<bool> cache_evict_hot_for_cold,
    int* window_size,
    std::optional<double> learning_rate,
    std::optional<int> features_length,
    std::optional<double> decay_factor,
    std::optional<double> hazard_bandwidth,
    std::optional<bool> hazard_discrete,
    std::optional<bool> future_labeling,
    std::optional<bool> one_time_training,
    std::optional<int> max_boost_rounds,
    std::optional<std::unordered_map<HR_FEATURE, bool>> features,
    std::optional<int> report_interval,
    std::optional<bool> log_file,
    std::optional<bool> log_requests,
    std::optional<std::string> log_file_name
) {
    HRCache* hr = new HRCache;
    hr->key = key;
    hr->verbose = verbose.value_or(false);
    hr->lru_cache = create_lru_cache(
        cache_size.value_or(CACHE_SIZE),
        cache_hot_lower_bound.value_or(CACHE_HOT_LOWER_BOUND),
        cache_cold_lower_bound.value_or(CACHE_COLD_LOWER_BOUND),
        cache_evict_hot_for_cold.value_or(CACHE_EVICT_HOT_FOR_COLD)
    );
    hr->concurrency = concurrency.value_or(CONCURRENCY);

    std::unordered_map<HR_FEATURE, bool> final_features = features.value_or(FEATURES);
    int final_features_length = features_length.value_or(FEATURES_LENGTH);
    double final_decay_factor = final_features.at(FEAT_DECAYED_FREQUENCY) ? decay_factor.value_or(DECAY_FACTOR) : 0;
    hr->last_processed_request = NULL;

    hr->objects_metadata = new HR_ObjectsMetadata(
        static_cast<int>(hr->lru_cache->capacity * 0.02),
        final_features_length,
        final_decay_factor
    );
    hr->request_window = create_request_window(
        window_size,
        hr->lru_cache->capacity,
        final_features_length,
        final_features,
        hr->objects_metadata
    );
    hr->model = create_hr_model(
        static_cast<int>(hr->lru_cache->capacity * 0.03),
        hr->request_window->features_length,
        max_boost_rounds.value_or(MAX_BOOST_ROUNDS)
    );

    hr->learning_rate = learning_rate.value_or(DEFAULT_LEARNING_RATE);
    hr->hazard_bandwidth = hazard_bandwidth.value_or(HAZARD_BANDWIDTH);
    hr->hazard_discrete = hazard_discrete.value_or(HAZARD_DISCRETE);
    hr->future_labeling = future_labeling.value_or(FUTURE_LABELING);
    hr->one_time_training = one_time_training.value_or(ONE_TIME_TRAINING);
    hr->report_interval = report_interval.value_or(REPORT_INTERVAL);
    hr->log_file = log_file.value_or(false);
    hr->log_requests = log_requests.value_or(false);
    hr->requests_count = 0;

    hr->start_counting_cumulative = false;
    hr->without_training_count = 0;
    hr->cumulative_hot_evicted_bytes = 0;
    hr->cumulative_hot_evicted_reqs = 0;
    hr->cumulative_cold_evicted_bytes = 0;
    hr->cumulative_cold_evicted_reqs = 0;
    hr->analytics_hot_evicted_bytes = 0;
    hr->analytics_hot_evicted_reqs = 0;
    hr->analytics_cold_evicted_bytes = 0;
    hr->analytics_cold_evicted_reqs = 0;
    hr->cumulative_reqs = 0;
    hr->cumulative_reqs_hits = 0;
    hr->cumulative_times = 0;
    hr->cumulative_cpu_times = 0;
    hr->cumulative_bytes = 0;
    hr->cumulative_bytes_hit = 0;
    hr->analytics_reqs = 0;
    hr->analytics_reqs_hit = 0;
    hr->analytics_times = 0;
    hr->analytics_cpu_times = 0;
    hr->analytics_bytes = 0;
    hr->analytics_bytes_hit = 0;
    hr->analytics_round = 0;

    if (hr->log_requests) {
        hr->requests_file.open("requests.txt");
        hr->requests_file << std::setprecision(15);
    }

    if (hr->log_file) {
        std::string filename = log_file_name.value_or("analytics.csv");
        bool file_exists = std::filesystem::exists(filename);

        hr->analytics_file.open(filename, std::ios::app);
        hr->analytics_file << std::setprecision(15);

        if (!file_exists) {
            hr->analytics_file << "key,cache_size,cache_hot_lower_bound,cache_cold_lower_bound,cache_evict_hot_for_cold,";
            hr->analytics_file << "window_size,learning_rate,features_length,feature_size,feature_frequency,feature_decayed_frequency,";
            hr->analytics_file << "hazard_bandwidth,hazard_discrete,future_labeling,one_time_training,max_boost_rounds,";
            hr->analytics_file << "report_interval,analytics_round,miss_bytes_percentage,miss_percentage,cumulative_miss_bytes_percentage,";
            hr->analytics_file << "cumulative_miss_percentage" << std::endl;
        }
    }

    return hr;
}

double get_current_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // ru_maxrss는 kilobytes 기준이므로 MB로 변환
    return usage.ru_maxrss / 1024.0;
}

void log_args(HRCache* hr) {
    int window_size = hr->request_window->size ? *hr->request_window->size : 0;
    std::cout << std::setprecision(15);
    std::cout << "------------------------" << std::endl;
    std::cout << "Cache size: " << hr->lru_cache->capacity << std::endl;
    std::cout << "Cache hot lower bound: " << hr->lru_cache->hot_lower_bound << std::endl;
    std::cout << "Cache cold lower bound: " << hr->lru_cache->cold_lower_bound << std::endl;
    std::cout << "Cache evict hot for cold: " << hr->lru_cache->evict_hot_for_cold << std::endl;
    std::cout << "Window size: " << window_size << std::endl;
    std::cout << "Learning rate: " << hr->learning_rate << std::endl;
    std::cout << "Features length: " << hr->request_window->features_length << std::endl;
    std::cout << "Size feature: " << hr->request_window->features.at(FEAT_SIZE) << std::endl;
    std::cout << "Frequency feature: " << hr->request_window->features.at(FEAT_FREQUENCY) << std::endl;
    std::cout << "Decayed frequency feature: " << hr->objects_metadata->decay_factor << std::endl;
    std::cout << "Hazard bandwidth: " << hr->hazard_bandwidth << std::endl;
    std::cout << "Hazard discrete: " << hr->hazard_discrete << std::endl;
    std::cout << "Future labeling: " << hr->future_labeling << std::endl;
    std::cout << "One time training: " << hr->one_time_training << std::endl;
    std::cout << "Max boost rounds: " << hr->model->max_boost_round << std::endl;
    std::cout << "Report interval: " << hr->report_interval << std::endl;
    std::cout << "------------------------" << std::endl;
}

void log_analytics(HRCache* hr, bool last_log) {

    if (hr->analytics_bytes != 0 && hr->analytics_reqs != 0) {
        hr->analytics_round++;
        std::ofstream outfile("/Users/kjiyun/Desktop/HR-Cache/HR-Cache/hr_cache_result.txt");
        double miss_bytes_percentage = 100 - 100.0 * hr->analytics_bytes_hit / hr->analytics_bytes;
        double miss_percentage = 100 - 100.0 * hr->analytics_reqs_hit / hr->analytics_reqs;
        double avg_time = hr->analytics_times / hr->analytics_reqs;
        double avg_cpu_time = hr->analytics_cpu_times / hr->analytics_reqs;
        double cumulative_miss_bytes_percentage = 0, cumulative_miss_percentage = 0, cumulative_avg_time = 0, cumulative_avg_cpu_time = 0, cumulative_potential_reqs_per_sec = 0;
        double potential_reqs_per_sec = 1e9 / avg_time;
        if (hr->cumulative_reqs > 0) {
            cumulative_miss_bytes_percentage = 100 - 100.0 * hr->cumulative_bytes_hit / hr->cumulative_bytes;
            cumulative_miss_percentage = 100 - 100.0 * hr->cumulative_reqs_hits / hr->cumulative_reqs;
            cumulative_avg_time = hr->cumulative_times / hr->cumulative_reqs;
            cumulative_avg_cpu_time = hr->cumulative_cpu_times / hr->cumulative_reqs;
            cumulative_potential_reqs_per_sec = 1e9 / cumulative_avg_time;
        }

        std::cout << std::setprecision(5);
        std::cout << "Round: " << hr->analytics_round << std::endl;
        std::cout << "Bytes miss: " << miss_bytes_percentage << "%" << std::endl;
        std::cout << "Reqs miss: " << miss_percentage << "%" << std::endl;
        std::cout << "Total bytes miss: " << cumulative_miss_bytes_percentage << "%" << std::endl;
        std::cout << "Total reqs miss: " << cumulative_miss_percentage << "%" << std::endl;
        std::cout << "Bytes: " << hr->analytics_bytes << std::endl;
        std::cout << "Total bytes: " << hr->cumulative_bytes << std::endl;
        std::cout << "Avg time: " << avg_time << " ns" << std::endl;
        std::cout << "Avg CPU time: " << avg_cpu_time << " ns" << std::endl;
        std::cout << "Total avg time: " << cumulative_avg_time << " ns" << std::endl;
        std::cout << "Total avg CPU time: " << cumulative_avg_cpu_time << " ns" << std::endl;
        std::cout << "reqs/s: " << static_cast<long long>(potential_reqs_per_sec) << std::endl;
        std::cout << "Total reqs/s: " << static_cast<long long>(cumulative_potential_reqs_per_sec) << std::endl;
        report_memory();
        std::cout << "Hot evictions count percentage: " << 100.0 * hr->analytics_hot_evicted_reqs / (hr->analytics_hot_evicted_reqs + hr->analytics_cold_evicted_reqs) << "%" << std::endl;
        std::cout << "Hot evictions bytes percentage: " << 100.0 * hr->analytics_hot_evicted_bytes / (hr->analytics_hot_evicted_bytes + hr->analytics_cold_evicted_bytes) << "%" << std::endl;
        std::cout << "------------------------" << std::endl;

        if (outfile.fail()) {
            std::cerr << "Failed to open hr_cache_result.txt for writing." << std::endl;
            return;
        }

        outfile << std::setprecision(5);
        outfile << "MissRate: " << miss_percentage << std::endl;
        outfile << "HitRate: " << (100.0 - miss_percentage) << std::endl;
        outfile << "MemoryMB: " << get_current_memory_usage() << std::endl;
        outfile << "ReqsPerSec: " << static_cast<long long>(potential_reqs_per_sec) << std::endl;
        outfile.close();

        if (hr->log_file) {
            int window_size = hr->request_window->size ? *hr->request_window->size : 0;
            hr->analytics_file << hr->key << "," << hr->lru_cache->capacity << "," << hr->lru_cache->hot_lower_bound << ",";
            hr->analytics_file << hr->lru_cache->cold_lower_bound << "," << hr->lru_cache->evict_hot_for_cold << ",";
            hr->analytics_file << window_size << "," << hr->learning_rate << "," << hr->request_window->features_length << ",";
            hr->analytics_file << hr->request_window->features.at(FEAT_SIZE) << "," << hr->request_window->features.at(FEAT_FREQUENCY) << ",";
            hr->analytics_file << hr->objects_metadata->decay_factor << "," << hr->hazard_bandwidth << ",";
            hr->analytics_file << hr->hazard_discrete << "," << hr->future_labeling << ",";
            hr->analytics_file << hr->one_time_training << "," << hr->model->max_boost_round << "," << hr->report_interval << ",";
            hr->analytics_file << hr->analytics_round << "," << miss_bytes_percentage << "," << miss_percentage << ",";
            hr->analytics_file << cumulative_miss_bytes_percentage << "," << cumulative_miss_percentage << std::endl;
        }
    }

    if (last_log) {
        std::cout << "Without training requests count: " << hr->without_training_count << std::endl;
        std::cout << "Hot evictions count percentage: " << 100.0 * hr->cumulative_hot_evicted_reqs / (hr->cumulative_hot_evicted_reqs + hr->cumulative_cold_evicted_reqs) << "%" << std::endl;
        std::cout << "Hot evictions bytes percentage: " << 100.0 * hr->cumulative_hot_evicted_bytes / (hr->cumulative_hot_evicted_bytes + hr->cumulative_cold_evicted_bytes) << "%" << std::endl;
        std::cout << "------------------------" << std::endl;
    }
}

void close_files(HRCache* hr) {
    if (hr->log_requests) {
        hr->requests_file.close();
    }

    if (hr->log_file) {
        hr->analytics_file << ",,,,,,,,,,,,,,,,,,,,," << std::endl;
        hr->analytics_file.close();
    }
}

void update_analytics(HRCache* hr, bool cache_hit, int size, bool predicted) {
    hr->requests_count++;

    if (predicted && hr->requests_count % hr->report_interval == 1) {
        hr->start_counting_cumulative = true;
    }
    
    if (hr->start_counting_cumulative && predicted) {
        hr->cumulative_reqs++;
        hr->cumulative_bytes += size;
    } else {
        hr->without_training_count++;
    }

    hr->analytics_reqs++;
    hr->analytics_bytes += size;
    if (cache_hit) {
        if (hr->start_counting_cumulative && predicted) {
            hr->cumulative_reqs_hits++;
            hr->cumulative_bytes_hit += size;
        }

        hr->analytics_reqs_hit++;
        hr->analytics_bytes_hit += size;
    }

    if (hr->requests_count % hr->report_interval == 0) {
        log_analytics(hr, false);
        hr->analytics_bytes_hit = 0;
        hr->analytics_reqs_hit = 0;
        hr->analytics_reqs = 0;
        hr->analytics_bytes = 0;
        hr->analytics_times = 0;
        hr->analytics_cpu_times = 0;
        hr->analytics_hot_evicted_bytes = 0;
        hr->analytics_hot_evicted_reqs = 0;
        hr->analytics_cold_evicted_bytes = 0;
        hr->analytics_cold_evicted_reqs = 0;
    }
}

void sync_requests(HRCache* hr) {
    if (hr->request_window->requests_count <= 1) {
        return;
    }

    if (hr->last_processed_request == hr->request_window->request->prev_in_time) {
        return;
    }

    if (!hr->last_processed_request) {
        hr->last_processed_request = hr->request_window->request;
    } else {
        hr->last_processed_request = hr->last_processed_request->next_in_time;
    }

    HR_Request** requests = new HR_Request*[hr->concurrency];
    int requests_count = 0;
    while (true) {
        requests[requests_count++] = hr->last_processed_request;

        if (hr->last_processed_request->next_in_time == hr->request_window->request) {
            break;
        }
        hr->last_processed_request = hr->last_processed_request->next_in_time;
    }

    if (hr->model->available) {
        predict_requests(hr->model, requests, requests_count);
    }

    for (int i = 0; i < requests_count; i++) {
        lookup_and_admit(hr->lru_cache, requests[i]);
    }

    delete[] requests;
}

void update_model(HRCache* hr, bool wait_for_model) {
    if (hr->model_thread.joinable()) {
        hr->model_thread.join();
    }
    
    HR_RequestWindow* old_request_window = hr->request_window;
    update_default_features(old_request_window);

    hr->request_window = create_request_window(
        old_request_window->size,
        old_request_window->cache_size,
        old_request_window->features_length,
        old_request_window->features,
        old_request_window->objects_metadata
    );

    hr->model_thread = std::thread([hr, old_request_window]() {
        if ((hr->model->row_count == 0 && !hr->model->full) || !hr->one_time_training) {
            prepare_request_window(
                old_request_window,
                hr->model->max_train_set_count / 2,
                hr->hazard_bandwidth,
                hr->hazard_discrete,
                hr->future_labeling,
                hr->verbose
            );
            update_hr_model(
                hr->model,
                old_request_window->sampled_requests,
                old_request_window->sampled_requests_count,
                hr->verbose
            );
        }

        // TODO: takes too much time on huge windows
        destroy_request_window(old_request_window);
    });

    if (wait_for_model) {
        hr->model_thread.join();
    }
}

bool new_request(HRCache* hr, double timestamp, int object_id, int size) {
    std::cout << "[DEBUG] new_request 호출, objects_metadata 포인터: " 
          << hr->objects_metadata << std::endl;
    std::clock_t cpu_start = std::clock();
    auto start = std::chrono::high_resolution_clock::now();

    HR_Request* request = add_request(hr->request_window, object_id, timestamp, size);
    HR_LookupAdmitResult result = lookup_and_admit(hr->lru_cache, request);
    update_analytics(hr, result.hit, size, hr->model->available);

    if (result.hot_evictions_count > 0) {
        hr->cumulative_hot_evicted_bytes += result.hot_evictions_bytes / 1e6;
        hr->cumulative_hot_evicted_reqs += result.hot_evictions_count;
        hr->analytics_hot_evicted_bytes += result.hot_evictions_bytes / 1e6;
        hr->analytics_hot_evicted_reqs += result.hot_evictions_count;
    }
    if (result.cold_evictions_count > 0) {
        hr->cumulative_cold_evicted_bytes += result.cold_evictions_bytes / 1e6;
        hr->cumulative_cold_evicted_reqs += result.cold_evictions_count;
        hr->analytics_cold_evicted_bytes += result.cold_evictions_bytes / 1e6;
        hr->analytics_cold_evicted_reqs += result.cold_evictions_count;
    }
    
    if (window_is_ready(hr->request_window, 1 / hr->learning_rate)) {
        // cleanup_expired_hot(hr->lru_cache, timestamp - (12 * 60 * 60));
        sync_requests(hr);
        hr->last_processed_request = NULL;
        update_model(hr, true);
    }

    if (hr->request_window->requests_count % hr->concurrency == 0) {
        sync_requests(hr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed = end - start;
    hr->analytics_times += elapsed.count();
    std::clock_t cpu_end = std::clock();
    double cpu_time_in_seconds = static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    double cpu_elapsed = 1e9 * cpu_time_in_seconds;
    hr->analytics_cpu_times += cpu_elapsed;

    // if (hr->model->available) {
    if (true) {
        double prob = request->admit_probability;
        double base_ttl = 60.0;
        double ttl_seconds = 1.0 * prob; 
        hr->cumulative_cpu_times += cpu_elapsed;
        hr->cumulative_times += elapsed.count();
        hr->objects_metadata->set_ttl_for_object(object_id, ttl_seconds);
    }

    return result.admitted;
}

void destroy_hr(HRCache* hr) {
    if (hr->model_thread.joinable()) {
        hr->model_thread.join();
    }

    if (hr->lru_cache) {
        destroy_lru_cache(hr->lru_cache);
    }
    if (hr->request_window) {
        destroy_request_window(hr->request_window);
    }
    if (hr->model) {
        destroy_hr_model(hr->model);
    }
    if (hr->objects_metadata) {
        delete hr->objects_metadata;
    }

    close_files(hr);
    delete hr;
}