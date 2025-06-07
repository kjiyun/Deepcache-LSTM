#ifndef HR_H
#define HR_H

#include "requests.h"
#include "cache.h"
#include "model.h"
#include <unordered_map>
#include <optional>
#include <thread>
#include <fstream>

struct HRCache {
    std::string key;
    bool verbose;
    int concurrency;

    HR_ObjectsMetadata* objects_metadata;
    HR_Cache* lru_cache;
    HR_RequestWindow* request_window;
    HR_Model* model;
    double learning_rate;
    double hazard_bandwidth;
    bool hazard_discrete;
    bool future_labeling;
    bool one_time_training;
    int report_interval;
    bool log_file;
    bool log_requests;
    int requests_count;

    double cumulative_hot_evicted_bytes;
    double cumulative_hot_evicted_reqs;
    double cumulative_cold_evicted_bytes;
    double cumulative_cold_evicted_reqs;
    double analytics_hot_evicted_bytes;
    double analytics_hot_evicted_reqs;
    double analytics_cold_evicted_bytes;
    double analytics_cold_evicted_reqs;

    bool start_counting_cumulative;
    int without_training_count;
    int cumulative_reqs;
    int cumulative_reqs_hits;
    double cumulative_times;
    double cumulative_cpu_times;
    long long cumulative_bytes;
    long long cumulative_bytes_hit;
    int analytics_reqs;
    int analytics_reqs_hit;
    double analytics_times;
    double analytics_cpu_times;
    long long analytics_bytes;
    long long analytics_bytes_hit;
    long long analytics_round;

    std::thread model_thread;
    std::ofstream requests_file;
    std::ofstream analytics_file;
    HR_Request* last_processed_request;
};

HRCache* create_hr(
    std::string key,
    std::optional<int> concurrency=std::nullopt,
    std::optional<bool> verbose=std::nullopt,
    std::optional<long long> cache_size=std::nullopt,
    std::optional<double> cache_hot_lower_bound=std::nullopt,
    std::optional<double> cache_cold_lower_bound=std::nullopt,
    std::optional<bool> cache_evict_hot_for_cold=std::nullopt,
    int* window_size=NULL,
    std::optional<double> learning_rate=std::nullopt,
    std::optional<int> features_length=std::nullopt,
    std::optional<double> decay_factor=std::nullopt,
    std::optional<double> hazard_bandwidth=std::nullopt,
    std::optional<bool> hazard_discrete=std::nullopt,
    std::optional<bool> future_labeling=std::nullopt,
    std::optional<bool> one_time_training=std::nullopt,
    std::optional<int> max_boost_rounds=std::nullopt,
    std::optional<std::unordered_map<HR_FEATURE, bool>> features=std::nullopt,
    std::optional<int> report_interval=std::nullopt,
    std::optional<bool> log_file=std::nullopt,
    std::optional<bool> log_requests=std::nullopt,
    std::optional<std::string> log_file_name=std::nullopt
);
void log_args(HRCache* hr);
void log_analytics(HRCache* hr, bool log_without_training);
bool new_request(HRCache* hr, double timestamp, int object_id, int size);

void destroy_hr(HRCache* hr);

#endif // HR_H