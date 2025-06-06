#include "hr.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

int simulate(
    std::string file_path,
    std::optional<int> concurrency=std::nullopt,
    std::optional<bool> verbose=std::nullopt,
    std::optional<long long> cache_size=std::nullopt,
    std::optional<double> hot_lower_bound=std::nullopt,
    std::optional<double> cold_lower_bound=std::nullopt,
    std::optional<bool> evict_hot_for_cold=std::nullopt,
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
    std::optional<std::string> log_file_name=std::nullopt
) {
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "Unable to open file: " << file_path << std::endl;
        return 1;  // Return with error
    }

    HRCache* hr = create_hr(
        file_path,
        concurrency,
        verbose,
        cache_size,
        hot_lower_bound,
        cold_lower_bound,
        evict_hot_for_cold,
        window_size,
        learning_rate,
        features_length,
        decay_factor,
        hazard_bandwidth,
        hazard_discrete,
        future_labeling,
        one_time_training,
        max_boost_rounds,
        features,
        report_interval,
        log_file,
        false,
        log_file_name
    );
    log_args(hr);

    std::string line;
    while (getline(file, line)) {
        std::istringstream iss(line);
        double timestamp;
        int object_id, size;
        if (!(iss >> timestamp >> object_id >> size)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            break;
        }

        new_request(hr, timestamp, object_id, size);
    }
    log_analytics(hr, true);
    destroy_hr(hr);

    file.close();
    return 0;
}

int main(int argc, char* argv[]) {
    std::string file_path;
    int rounds = 1;
    int* window_size = NULL;
    std::optional<std::string> log_file_name;
    std::optional<long long> cache_size;
    std::optional<int> concurrency, features_length, report_interval, max_boost_rounds;
    std::optional<double> learning_rate, hot_lower_bound, cold_lower_bound, hazard_bandwidth, decay_factor;
    std::optional<bool> verbose, evict_hot_for_cold, hazard_discrete, future_labeling, one_time_training, 
        feature_frequency, feature_size;
    bool with_features = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--file-path=") == 0) {
            file_path = arg.substr(strlen("--file-path="));
        }
        if (arg.find("--concurrency=") == 0) {
            concurrency = stoi(arg.substr(strlen("--concurrency=")));
        }
        if (arg.find("--verbose") == 0) {
            if (arg.find("--verbose=") == 0) {
                verbose = arg.substr(strlen("--verbose=")) == "true";
            } else {
                verbose = true;
            }
        }
        if (arg.find("--rounds=") == 0) {
            rounds = stoi(arg.substr(strlen("--rounds=")));
        }
        if (arg.find("--cache-size=") == 0) {
            cache_size = stoll(arg.substr(strlen("--cache-size=")));
        }
        if (arg.find("--hot-lower-bound=") == 0) {
            hot_lower_bound = stod(arg.substr(strlen("--hot-lower-bound=")));
        }
        if (arg.find("--cold-lower-bound=") == 0) {
            cold_lower_bound = stod(arg.substr(strlen("--cold-lower-bound=")));
        }
        if (arg.find("--evict-hot-for-cold") == 0) {
            if (arg.find("--evict-hot-for-cold=") == 0) {
                evict_hot_for_cold = arg.substr(strlen("--evict-hot-for-cold=")) == "true";
            } else {
                evict_hot_for_cold = true;
            }
        }
        if (arg.find("--window-size=") == 0) {
            window_size = new int(stoi(arg.substr(strlen("--window-size="))));
        }
        if (arg.find("--learning-rate=") == 0) {
            learning_rate = stod(arg.substr(strlen("--learning-rate=")));
        }
        if (arg.find("--features-length=") == 0) {
            features_length = stoi(arg.substr(strlen("--features-length=")));
        }
        if (arg.find("--hazard-bandwidth=") == 0) {
            hazard_bandwidth = stod(arg.substr(strlen("--hazard-bandwidth=")));
        }
        if (arg.find("--hazard-discrete") == 0) {
            if (arg.find("--hazard-discrete=") == 0) {
                hazard_discrete = arg.substr(strlen("--hazard-discrete=")) == "true";
            } else {
                hazard_discrete = true;
            }
        }
        if (arg.find("--future-labeling") == 0) {
            if (arg.find("--future-labeling=") == 0) {
                future_labeling = arg.substr(strlen("--future-labeling=")) == "true";
            } else {
                future_labeling = true;
            }
        }
        if (arg.find("--one-time-training") == 0) {
            if (arg.find("--one-time-training=") == 0) {
                one_time_training = arg.substr(strlen("--one-time-training=")) == "true";
            } else {
                one_time_training = true;
            }
        }
        if (arg.find("--max-boost-rounds=") == 0) {
            max_boost_rounds = stoi(arg.substr(strlen("--max-boost-rounds=")));
        }
        if (arg.find("--feature-frequency") == 0) {
            with_features = true;
            if (arg.find("--feature-frequency=") == 0) {
                feature_frequency = arg.substr(strlen("--feature-frequency=")) == "true";
            } else {
                feature_frequency = true;
            }
        }
        if (arg.find("--feature-decayed-frequency=") == 0) {
            with_features = true;
            decay_factor = stod(arg.substr(strlen("--feature-decayed-frequency=")));
        }
        if (arg.find("--feature-size") == 0) {
            with_features = true;
            if (arg.find("--feature-size=") == 0) {
                feature_size = arg.substr(strlen("--feature-size=")) == "true";
            } else {
                feature_size = true;
            }
        }
        if (arg.find("--report-interval=") == 0) {
            report_interval = stoi(arg.substr(strlen("--report-interval=")));
        }
        if (arg.find("--log-file=") == 0) {
            log_file_name = arg.substr(strlen("--log-file="));
        }
    }

    if (!file_path.empty()) {
        std::cout << "File path: " << file_path << std::endl;
    } else {
        std::cout << "No file path provided" << std::endl;
        return 1;
    }

    std::optional<std::unordered_map<HR_FEATURE, bool>> features;
    if (with_features) {
        features = std::unordered_map<HR_FEATURE, bool>({
            {FEAT_FREQUENCY, feature_frequency.value_or(false)},
            {FEAT_SIZE, feature_size.value_or(false)},
            {FEAT_DECAYED_FREQUENCY, decay_factor.value_or(0) ? true : false}
        });
    }

    for (int i = 0; i < rounds; ++i) {
        std::cout << "------------------------ Simulate Round " << i + 1 << " ------------------------" << std::endl;
        int has_error = simulate(
            file_path,
            concurrency,
            verbose,
            cache_size,
            hot_lower_bound,
            cold_lower_bound,
            evict_hot_for_cold,
            window_size,
            learning_rate,
            features_length,
            decay_factor,
            hazard_bandwidth,
            hazard_discrete,
            future_labeling,
            one_time_training,
            max_boost_rounds,
            features,
            report_interval,
            log_file_name.value_or("") != "",
            log_file_name
        );
        if (has_error) {
            return has_error;
        }
    }
    return 0;
}