#ifndef HR_REQUESTS_H
#define HR_REQUESTS_H

#include "metadata.h"
#include <unordered_map>
#include <set>

enum HR_FEATURE {
    FEAT_FREQUENCY = 0,
    FEAT_SIZE = 1,
    FEAT_DECAYED_FREQUENCY = 2
};

struct HR_Request {
    int object_id;
    double timestamp;
    int size;
    double admit_probability;
    int label;
    double *features;
    HR_Request *next;
    HR_Request *prev;
    HR_Request *next_in_time;
    HR_Request *prev_in_time;
};

struct Object {
    int idx;
    int id;
    int size;
    HR_Request *request;
    int requests_count;                     // number of requests
    
    bool sampled;                           // whether the object is sampled
    double *timestamps;                     // timestamps of the requests in the window
    double *timestamps_diffs;               // intervals between timestamps
    double *cumulative_hazards_diffs;       // cumulative hazards for each interval
    double hazard_bandwidth;                // bandwidth for the hazard estimation
    int diffs_count;                        // number of intervals
    std::unordered_map<double, double> hazards;  // hazards for each timestamp diff
};

struct HR_RequestWindow {
    int *size;
    long long cache_size;
    int requests_count;
    HR_Request *request;
    int objects_count;
    long long objects_size;
    std::unordered_map<int, Object*> objects;
    HR_ObjectsMetadata* objects_metadata;

    double avg_req_size;
    double sample_rate;
    int sampled_requests_count;
    int custom_features_count;
    int features_length;
    std::unordered_map<HR_FEATURE, bool> features;
    HR_Request **sampled_requests;
};

HR_RequestWindow* create_request_window(
    int *size,
    long long cache_size,
    int features_length,
    std::unordered_map<HR_FEATURE, bool> features,
    HR_ObjectsMetadata* objects_metadata
);
void update_default_features(HR_RequestWindow* request_window);
Object* get_object(HR_RequestWindow* request_window, const int object_id, int size);
HR_Request* add_request(HR_RequestWindow* request_window, int object_id, double timestamp, int size);
bool window_is_ready(HR_RequestWindow* request_window, double weight=1);
void prepare_request_window(HR_RequestWindow* request_window, int max_requests_count, double bandwidth, bool discrete, bool future_labeling, bool verbose=false);

void destroy_request_window(HR_RequestWindow* request_window);

#endif // HR_REQUESTS_H