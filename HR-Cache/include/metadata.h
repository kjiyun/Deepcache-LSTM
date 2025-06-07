#pragma once
#ifndef HR_METADATA_H
#define HR_METADATA_H

#include <mutex>
#include <unordered_map>
#include <set>
#include <string.h>

const double INF = 100.0 * 1000 * 1000;
const int MINIMUM_OBJECTS_COUNT = 100 * 1000 * 1000;

struct HR_ObjectLastSeen {
    int object_id;
    double timestamp;
};

struct HR_ObjectLastSeenCompare {
    bool operator()(const HR_ObjectLastSeen* lhs, const HR_ObjectLastSeen* rhs) const {
        if (lhs->timestamp == rhs->timestamp) {
            return lhs->object_id < rhs->object_id;
        }

        return lhs->timestamp < rhs->timestamp;
    }
};

struct HR_ObjectMetadata {
    double decayed_frequency;
    double* features;
    HR_ObjectLastSeen* last_seen;

    // 소멸자: 자신이 new[]/new 로 할당한 메모리만 해제
    ~HR_ObjectMetadata() {
        delete[] features;
        delete  last_seen;
    }
};

class HR_ObjectsMetadata {
public:
    double decay_factor;
    int max_objects_;
    int features_length;

    // 생성자·소멸자 선언
    HR_ObjectsMetadata(int capacity, int features_length, double decay_factor);
    ~HR_ObjectsMetadata();

    HR_ObjectMetadata* get_metadata(int object_id, int timestamp = 0);
    void               update_features(int object_id, double* features);
    double*            get_features(int object_id);
    double             get_decayed_frequency(int object_id);
    void               seen(int object_id, double timestamp);

    // 멤버 함수 선언
    void set_ttl_for_object(int object_id, double unused_ttl);
    double get_ttl_for_object(int object_id) const;
    bool is_expired(int object_id, double now_timestamp) const;
    double predict_hazard_rate(int object_id, double now_timestamp) const;

private:
    mutable std::mutex ttl_mutex_;
    std::unordered_map<int, double> object_ttl_map_;
    std::unordered_map<int, HR_ObjectMetadata*> objects;

    int max_objects_count;
    double decayed_frequency;
};

#endif