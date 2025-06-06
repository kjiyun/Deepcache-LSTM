// metadata.cpp

#include "metadata.h"
#include <cmath>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <mutex>
#include <unordered_map>

std::unordered_map<int, double> g_insert_time_map;

// 기존 생성자/소멸자 구현
HR_ObjectsMetadata::HR_ObjectsMetadata(int capacity, int features_length, double decay_factor)
    : max_objects_(capacity),
        features_length(features_length),
        decay_factor(decay_factor) 
{
        this->features_length = features_length;
        this->decay_factor = decay_factor;
        int object_meta_size = sizeof(HR_ObjectMetadata) + sizeof(double) * features_length + sizeof(HR_ObjectLastSeen);
        this->max_objects_count = MINIMUM_OBJECTS_COUNT + capacity / object_meta_size;
        this->decayed_frequency = 0;
}

HR_ObjectsMetadata::~HR_ObjectsMetadata() {

    for (auto const& [object_id, object_metadata] : objects) {
            // delete[] object_metadata->features;
            // delete object_metadata->last_seen;
            delete object_metadata;
        }
    objects.clear();
}

HR_ObjectMetadata* HR_ObjectsMetadata::get_metadata(int object_id, int timestamp) {
    auto it = objects.find(object_id);
    if (it == objects.end()) {
        HR_ObjectMetadata* object_metadata = new HR_ObjectMetadata;
        object_metadata->decayed_frequency = 0;

        // features 배열 할당하고, INF로 초기화
        object_metadata->features = new double[features_length];
        std::fill(
            object_metadata->features,
            object_metadata->features + features_length,
            INF
        );

        // last_seen 구조체 할당하고 필드 설정
        object_metadata->last_seen = new HR_ObjectLastSeen{ object_id, (double)timestamp };

        // 맵과 set에 삽입
        objects[object_id] = object_metadata;
        //objects_last_seen.insert(object_metadata->last_seen);

        return object_metadata;
    }
    return it->second;
}

void HR_ObjectsMetadata::update_features(int object_id, double* features) {
    auto it = objects.find(object_id);
    if (it == objects.end()) return;

    HR_ObjectMetadata* object_metadata = it->second;
    std::memcpy(
        object_metadata->features,
        features,
        sizeof(double) * features_length
    );
}

double* HR_ObjectsMetadata::get_features(int object_id) {
    auto it = objects.find(object_id);
    if (it == objects.end()) return nullptr;
    return it->second->features;
}

double HR_ObjectsMetadata::get_decayed_frequency(int object_id) {
    auto it = objects.find(object_id);
    if (it == objects.end()) return 0.0;
    return it->second->decayed_frequency / decayed_frequency;
}

void HR_ObjectsMetadata::seen(int object_id, double timestamp) {
    HR_ObjectMetadata* object_metadata = get_metadata(object_id, timestamp);

    // 전체 decayed_frequency, 개별 decayed_frequency 갱신
    decayed_frequency = decayed_frequency * decay_factor + 1;
    object_metadata->decayed_frequency = object_metadata->decayed_frequency * decay_factor + 1;
}

void HR_ObjectsMetadata::set_ttl_for_object(int object_id, double unused_ttl) {
    // 1) 현재 시각 얻기
    double now_ts = static_cast<double>(time(nullptr));

    // 2) 해저드율 예측
    double lambda = predict_hazard_rate(object_id, now_ts);
    //    λ(t)가 0이 되면 안 되므로, 최소값 보장
    if (lambda <= 0.0) {
        lambda = 0.01; 
    }

    // 3) TTL 계산
    //    보통 해저드율 λ(t)를 “단위시간당 재요청 확률”로 본다면,
    //    기대 재요청 간격(평균 시간) ≈ 1 / λ(t)
    //    → 이 값을 TTL로 사용
    double ttl_seconds = 1.0 / lambda;

    {
        // 4) 뮤텍스를 잡고 맵 갱신
        std::lock_guard<std::mutex> guard(ttl_mutex_);
        object_ttl_map_[object_id] = ttl_seconds;

        // 전역 삽입 시간 맵 업데이트
        extern std::unordered_map<int, double> g_insert_time_map;
        g_insert_time_map[object_id] = now_ts;
    }

    // debug 로그
    std::cout << "[Metadata] object " << object_id 
              << " predicted λ=" << lambda 
              << ", set TTL=" << ttl_seconds << "s\n";
}

double HR_ObjectsMetadata::get_ttl_for_object(int object_id) const {
    std::lock_guard<std::mutex> lock(ttl_mutex_);
    auto it = object_ttl_map_.find(object_id);
    if (it == object_ttl_map_.end()) {
        return 0.0;
    }
    return it->second;
}

bool HR_ObjectsMetadata::is_expired(int object_id, double now_timestamp) const {
    std::lock_guard<std::mutex> lock(ttl_mutex_);

    auto it_ttl = object_ttl_map_.find(object_id);
    if (it_ttl == object_ttl_map_.end()) {
        // TTL 정보가 없으면, “만료되지 않음”으로 간주
        return false;
    }
    double ttl_seconds = it_ttl->second;

    // 삽입 시각을 기록한 데이터가 있어야 비교 가능하지만,
    // 여기서는 예시로 ‘insert_time’ 맵이 있다고 가정합니다.
    // 만약 HR_ObjectsMetadata 내부에 insert_time_map_가 없다면, 
    // 삽입 시각을 저장하는 로직을 별도로 추가해야 합니다.

    extern std::unordered_map<int, double> g_insert_time_map;
    auto it_insert = g_insert_time_map.find(object_id);
    if (it_insert == g_insert_time_map.end()) {
        return false;
    }

    double insert_time = it_insert->second;
    return (now_timestamp >= insert_time + ttl_seconds);
}

double HR_ObjectsMetadata::predict_hazard_rate(int object_id, double now_timestamp) const {
    
    // 처음 들어온 객체인 경우 기본 5초 ttl
    if (object_ttl_map_.count(object_id) == 0) {
        return 5.0;
    }

    // 이미 존재한다면 이전 ttl의 80% 만큼 줄여서 반환
    double prev_ttl = object_ttl_map_.at(object_id);
    double new_ttl = prev_ttl * 0.8;
    if (new_ttl < 1.0) {
        new_ttl = 1.0;  // TTL이 너무 작아지면 최소 1초는 보장
    }
    return new_ttl;
}