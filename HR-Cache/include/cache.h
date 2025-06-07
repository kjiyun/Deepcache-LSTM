#ifndef HR_CACHE_H
#define HR_CACHE_H

#include "requests.h"
#include <unordered_map>

typedef enum {
    HOT = 0,
    COLD = 1
} HR_CacheNodeMode;

struct HR_CacheNode {
    HR_CacheNodeMode mode;
    int id;
    int size;
    double last_seen;
    struct HR_CacheNode* prev;
    struct HR_CacheNode* next;
};

struct HR_Cache {
    HR_CacheNode* hot_cache;
    HR_CacheNode* cold_cache;
    std::unordered_map<int, HR_CacheNode*> lookup_table;
    long long capacity;
    long long current_size;
    long long current_hot_size;
    long long current_cold_size;
    double hot_lower_bound;
    double cold_lower_bound;
    bool evict_hot_for_cold;
};

struct HR_LookupAdmitResult {
    bool admitted;
    bool hit;
    int hot_evictions_count;
    int cold_evictions_count;
    int hot_evictions_bytes;
    int cold_evictions_bytes;
};

HR_Cache* create_lru_cache(long long capacity, double hot_lower_bound, double cold_lower_bound, bool evict_hot_for_cold);
HR_CacheNode* lookup_without_move(HR_Cache* cache, int request_id);
HR_CacheNode* lookup(HR_Cache* cache, HR_Request* request);
HR_LookupAdmitResult lookup_and_admit(HR_Cache* cache, HR_Request* request);
int cleanup_cache(HR_Cache* cache, double last_seen_threshold);
int cleanup_expired_hot(HR_Cache* cache, double last_seen_threshold);

void destroy_lru_cache(HR_Cache* cache);

#endif // HR_CACHE_H