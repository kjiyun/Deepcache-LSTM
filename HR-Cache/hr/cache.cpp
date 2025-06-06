#include "requests.h"
#include "cache.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>

HR_Cache* create_lru_cache(long long capacity, double hot_lower_bound, double cold_lower_bound, bool evict_hot_for_cold) {
    HR_Cache* cache = new HR_Cache;
    cache->hot_cache = NULL;
    cache->cold_cache = NULL;
    cache->capacity = capacity;
    cache->current_size = 0;
    cache->current_hot_size = 0;
    cache->current_cold_size = 0;
    cache->hot_lower_bound = hot_lower_bound;
    cache->cold_lower_bound = cold_lower_bound;
    cache->evict_hot_for_cold = evict_hot_for_cold;
    return cache;
}

HR_CacheNode* create_node(int id, int size, double timestamp) {
    HR_CacheNode* node = (HR_CacheNode*)malloc(sizeof(HR_CacheNode));
    node->id = id;
    node->size = size;
    node->last_seen = timestamp;
    node->prev = NULL;
    node->next = NULL;
    return node;
}

HR_CacheNode* move_node_to_end(HR_CacheNode* head, HR_CacheNode* node) {
    // if node is NULL, return head
    if (!node) {
        return head;
    }

    if (!head) {
        node->next = node;
        node->prev = node;
        return node;
    }

    // node is already at the end
    if (head->prev->id == node->id) {
        return head;
    }

    // node is the head
    if (head->id == node->id) {
        return head->next;
    }

    // connect node's prev and next to each other
    if (node->next) {
        node->next->prev = node->prev;
    }
    if (node->prev) {
        node->prev->next = node->next;
    }
    
    HR_CacheNode* end = head->prev;
    end->next = node;
    node->prev = end;
    node->next = head;
    head->prev = node;
    return head;
}

HR_CacheNode* remove_node(HR_CacheNode* head, HR_CacheNode* node) {
    // if head or node is NULL, return
    if (!head || !node) {
        return head;
    }

    // node is the head
    if (head->id == node->id) {
        HR_CacheNode* new_head = head->next;
        if (new_head->id == node->id) {
            return NULL;
        }

        new_head->prev = head->prev;
        head->prev->next = new_head;
        return new_head;
    }

    node->next->prev = node->prev;
    node->prev->next = node->next;
    return head;
}

void evict(HR_Cache* cache, HR_LookupAdmitResult* result) {
    HR_CacheNode* node;
    if (cache->cold_cache) {
        node = cache->cold_cache;
        cache->cold_cache = remove_node(node, node);
    } else if (cache->hot_cache) {
        node = cache->hot_cache;
        cache->hot_cache = remove_node(node, node);
    }

    cache->current_size -= node->size;
    if (node->mode == HOT) {
        cache->current_hot_size -= node->size;
        result->hot_evictions_count++;
        result->hot_evictions_bytes += node->size;
    } else {
        cache->current_cold_size -= node->size;
        result->cold_evictions_count++;
        result->cold_evictions_bytes += node->size;
    }

    node->next = NULL;
    node->prev = NULL;
    cache->lookup_table.erase(node->id);
    delete node;
}

void admit(HR_Cache* cache, HR_Request* request, HR_LookupAdmitResult* result) {
    if (request->size > cache->capacity) {
        return;
    }

    if (request->admit_probability < cache->cold_lower_bound) {
        return;
    }

    if (
        !cache->evict_hot_for_cold && 
        request->admit_probability < cache->hot_lower_bound && 
        cache->current_hot_size + request->size > cache->capacity
    ) {
        return;
    }

    result->admitted = true;
    HR_CacheNode* node = create_node(request->object_id, request->size, request->timestamp);
    while (cache->current_size + node->size > cache->capacity) {
        evict(cache, result);
    }

    cache->lookup_table[node->id] = node;
    if (request->admit_probability >= cache->hot_lower_bound) {
        node->mode = HOT;
        cache->hot_cache = move_node_to_end(cache->hot_cache, node);
        cache->current_hot_size += node->size;
    } else {
        node->mode = COLD;
        cache->cold_cache = move_node_to_end(cache->cold_cache, node);
        cache->current_cold_size += node->size;
    }
    cache->current_size += node->size;
}

HR_CacheNode* lookup_without_move(HR_Cache* cache, int request_id) {
    return cache->lookup_table[request_id];
}

HR_CacheNode* lookup(HR_Cache* cache, HR_Request* request) {
    HR_CacheNode *node = lookup_without_move(cache, request->object_id);
    if (!node) {
        return NULL;
    }

    if (node->mode == HOT && request->admit_probability >= cache->hot_lower_bound) {
        node->last_seen = request->timestamp;
        cache->hot_cache = move_node_to_end(cache->hot_cache, node);
    } else if (node->mode == HOT) {
        node->last_seen = request->timestamp;
        cache->hot_cache = remove_node(cache->hot_cache, node);
        cache->current_hot_size -= node->size;
        node->mode = COLD;
        node->next = NULL;
        node->prev = NULL;
        cache->cold_cache = move_node_to_end(cache->cold_cache, node);
        cache->current_cold_size += node->size;
    } else if (node->mode == COLD && request->admit_probability >= cache->hot_lower_bound) {
        node->last_seen = request->timestamp;
        cache->cold_cache = remove_node(cache->cold_cache, node);
        cache->current_cold_size -= node->size;
        node->mode = HOT;
        node->next = NULL;
        node->prev = NULL;
        cache->hot_cache = move_node_to_end(cache->hot_cache, node);
        cache->current_hot_size += node->size;
    } else if (node->mode == COLD && request->admit_probability >= cache->cold_lower_bound) {
        node->last_seen = request->timestamp;
        cache->cold_cache = move_node_to_end(cache->cold_cache, node);
    }

    return node;
}

HR_LookupAdmitResult lookup_and_admit(HR_Cache* cache, HR_Request* request) {
    HR_LookupAdmitResult result;
    result.admitted = false;
    result.hit = false;
    result.hot_evictions_count = 0;
    result.cold_evictions_count = 0;
    result.hot_evictions_bytes = 0;
    result.cold_evictions_bytes = 0;

    HR_CacheNode* node = lookup(cache, request);
    if (!node) {
        result.hit = false;
        admit(cache, request, &result);
        return result;
    }

    result.hit = true;
    return result;
}

int cleanup_expired_hot(HR_Cache* cache, double last_seen_threshold) {
    int counter = 0;
    HR_CacheNode *cold_node = cache->cold_cache;
    while (cache->hot_cache) {
        if (cache->hot_cache->last_seen > last_seen_threshold) {
            break;
        }

        counter++;
        HR_CacheNode* node = cache->hot_cache;
        cache->hot_cache = remove_node(cache->hot_cache, node);
        cache->current_hot_size -= node->size;
        node->mode = COLD;

        cache->current_cold_size += node->size;
        if (!cold_node) {
            cache->cold_cache = node;
            node->next = node;
            node->prev = node;
            cache->current_cold_size += node->size;
            cold_node = node;
        } else {
            while (cold_node) {
                if (node->last_seen < cold_node->last_seen) {
                    if (cold_node == cache->cold_cache) {
                        cache->cold_cache = node;
                    }

                    node->next = cold_node;
                    node->prev = cold_node->prev;
                    cold_node->prev->next = node;
                    cold_node->prev = node;
                    break;
                }
                    
                if (cold_node->next == cache->cold_cache) {
                    node->next = cold_node->next;
                    node->prev = cold_node;
                    cold_node->next->prev = node;
                    cold_node->next = node;
                    cold_node = node;
                    break;
                }

                cold_node = cold_node->next;
            }
        }
    }
    return counter;
}

void cleanup_cache(HR_Cache* cache, HR_Model* model, int period) {
    if (!cache->cold_cache) {
        return;
    }
    
    HR_CacheNode* cold_start = cache->cold_cache;
    HR_CacheNode** nodes = new HR_CacheNode*[period];
    HR_CacheNode* node = cache->cold_cache;
    int nodes_count;
    while (nodes_count < period) {
        nodes[nodes_count++] = node;
        node = node->next;
        if (node == cold_start) {
            break;
        }
    }
}

void destroy_nodes(HR_CacheNode* node) {
    if (!node) {
        return;
    }

    HR_CacheNode* first = node;
    do {
        HR_CacheNode* tmp = node;
        node = node->next;
        delete tmp;
    } while (node != first);
}

void destroy_lru_cache(HR_Cache* cache) {
    destroy_nodes(cache->hot_cache);
    destroy_nodes(cache->cold_cache);
    delete cache;
}