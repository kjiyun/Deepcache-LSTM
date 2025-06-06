#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif


const double HAZARD_CONST = 3.0 / 4;
const double EPSILON = std::numeric_limits<double>::min();

void report_memory() {
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);

#if defined(__APPLE__)
    // On macOS, use the task_info function
    task_basic_info_data_t info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kl = task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size);

    double memory_usage;
    if (kl == KERN_SUCCESS) {
        memory_usage = info.resident_size / 1024.0 / 1024.0;  // MB
    } else {
        memory_usage = -1.0;
    }
#else    // On most other Unix-like systems, ru_maxrss is in kilobytes
    double memory_usage = rusage.ru_maxrss / 1024.0;
#endif

    std::cout << "Memory usage: " << memory_usage << " MB\n";
}

void calculate_diffs(const double *input, const int input_count, double *intervals, int *diff_count) {
    // Only one timestamp, no intervals -> return infinity
    if (input_count == 1) {
        *diff_count = 1;
        intervals[0] = input[0];
    }

    for (int i = 0; i < input_count - 1; i++) {
        intervals[i] = input[i + 1] - input[i];
    }
    *diff_count = input_count - 1;
}

int qsort_compare(const void *a, const void *b) {
    double arg1 = *reinterpret_cast<const double*>(a);
    double arg2 = *reinterpret_cast<const double*>(b);

    if(arg1 < arg2) return -1;
    if(arg1 > arg2) return 1;
    return 0;
}

void nelson_aalen_fitter(double* durations, double* hazards, int* data_count, bool discrete) { 
    // Sort the data array in ascending order of durations
    double last_duration = durations[*data_count - 1];
    qsort(durations, *data_count, sizeof(double), qsort_compare);
    
    double risk_set_size = *data_count;
    int i = 0;
    int j = 0;
    while(i < *data_count) {
        double current_duration = durations[i];
        int event_count = 0;
        int missed_count = 0;
        // Count the number of events and censored observations at the current duration
        while(i < *data_count && durations[i] == current_duration) {
            event_count++;
            i++;
        }
        if (current_duration == last_duration) {
            event_count--;
            missed_count++;
        }

        double hazard = event_count / risk_set_size;
        if (!discrete && event_count > 1) {
            hazard = 0;
            for (int k = 0; k < event_count; k++) {
                hazard += 1 / (risk_set_size - k);
            }
        }

        // Update the cumulative hazard and risk set size
        risk_set_size -= event_count + missed_count;
        // Store the cumulative hazard for all observations at the current duration
        durations[j] = current_duration;
        hazards[j] = hazard;
        j++;
    }
    *data_count = j;

    // Add 0 to the beginning of the durations
    memmove(durations + 1, durations, sizeof(double) * (*data_count));
    durations[0] = 0.0;

    // Add 0 to the beginning of the cumulative hazards
    memmove(hazards + 1, hazards, sizeof(double) * (*data_count));
    hazards[0] = 0.0;

    *data_count += 1;
}

double simple_calculate_hazard(double input, double* timestamps_diffs, double* cumulative_hazards_diffs, int diffs_count, double bandwidth) {
    double result = 0;
    double constant = HAZARD_CONST / bandwidth;
    for (int i = 0; i < diffs_count; i++) {
        double diff = input - timestamps_diffs[i];
        if (diff <= bandwidth && diff >= -bandwidth) {
            double hazard = diff / bandwidth;
            result += constant * (1 - (hazard * hazard)) * cumulative_hazards_diffs[i];
        }
    }
    return result;
}

double calculate_hazard(double input, double* timestamps_diffs, double* cumulative_hazards_diffs, int diffs_count, double bandwidth) {
    double lower_bound_value = input - bandwidth;
    double upper_bound_value = input + bandwidth + EPSILON;
    
    // Use lower_bound to find the first index where timestamps_diffs[i] >= lower_bound_value
    int first = std::lower_bound(timestamps_diffs, timestamps_diffs + diffs_count, lower_bound_value) - timestamps_diffs;

    if (first == diffs_count || timestamps_diffs[first] > upper_bound_value) {
        return 0;
    }

    // Use upper_bound to find the first index where timestamps_diffs[i] < upper_bound_value
    int last = std::upper_bound(timestamps_diffs, timestamps_diffs + diffs_count, upper_bound_value) - timestamps_diffs;

    double result = 0;
    double constant = HAZARD_CONST / bandwidth;
    for (int i = first; i < last; i++) {
        double hazard = (input - timestamps_diffs[i]) / bandwidth;
        result += constant * (1 - (hazard * hazard)) * cumulative_hazards_diffs[i];
    }
    return result;
}