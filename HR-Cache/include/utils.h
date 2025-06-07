#ifndef HR_UTILS_H
#define HR_UTILS_H

void report_memory();
void calculate_diffs(const double *input, const int input_count, double *intervals, int *diff_count);
void nelson_aalen_fitter(double* durations, double* cumulative_hazards, int* data_count, bool discrete);
double calculate_hazard(double input, double* timestamps_diffs, double* cumulative_hazards_diffs, int diffs_count, double bandwidth);

#endif // HR_UTILS_H