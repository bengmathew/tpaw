#ifndef GET_SERIES_STATS_H
#define GET_SERIES_STATS_H

#include "src/public_headers/numeric_types.h"
#include "src/public_headers/stats.h"
#include <thrust/device_vector.h>

FLOAT get_series_mean(const thrust::device_vector<FLOAT> &series);
Stats get_series_stats(const thrust::device_vector<FLOAT> &series);

#endif // GET_SERIES_STATS_H
