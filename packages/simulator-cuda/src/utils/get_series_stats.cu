#include "./get_series_stats.h"
#include "extern/doctest.h"
#include <stdint.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

FLOAT get_series_mean(const thrust::device_vector<FLOAT> &series) {
  return thrust::reduce(series.begin(), series.end()) /
         static_cast<FLOAT>(series.size());
}

TEST_CASE("STAR::get_series_mean") {
  const thrust::device_vector<FLOAT> series = {1, 2, 3, 4, 5};
  CHECK(get_series_mean(series) == 3);
}

FLOAT _get_series_variance(const thrust::device_vector<FLOAT> &series,
                          const FLOAT mean) {
  const FLOAT sum = thrust::transform_reduce(
      series.begin(),
      series.end(),
      [mean] __device__(const FLOAT x) -> FLOAT {
        const FLOAT diff = x - mean;
        return diff * diff;
      },
      FLOAT_L(0.0),
      thrust::plus<FLOAT>());
  return sum / static_cast<FLOAT>(series.size() - 1);
}

TEST_CASE("STAR::get_series_variance") {
  const thrust::device_vector<FLOAT> series = {1, 2, 3, 4, 5};
  const FLOAT mean = get_series_mean(series);
  CHECK(_get_series_variance(series, mean) == FLOAT_L(2.5));
}

Stats get_series_stats(const thrust::device_vector<FLOAT> &series) {
  const FLOAT mean = get_series_mean(series);
  const FLOAT variance = _get_series_variance(series, mean);
  return {.n = static_cast<uint32_t>(series.size()),
          .mean = mean,
          .variance = variance};
}
