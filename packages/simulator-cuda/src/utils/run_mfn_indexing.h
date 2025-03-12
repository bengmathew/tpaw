#ifndef RUN_MFN_INDEXING_H
#define RUN_MFN_INDEXING_H

#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

struct RunAndMonthIndex {
  uint32_t run_index;
  uint32_t month_index;

  __device__ __host__ __forceinline__ bool
  operator==(const RunAndMonthIndex &other) const {
    return run_index == other.run_index && month_index == other.month_index;
  }
};

namespace RunMajor {

  __device__ __host__ __forceinline__ uint32_t get_run_by_mfn_index(
      uint32_t num_months, uint32_t run_index, uint32_t month_index) {
    return run_index * num_months + month_index;
  }

  // Pair is (run_index, month_index)
  __device__ __host__ __forceinline__ RunAndMonthIndex
  get_run_and_month_index(uint32_t num_months, uint32_t index) {
    return {.run_index = index / num_months, .month_index = index % num_months};
  }

} // namespace RunMajor

namespace MonthMajor {

  __device__ __host__ __forceinline__ uint32_t get_run_by_mfn_index(
      uint32_t num_runs, uint32_t run_index, uint32_t month_index) {
    return month_index * num_runs + run_index;
  }

  // Pair is (run_index, month_index)
  __device__ __host__ __forceinline__ RunAndMonthIndex
  get_run_and_month_index(uint32_t num_runs, uint32_t index) {
    return {.run_index = index % num_runs, .month_index = index / num_runs};
  }

  __device__ __host__ __forceinline__ uint32_t
  get_run_by_mfn_index_from_run_major_index(uint32_t num_runs,
                                            uint32_t num_months,
                                            uint32_t index) {
    const auto pair = RunMajor::get_run_and_month_index(num_months, index);
    return get_run_by_mfn_index(num_runs, pair.run_index, pair.month_index);
  }

} // namespace MonthMajor

// Simulation is marginally faster with MONTH_MAJOR.
#define MONTH_MAJOR
#ifdef MONTH_MAJOR

__device__ __host__ __forceinline__ uint32_t
get_run_by_mfn_index(uint32_t num_runs,
                     __attribute__((unused)) uint32_t num_months,
                     uint32_t run_index,
                     uint32_t month_index) {
  return MonthMajor::get_run_by_mfn_index(num_runs, run_index, month_index);
}

__device__ __host__ __forceinline__ uint32_t
get_run_by_mfn_index_from_run_major_index(uint32_t num_runs,
                                          uint32_t num_months,
                                          uint32_t index) {
  return MonthMajor::get_run_by_mfn_index_from_run_major_index(
      num_runs, num_months, index);
}
template <typename T>
thrust::device_vector<T>
convert_to_run_major(const thrust::device_vector<T> &value_by_run_by_mfn,
                     const uint32_t num_runs,
                     const uint32_t num_months) {
  const T *value_by_run_by_mfn_ptr = value_by_run_by_mfn.data().get();
  thrust::device_vector<T> value_in_run_major(value_by_run_by_mfn.size());
  thrust::transform(
      thrust::counting_iterator<uint32_t>(0),
      thrust::counting_iterator<uint32_t>(value_by_run_by_mfn.size()),
      value_in_run_major.begin(),
      [num_runs, num_months, value_by_run_by_mfn_ptr] __device__(
          const uint32_t run_major_index) -> T {
        const uint32_t index = get_run_by_mfn_index_from_run_major_index(
            num_runs, num_months, run_major_index);
        return value_by_run_by_mfn_ptr[index];
      });
  return value_in_run_major;
}

#endif

// #define RUN_MAJOR
#ifdef RUN_MAJOR
__device__ __host__ __forceinline__ uint32_t
get_run_by_mfn_index(__attribute__((unused)) uint32_t num_runs,
                     uint32_t num_months,
                     uint32_t run_index,
                     uint32_t month_index) {
  return RunMajor::get_run_by_mfn_run_major_index(
      num_months, run_index, month_index);
}

__device__ __host__ __forceinline__ uint32_t
get_run_by_mfn_index_from_run_major_index(__attribute__((unused))
                                          uint32_t num_runs,
                                          __attribute__((unused))
                                          uint32_t num_months,
                                          uint32_t index) {
  return index;
}
#endif

#endif // RUN_MFN_INDEXING_H
