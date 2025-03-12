
#include "extern/doctest.h"
#include "extern/nanobench.h"
#include "pick_percentiles.h"
#include "src/simulate/run/run_result_padded.h"
#include "src/simulate/supported_num_runs.h"
#include "src/utils/bench_utils.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/get_result_cuda_for_testing.h"
#include "src/utils/run_mfn_indexing.h"
#include <algorithm>
#include <cstdint>

template <typename T>
__global__ void
_kernel(const uint32_t num_runs_padded,
        const uint32_t num_months_simulated,
        const T *const padded_data_by_run_by_mfn_simulated_month_major,
        const uint32_t num_percentiles,
        const uint32_t *percentile_indices,
        T *const out_by_percentile_by_mfn_percentile_major) {

  uint32_t month_index = (blockIdx.x * blockDim.x + threadIdx.x);
  if (month_index >= num_months_simulated)
    return;
  for (uint32_t j = 0; j < num_percentiles; ++j) {
    uint32_t src_index = MonthMajor::get_run_by_mfn_index(
        num_runs_padded, percentile_indices[j], month_index);
    uint32_t result_index = num_months_simulated * j + month_index;
    out_by_percentile_by_mfn_percentile_major[result_index] =
        padded_data_by_run_by_mfn_simulated_month_major[src_index];
  }
}

template <typename T>
void _pick_single(
    const uint32_t num_runs_padded,
    const uint32_t num_months_simulated,
    const thrust::device_vector<T>
        &padded_data_by_run_by_mfn_simulated_month_major,
    const thrust::device_vector<uint32_t> &percentile_indices,
    T *const out_by_percentile_by_mfn_simulated_percentile_major) {
  thrust::device_vector<T> result(percentile_indices.size() *
                                  num_months_simulated);
  const int block_size{32};
  _kernel<T>
      <<<(num_months_simulated + block_size - 1) / block_size, block_size>>>(
          num_runs_padded,
          num_months_simulated,
          padded_data_by_run_by_mfn_simulated_month_major.data().get(),
          percentile_indices.size(),
          percentile_indices.data().get(),
          result.data().get());

  thrust::copy(result.begin(),
               result.end(),
               out_by_percentile_by_mfn_simulated_percentile_major);
}

TEST_CASE("pick_percentiles::_pick_single") {
  const uint32_t num_runs_padded = 10;
  const uint32_t num_months = 2;
  const std::vector<CURRENCY> currency_data_host =
      RunResultPadded::get_test_data_single<CURRENCY>(num_runs_padded,
                                                      num_months);
  const thrust::device_vector<CURRENCY> currency_data_device(
      currency_data_host);
  const std::vector<uint32_t> percentile_indices_host = {0, 4};
  const thrust::device_vector<uint32_t> percentile_indices_device(
      percentile_indices_host);

  thrust::device_vector<CURRENCY> result_device(
      percentile_indices_device.size() * num_months);

  _pick_single(num_runs_padded,
               num_months,
               currency_data_device,
               percentile_indices_device,
               result_device.data().get());

  std::vector<CURRENCY> result_host = device_vector_to_host(result_device);

  CHECK(result_host == std::vector<CURRENCY>{10010, 20010, 10006, 20006});
}

uint32_t get_percentile_index(uint32_t percentile, uint32_t num_runs) {
  double as_double = static_cast<double>(percentile) /
                     static_cast<double>(100) *
                     (static_cast<double>(num_runs) - 1.0);

  return static_cast<uint32_t>(
      // Clamp due to floating point imprecision.
      std::clamp(static_cast<int32_t>(std::round(as_double)),
                 static_cast<int32_t>(0),
                 static_cast<int32_t>(num_runs - 1)));
}

void pick_percentiles(const RunResultPadded &run_result_padded_sorted,
                      const uint32_t num_runs,
                      const std::vector<uint32_t> &percentiles,
                      const uint32_t num_months_simulated,
                      ResultCudaArrays &result_cuda) {
  std::vector<uint32_t> percentile_indices_host(percentiles.size());
  for (uint32_t i = 0; i < percentile_indices_host.size(); i++) {
    percentile_indices_host[i] = get_percentile_index(percentiles[i], num_runs);
  }
  thrust::device_vector<uint32_t> percentile_indices_device(
      percentile_indices_host);

  const auto _helper =
      [&]<typename T>(
          const uint32_t num_months_simulated,
          const thrust::device_vector<T>
              &padded_data_by_run_by_mfn_simulated_month_major,
          T *const out_by_percentile_by_mfn_simulated_percentile_major) {
        _pick_single(run_result_padded_sorted.num_runs_padded,
                     num_months_simulated,
                     padded_data_by_run_by_mfn_simulated_month_major,
                     percentile_indices_device,
                     out_by_percentile_by_mfn_simulated_percentile_major);
      };
  _helper(num_months_simulated,
          run_result_padded_sorted
              .by_run_by_mfn_simulated_month_major_balance_start,
          result_cuda
              .by_percentile_by_mfn_simulated_percentile_major_balance_start);

  _helper(
      num_months_simulated,
      run_result_padded_sorted
          .by_run_by_mfn_simulated_month_major_withdrawals_essential,
      result_cuda
          .by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential);

  _helper(
      num_months_simulated,
      run_result_padded_sorted
          .by_run_by_mfn_simulated_month_major_withdrawals_discretionary,
      result_cuda
          .by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary);

  _helper(
      num_months_simulated,
      run_result_padded_sorted
          .by_run_by_mfn_simulated_month_major_withdrawals_general,
      result_cuda
          .by_percentile_by_mfn_simulated_percentile_major_withdrawals_general);

  _helper(
      num_months_simulated,
      run_result_padded_sorted
          .by_run_by_mfn_simulated_month_major_withdrawals_total,
      result_cuda
          .by_percentile_by_mfn_simulated_percentile_major_withdrawals_total);

  _helper(
      num_months_simulated,
      run_result_padded_sorted
          .by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate,
      result_cuda
          .by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate);

  _helper(
      num_months_simulated,
      run_result_padded_sorted
          .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio,
      result_cuda
          .by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio);

  _helper(
      num_months_simulated,
      run_result_padded_sorted
          .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth,
      result_cuda
          .by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth);

  _helper(
      num_months_simulated,
      run_result_padded_sorted
          .tpaw_by_run_by_mfn_simulated_month_major_spending_tilt,
      result_cuda
          .tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt);

  _helper(1,
          run_result_padded_sorted.by_run_ending_balance,
          result_cuda.by_percentile_ending_balance);
}

TEST_CASE("bench::pick_percentiles") {

  const std::vector<uint32_t> percentiles = {5, 50, 95};
  const auto do_bench = [&](const std::string &name,
                            const uint32_t num_runs,
                            const uint32_t num_months) {
    auto [result_cuda, raii] =
        get_result_cuda_for_testing(percentiles.size(), num_months);

    const RunResultPadded run_result_padded =
        RunResultPadded::get_test_data_for_block_size(
            num_runs, num_months, BLOCK_SORT_BLOCK_SIZE);

    ankerl::nanobench::Bench()
        .timeUnit(std::chrono::milliseconds{1}, "ms")
        .run(name.c_str(), [&]() {
          pick_percentiles(run_result_padded,
                           num_runs,
                           percentiles,
                           num_months,
                           result_cuda);
        });
  };

  for (auto num_runs : bench_num_runs_vec) {
    for (auto num_years : bench_num_years_vec) {
      do_bench(("pick_percentiles:: " + std::to_string(num_runs) + " x " +
                std::to_string(num_years))
                   .c_str(),
               num_runs,
               num_years * 12);
    }
  }
}
