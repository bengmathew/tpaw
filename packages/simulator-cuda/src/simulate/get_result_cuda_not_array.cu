#include "extern/doctest.h"
#include "extern/nanobench.h"
#include "get_result_cuda_not_array.h"
#include "src/simulate/run/run_result_padded.h"
#include "src/simulate/supported_num_runs.h"
#include "src/utils/bench_utils.h"

ResultCudaNotArrays
get_result_cuda_not_array(const uint32_t num_runs,
                          const RunResultPadded &run_result_padded_not_sorted,
                          const CURRENCY tpaw_net_present_value_exact_month_0_legacy) {
  std::vector<uint32_t> by_run_num_insufficient_fund_months(num_runs);
  thrust::copy_n(
      run_result_padded_not_sorted.by_run_num_insufficient_fund_months.begin(),
      num_runs, // not .end() because of padding.
      by_run_num_insufficient_fund_months.begin());
  uint32_t num_runs_with_insufficient_funds = 0;
  for (const auto &num_insufficient_fund_months :
       by_run_num_insufficient_fund_months) {
    if (num_insufficient_fund_months > 0) {
      num_runs_with_insufficient_funds++;
    }
  }
  return {
      .num_runs = num_runs,
      .num_runs_with_insufficient_funds = num_runs_with_insufficient_funds,
      .tpaw_net_present_value_exact_month_0_legacy =
          tpaw_net_present_value_exact_month_0_legacy,
  };
}

void print_result_cuda_not_array(const ResultCudaNotArrays &result,
                                 const uint32_t num_tabs,
                                 const uint32_t tab_width) {
  const std::string indent0(static_cast<size_t>(num_tabs * tab_width), ' ');
  const std::string indent1(static_cast<size_t>((num_tabs + 1) * tab_width),
                            ' ');

  std::cout << indent0 << "num_runs: " << result.num_runs << std::endl;
  std::cout << indent0 << "num_runs_with_insufficient_funds: "
            << result.num_runs_with_insufficient_funds << std::endl;
}

TEST_CASE("get_result_cuda_not_array") {
  const uint32_t num_runs = 10;
  const uint32_t num_months = 10;
  const RunResultPadded run_result_padded =
      RunResultPadded::get_test_data_for_block_size(num_runs, num_months, 32);
  ResultCudaNotArrays result =
      get_result_cuda_not_array(num_runs, run_result_padded, 1.0);
  CHECK(result.num_runs == num_runs);
  CHECK(result.num_runs_with_insufficient_funds == num_runs);
  CHECK(result.tpaw_net_present_value_exact_month_0_legacy == 1.0);
}

TEST_CASE("bench::get_result_cuda_not_array") {
  const auto do_bench = [](const std::string &name,
                           const uint32_t num_runs,
                           const uint32_t num_months) {
    const RunResultPadded run_result_padded =
        RunResultPadded::get_test_data_for_block_size(
            num_runs, num_months, BLOCK_SORT_BLOCK_SIZE);
    ankerl::nanobench::Bench()
        .timeUnit(std::chrono::milliseconds{1}, "ms")
        .run(name.c_str(), [&]() {
          get_result_cuda_not_array(num_runs, run_result_padded, 1.0);
        });
  };

  for (auto num_runs : bench_num_runs_vec) {
    for (auto num_years : bench_num_years_vec) {
      do_bench(("get_result_cuda_not_array:: " + std::to_string(num_runs) +
                " x " + std::to_string(num_years)),
               num_runs,
               num_years * 12);
    }
  }
}