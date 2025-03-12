#include "extern/doctest.h"
#include "extern/nanobench.h"
#include "src/public_headers/numeric_types.h"
#include "src/simulate/run/run_result_padded.h"
#include "src/simulate/supported_num_runs.h"
#include "src/utils/bench_utils.h"
#include "src/utils/cuda_utils.h"

RunResultPadded RunResultPadded::make(uint32_t num_runs,
                                      uint32_t num_months_to_simulate) {
  // Padding is needed for block sorting to pick percentiles.
  return make_for_block_size(
      num_runs, num_months_to_simulate, BLOCK_SORT_BLOCK_SIZE);
}

RunResultPadded RunResultPadded::make_for_block_size(
    uint32_t num_runs, uint32_t num_months_to_simulate, uint32_t block_size) {
  // Padding is needed for block sorting to pick percentiles.
  const uint32_t num_runs_padded =
      get_padded_num_runs_for_block_size(num_runs, block_size);
  const uint32_t n1 = num_runs_padded * num_months_to_simulate;
  return {num_runs_padded, n1};
}

RunResultPadded::RunResultPadded(uint32_t num_runs_padded, uint32_t n1)
    : num_runs_padded(num_runs_padded),
      by_run_by_mfn_simulated_month_major_balance_start(n1, CURRENCY_MAX_VALUE),
      by_run_by_mfn_simulated_month_major_withdrawals_essential(
          n1, CURRENCY_MAX_VALUE),
      by_run_by_mfn_simulated_month_major_withdrawals_discretionary(
          n1, CURRENCY_MAX_VALUE),
      by_run_by_mfn_simulated_month_major_withdrawals_general(
          n1, CURRENCY_MAX_VALUE),
      by_run_by_mfn_simulated_month_major_withdrawals_total(n1,
                                                            CURRENCY_MAX_VALUE),

      by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate(
          n1, FLOAT_MAX_VALUE),
      by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio(
          n1, FLOAT_MAX_VALUE),
      by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth(
          n1, FLOAT_MAX_VALUE),
      tpaw_by_run_by_mfn_simulated_month_major_spending_tilt(n1,
                                                             FLOAT_MAX_VALUE),
      // In principle, this does not need to be padded because we don't pick
      // percentiles on it, but just for consistency.
      by_run_num_insufficient_fund_months(
          num_runs_padded, 0), // init to zero because we increment it.
      by_run_ending_balance(num_runs_padded, CURRENCY_MAX_VALUE) {}

TEST_CASE("bench::RunResultPadded::constructor") {
  for (const auto &num_runs : bench_num_runs_vec) {
    for (const auto &num_years : bench_num_years_vec) {
      ankerl::nanobench::Bench()
          .timeUnit(std::chrono::milliseconds{1}, "ms")
          .run("RunResult::constructor:" + std::to_string(num_runs) + "x" +
                   std::to_string(num_years),
               [&]() {
                 RunResultPadded run_result =
                     RunResultPadded::make(num_runs, num_years * 12);
               });
    }
  }
}

[[nodiscard]] unique_ptr_gpu<RunResultPadded_GPU>
RunResultPadded::copy_to_gpu() {
  const RunResultPadded_GPU host = RunResultPadded_GPU{
      .num_runs_padded = this->num_runs_padded,
      .by_run_by_mfn_simulated_month_major_balance_start =
          this->by_run_by_mfn_simulated_month_major_balance_start.data().get(),
      .by_run_by_mfn_simulated_month_major_withdrawals_essential =
          this->by_run_by_mfn_simulated_month_major_withdrawals_essential.data()
              .get(),
      .by_run_by_mfn_simulated_month_major_withdrawals_discretionary =
          this->by_run_by_mfn_simulated_month_major_withdrawals_discretionary
              .data()
              .get(),
      .by_run_by_mfn_simulated_month_major_withdrawals_general =
          this->by_run_by_mfn_simulated_month_major_withdrawals_general.data()
              .get(),
      .by_run_by_mfn_simulated_month_major_withdrawals_total =
          this->by_run_by_mfn_simulated_month_major_withdrawals_total.data()
              .get(),
      .by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate =
          this->by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate
              .data()
              .get(),
      .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio =
          this->by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio
              .data()
              .get(),
      .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth =
          this->by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth
              .data()
              .get(),
      .tpaw_by_run_by_mfn_simulated_month_major_spending_tilt =
          this->tpaw_by_run_by_mfn_simulated_month_major_spending_tilt.data()
              .get(),
      .by_run_num_insufficient_fund_months =
          this->by_run_num_insufficient_fund_months.data().get(),
      .by_run_ending_balance = this->by_run_ending_balance.data().get(),
  };

  return copyStructToDevice(host);
}

RunResultPadded RunResultPadded::get_test_data_for_block_size(
    uint32_t num_runs, uint32_t num_months, uint32_t block_size) {
  RunResultPadded run_result_padded =
      RunResultPadded::make_for_block_size(num_runs, num_months, block_size);

  const uint32_t num_runs_padded = run_result_padded.num_runs_padded;
  run_result_padded.by_run_by_mfn_simulated_month_major_balance_start =
      get_test_data_single<CURRENCY>(num_runs_padded, num_months);
  run_result_padded.by_run_by_mfn_simulated_month_major_withdrawals_essential =
      get_test_data_single<CURRENCY>(num_runs_padded, num_months);
  run_result_padded
      .by_run_by_mfn_simulated_month_major_withdrawals_discretionary =
      get_test_data_single<CURRENCY>(num_runs_padded, num_months);
  run_result_padded.by_run_by_mfn_simulated_month_major_withdrawals_general =
      get_test_data_single<CURRENCY>(num_runs_padded, num_months);
  run_result_padded.by_run_by_mfn_simulated_month_major_withdrawals_total =
      get_test_data_single<CURRENCY>(num_runs_padded, num_months);
  run_result_padded
      .by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate =
      get_test_data_single<FLOAT>(num_runs_padded, num_months);
  run_result_padded
      .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio =
      get_test_data_single<FLOAT>(num_runs_padded, num_months);
  run_result_padded
      .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth =
      get_test_data_single<FLOAT>(num_runs_padded, num_months);
  run_result_padded.tpaw_by_run_by_mfn_simulated_month_major_spending_tilt =
      get_test_data_single<FLOAT>(num_runs_padded, num_months);

  run_result_padded.by_run_num_insufficient_fund_months =
      get_test_data_single<CURRENCY>(num_runs_padded, 1);
  run_result_padded.by_run_ending_balance =
      get_test_data_single<CURRENCY>(num_runs_padded, 1);
  return run_result_padded;
}

[[nodiscard]] RunResult_Single
RunResultPadded::get_single(const uint32_t run_index,
                            const uint32_t month_index) const {

  const uint32_t n1 = MonthMajor::get_run_by_mfn_index(
      this->num_runs_padded, run_index, month_index);
  return RunResult_Single{
      .balance_start = by_run_by_mfn_simulated_month_major_balance_start[n1],
      .withdrawals_essential =
          by_run_by_mfn_simulated_month_major_withdrawals_essential[n1],
      .withdrawals_discretionary =
          by_run_by_mfn_simulated_month_major_withdrawals_discretionary[n1],
      .withdrawals_general =
          by_run_by_mfn_simulated_month_major_withdrawals_general[n1],
      .withdrawals_total =
          by_run_by_mfn_simulated_month_major_withdrawals_total[n1],
      .withdrawals_from_savings_portfolio_rate =
          by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate
              [n1],
      .after_withdrawals_allocation_savings_portfolio =
          by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio
              [n1],
      .after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth =
          by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth
              [n1],
      .spending_tilt =
          tpaw_by_run_by_mfn_simulated_month_major_spending_tilt[n1],
      .num_insufficient_fund_months =
          by_run_num_insufficient_fund_months[run_index],
      .ending_balance = by_run_ending_balance[run_index],
  };
}

void RunResult_Single::print(uint32_t num_tabs, uint32_t tab_width) const {
  const std::string i(static_cast<size_t>(num_tabs * tab_width), ' ');
  std::cout << i << "balance_start: " << balance_start << "\n";
  std::cout << i << "withdrawals_essential: " << withdrawals_essential << "\n";
  std::cout << i << "withdrawals_discretionary: " << withdrawals_discretionary
            << "\n";
  std::cout << i << "withdrawals_general: " << withdrawals_general << "\n";
  std::cout << i << "withdrawals_total: " << withdrawals_total << "\n";
  std::cout << i << "withdrawals_from_savings_portfolio_rate: "
            << withdrawals_from_savings_portfolio_rate << "\n";
  std::cout << i << "after_withdrawals_allocation_savings_portfolio: "
            << after_withdrawals_allocation_savings_portfolio << "\n";
  std::cout
      << i
      << "after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth: "
      << after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth
      << "\n";
  std::cout << i << "spending_tilt: " << spending_tilt << "\n";
  std::cout << i
            << "num_insufficient_fund_months: " << num_insufficient_fund_months
            << "\n";
  std::cout << i << "ending_balance: " << ending_balance << "\n";
}

TEST_CASE("bench::run_result as thrust::device_vector") {
  const auto do_bench = [](const char *name,
                           const uint32_t num_runs,
                           const uint32_t num_years) {
    ankerl::nanobench::Bench()
        .timeUnit(std::chrono::milliseconds{1}, "ms")
        .run(name, [&]() {
          const uint32_t n = num_runs * num_years;
          const thrust::device_vector<CURRENCY> currency_out_by_run_by_mfn_1(n);
          const thrust::device_vector<CURRENCY> currency_out_by_run_by_mfn_2(n);
          const thrust::device_vector<CURRENCY> currency_out_by_run_by_mfn_3(n);
          const thrust::device_vector<CURRENCY> currency_out_by_run_by_mfn_4(n);
          const thrust::device_vector<CURRENCY> currency_out_by_run_by_mfn_5(n);

          const thrust::device_vector<FLOAT> float_by_run_by_mfn_1(n);
          const thrust::device_vector<FLOAT> float_by_run_by_mfn_2(n);
          const thrust::device_vector<FLOAT> float_by_run_by_mfn_3(n);
          const thrust::device_vector<FLOAT> float_by_run_by_mfn_4(n);

          const thrust::device_vector<uint32_t> uint32_by_run_1(1);
          const thrust::device_vector<CURRENCY> currency_out_by_run_1(1);
        });
  };
  for (auto num_runs : bench_num_runs_vec) {
    for (auto num_years : bench_num_years_vec) {
      do_bench(("run_result as thrust::device_vector:" +
                std::to_string(num_runs) + "x" + std::to_string(num_years))
                   .c_str(),
               num_runs,
               num_years);
    }
  }
}
