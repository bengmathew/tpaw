#ifndef RUN_RESULT_PADDED_H
#define RUN_RESULT_PADDED_H

#include "src/public_headers/numeric_types.h"
#include "src/utils/cuda_utils.h"

struct RunResultPadded_GPU {
  uint32_t num_runs_padded;

  // by_run_by_mfn_simulated
  CURRENCY *by_run_by_mfn_simulated_month_major_balance_start;
  CURRENCY *by_run_by_mfn_simulated_month_major_withdrawals_essential;
  CURRENCY *by_run_by_mfn_simulated_month_major_withdrawals_discretionary;
  CURRENCY *by_run_by_mfn_simulated_month_major_withdrawals_general;
  CURRENCY *by_run_by_mfn_simulated_month_major_withdrawals_total;
  FLOAT *
      by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate;
  FLOAT *
      by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio;
  FLOAT *
      by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth;
  FLOAT *tpaw_by_run_by_mfn_simulated_month_major_spending_tilt;

  // Note: These are in run_major and not month_major. This is because we don't
  // sort and pick percentiles on like on the other arrays. We use it to
  // stat
  // FLOAT

  // by_run
  uint32_t *by_run_num_insufficient_fund_months;
  CURRENCY *by_run_ending_balance;
};

struct RunResult_Single {
  CURRENCY balance_start;
  CURRENCY withdrawals_essential;
  CURRENCY withdrawals_discretionary;
  CURRENCY withdrawals_general;
  CURRENCY withdrawals_total;

  FLOAT withdrawals_from_savings_portfolio_rate;
  FLOAT after_withdrawals_allocation_savings_portfolio;
  FLOAT
  after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth;
  FLOAT spending_tilt;
  uint32_t num_insufficient_fund_months;
  CURRENCY ending_balance;

  void print(uint32_t num_tabs, uint32_t tab_width) const;
};

struct RunResultPadded {
  uint32_t num_runs_padded;

  // by_run_by_mfn_simulated
  thrust::device_vector<CURRENCY>
      by_run_by_mfn_simulated_month_major_balance_start;
  thrust::device_vector<CURRENCY>
      by_run_by_mfn_simulated_month_major_withdrawals_essential;
  thrust::device_vector<CURRENCY>
      by_run_by_mfn_simulated_month_major_withdrawals_discretionary;
  thrust::device_vector<CURRENCY>
      by_run_by_mfn_simulated_month_major_withdrawals_general;
  thrust::device_vector<CURRENCY>
      by_run_by_mfn_simulated_month_major_withdrawals_total;
  thrust::device_vector<FLOAT>
      by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate;
  thrust::device_vector<FLOAT>
      by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio;
  thrust::device_vector<FLOAT>
      by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth;
  thrust::device_vector<FLOAT>
      tpaw_by_run_by_mfn_simulated_month_major_spending_tilt;

  // by_run
  thrust::device_vector<uint32_t> by_run_num_insufficient_fund_months;
  thrust::device_vector<CURRENCY> by_run_ending_balance;

  unique_ptr_gpu<RunResultPadded_GPU> copy_to_gpu();
  [[nodiscard]] RunResult_Single get_single(const uint32_t run_index,
                                            const uint32_t month_index) const;

  static RunResultPadded make(uint32_t num_runs,
                              uint32_t num_months_to_simulate);

  static RunResultPadded make_for_block_size(uint32_t num_runs,
                                             uint32_t num_months_to_simulate,
                                             uint32_t block_size);

  template <typename T>
  static std::vector<T> get_test_data_single(uint32_t num_runs_padded,
                                             uint32_t num_months) {
    {
      std::vector<T> result(
          static_cast<uint32_t>(num_runs_padded * num_months));

      for (uint32_t month_index = 0; month_index < num_months; ++month_index) {
        for (uint32_t run_index = 0; run_index < num_runs_padded; ++run_index) {
          const uint32_t index = get_run_by_mfn_month_major_index(
              num_runs_padded, run_index, month_index);
          result[index] = static_cast<T>(num_runs_padded - run_index +
                                         (month_index + 1) * 10000);
        }
      }
      return result;
    }
  }

  static RunResultPadded get_test_data_for_block_size(uint32_t num_runs,
                                                      uint32_t num_months,
                                                      uint32_t block_size);

private:
  RunResultPadded(uint32_t num_runs_padded, uint32_t n1);
};

#endif // RUN_RESULT_PADDED_H
