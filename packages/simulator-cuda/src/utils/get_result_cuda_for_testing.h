#ifndef GET_RESULT_CUDA_FOR_TESTING_H
#define GET_RESULT_CUDA_FOR_TESTING_H

#include "src/public_headers/numeric_types.h"
#include "src/public_headers/result_cuda.h"
#include <cstdint>
#include <vector>

struct _ResultCudaArraysForTestingRAII {
  uint32_t num_percentiles;
  uint32_t num_months;

  std::vector<CURRENCY>
      by_percentile_by_mfn_simulated_percentile_major_balance_start;
  std::vector<CURRENCY>
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential;
  std::vector<CURRENCY>
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary;
  std::vector<CURRENCY>
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_general;
  std::vector<CURRENCY>
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_total;
  std::vector<FLOAT>
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate;
  std::vector<FLOAT>
      by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio;
  std::vector<FLOAT>
      by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth;
  std::vector<FLOAT>
      tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt;
  std::vector<CURRENCY> by_percentile_ending_balance;

  std::vector<FLOAT> tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn;

  _ResultCudaArraysForTestingRAII(const uint32_t num_percentiles,
                                  const uint32_t num_months);

  static void print_single(const std::vector<CURRENCY> &vec,
                           const uint32_t num_percentiles);

  void print(const uint32_t num_tabs, const uint32_t tab_width);
};

std::pair<ResultCudaArrays, _ResultCudaArraysForTestingRAII>
get_result_cuda_for_testing(const uint32_t num_percentiles,
                            const uint32_t num_months);

#endif // GET_RESULT_CUDA_FOR_TESTING_H
