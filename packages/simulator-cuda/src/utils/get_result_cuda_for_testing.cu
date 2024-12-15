#include "get_result_cuda_for_testing.h"

#include "src/public_headers/numeric_types.h"
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

_ResultCudaArraysForTestingRAII::_ResultCudaArraysForTestingRAII(
    const uint32_t num_percentiles, const uint32_t num_months)
    : num_percentiles(num_percentiles), num_months(num_months),
      by_percentile_by_mfn_simulated_percentile_major_balance_start(
          static_cast<uint32_t>(num_percentiles * num_months)),
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential(
          static_cast<uint32_t>(num_percentiles * num_months)),
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary(
          static_cast<uint32_t>(num_percentiles * num_months)),
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_general(
          static_cast<uint32_t>(num_percentiles * num_months)),
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_total(
          static_cast<uint32_t>(num_percentiles * num_months)),
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate(
          static_cast<uint32_t>(num_percentiles * num_months)),
      by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio(
          static_cast<uint32_t>(num_percentiles * num_months)),
      by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth(
          static_cast<uint32_t>(num_percentiles * num_months)),
      tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt(
          static_cast<uint32_t>(num_percentiles * num_months)),
      by_percentile_ending_balance(static_cast<size_t>(num_percentiles)),
      tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn(
          static_cast<size_t>(num_months)) {}

void _ResultCudaArraysForTestingRAII::print_single(
    const std::vector<CURRENCY> &vec, const uint32_t num_percentiles) {
  assert(vec.size() % num_percentiles == 0);
  const uint32_t num_months = vec.size() / num_percentiles;
  for (uint32_t p = 0; p < num_percentiles; p++) {
    std::cout << "Percentile " << p << ": ";
    for (uint32_t m = 0; m < num_months; m++) {
      const uint32_t index = p * num_months + m;
      std::cout << vec[index] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

std::pair<ResultCudaArrays, _ResultCudaArraysForTestingRAII>
get_result_cuda_for_testing(const uint32_t num_percentiles,
                            const uint32_t num_months) {
  _ResultCudaArraysForTestingRAII raii(num_percentiles, num_months);

  const ResultCudaArrays result_cuda = ResultCudaArrays{
      .by_percentile_by_mfn_simulated_percentile_major_balance_start =
          raii.by_percentile_by_mfn_simulated_percentile_major_balance_start
              .data(),
      .by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential =
          raii.by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential
              .data(),
      .by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary =
          raii.by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary
              .data(),
      .by_percentile_by_mfn_simulated_percentile_major_withdrawals_general =
          raii.by_percentile_by_mfn_simulated_percentile_major_withdrawals_general
              .data(),
      .by_percentile_by_mfn_simulated_percentile_major_withdrawals_total =
          raii.by_percentile_by_mfn_simulated_percentile_major_withdrawals_total
              .data(),
      .by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate =
          raii.by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate
              .data(),
      .by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio =
          raii.by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio
              .data(),
      .by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth =
          raii.by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth
              .data(),
      .tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt =
          raii.tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt
              .data(),
      .by_percentile_ending_balance = raii.by_percentile_ending_balance.data(),
      .tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn =
          raii.tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn.data(),
  };

  return std::make_pair(result_cuda, std::move(raii));
}