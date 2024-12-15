#ifndef RESULT_CUDA_H
#define RESULT_CUDA_H

#include "numeric_types.h"
#include <stdint.h> // bindgen cannot find cstdint (needed for sized number types).

// bindgen (to generate rust bindings) has trouble with extern "C"
// because it is expecting C not C++ and extern "C" is a C++ thing.
#ifdef __cplusplus
extern "C" {
#endif

struct ResultCudaNotArrays {
  uint32_t num_runs;
  uint32_t num_runs_with_insufficient_funds;
  CURRENCY tpaw_net_present_value_exact_month_0_legacy;
};

struct ResultCudaArrays {

  // by_run_by_mfn_simulated
  CURRENCY *by_percentile_by_mfn_simulated_percentile_major_balance_start;
  CURRENCY
  *by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential;
  CURRENCY *
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary;
  CURRENCY
  *by_percentile_by_mfn_simulated_percentile_major_withdrawals_general;
  CURRENCY
  *by_percentile_by_mfn_simulated_percentile_major_withdrawals_total;
  FLOAT *
      by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate;
  FLOAT *
      by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio;
  FLOAT *
      by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth;
  FLOAT *tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt;
  CURRENCY *by_percentile_ending_balance;

  FLOAT *tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn;
};

#ifdef __cplusplus
}
#endif

#endif // RESULT_CUDA_H
