#ifndef CUDA_PROCESS_TPAW_RUN_X_MFN_SIMULATED_X_MFN_H
#define CUDA_PROCESS_TPAW_RUN_X_MFN_SIMULATED_X_MFN_H

#include "src/public_headers/numeric_types.h"
#include "src/simulate/cuda_process_historical_returns.h"
#include <stdint.h>
#include <thrust/device_vector.h>

struct Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN {
  struct Entry {

    struct ApproxNPV {
      // Not including with_current_month. That should be calculated in the
      // simulation for that month because it can be done with full precision.
      // This allows npv values to be consistent within a single month even
      // though they are approximations.
      CURRENCY income_without_current_month;
      CURRENCY essential_expenses_without_current_month;
      CURRENCY discretionary_expenses_without_current_month;
      // legacy_with_current_month == legacy_without_current_month because
      // there is no legacy entry for a given month.
      CURRENCY legacy_exact;
      __device__ __host__ void print(uint32_t num_tabs) const;
    };

    // Approximation NPV because we calculate with CURRENCY_NPV (i.e. float) and
    // then convert to CURRENCY.
    ApproxNPV npv_approx;
    FLOAT stock_allocation_total_portfolio;
    FLOAT legacy_stock_allocation;
    FLOAT spending_tilt;
    FLOAT cumulative_1_plus_g_over_1_plus_r;

    __device__ __host__ void print(uint32_t num_tabs) const;
  };

  // This is needed for the balance sheet.
  thrust::device_vector<FLOAT>
      tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn;
  thrust::device_vector<Entry> for_expected_run;
  thrust::device_vector<Entry> for_normal_run;
};

Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN
cuda_process_tpaw_run_x_mfn_simulated_x_mfn(
    const uint32_t num_runs,
    const uint32_t num_months_to_simulate,
    const uint32_t num_months,
    const std::vector<CURRENCY> &income_by_mfn,
    const std::vector<CURRENCY> &essential_expense_by_mfn,
    const std::vector<CURRENCY> &discretionary_expense_by_mfn,
    const std::vector<FLOAT> &rra_including_pos_infinity_by_mfn,
    const FLOAT annual_empirical_log_variance_stocks,
    const FLOAT time_preference,
    const FLOAT annual_additional_spending_tilt,
    const FLOAT legacy_rra_including_pos_infinity,
    const CURRENCY legacy,
    const thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        &expected_returns_by_mfn_simulated_for_expected_run,
    const thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        &expected_returns_by_run_by_mfn_simulated);

#endif // CUDA_PROCESS_TPAW_RUN_X_MFN_SIMULATED_X_MFN_H
