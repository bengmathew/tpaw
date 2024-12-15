#ifndef CUDA_PROCESS_SWR_RUN_X_MFN_SIMULATED_X_MFN_H
#define CUDA_PROCESS_SWR_RUN_X_MFN_SIMULATED_X_MFN_H

#include "src/public_headers/numeric_types.h"
#include "src/simulate/cuda_process_historical_returns.h"
#include <stdint.h>
#include <thrust/device_vector.h>

struct Cuda_Processed_SWR_Run_x_MFNSimulated_x_MFN {
  struct Entry {
    struct ApproxNPV {
      // Not including with_current_month. That should be calculated in the
      // simulation for that month because it can be done with full precision.
      // This allows npv values to be consistent within a single month even
      // though they are approximations.
      CURRENCY income_bond_rate_without_current_month;
      __device__ __host__ void print(uint32_t num_tabs) const;
    };

    // Approximation NPV because we calculate with CURRENCY_NPV (i.e. float) and
    // then convert to CURRENCY.
    ApproxNPV npv_approx;

    __device__ __host__ void print(uint32_t num_tabs) const;
  };
  thrust::device_vector<Entry> for_normal_run;
};

Cuda_Processed_SWR_Run_x_MFNSimulated_x_MFN
cuda_process_swr_run_x_mfn_simulated_x_mfn(
    const uint32_t num_runs,
    const uint32_t num_months_to_simulate,
    const uint32_t num_months,
    const std::vector<CURRENCY> &income_by_mfn,
    const thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        &expected_returns_by_run_by_mfn_simulated);

#endif // CUDA_PROCESS_SWR_RUN_X_MFN_SIMULATED_X_MFN_H
