#ifndef CUDA_PROCESS_HISTORICAL_RETURNS_H
#define CUDA_PROCESS_HISTORICAL_RETURNS_H

#include "src/public_headers/historical_returns_cuda.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include "src/simulate/cuda_process_sampling.h"
#include "src/utils/cuda_utils.h"
#include <thrust/device_vector.h>

template <typename T> struct MonthlyAndAnnual {
  T monthly;
  T annual;

  __device__ __host__ void print(const uint32_t num_tabs) const;
};


struct HistoricalReturnsCudaProcessed {
  thrust::device_vector<StocksAndBondsFLOAT>
      historical_returns_by_run_by_mfn_simulated;
  thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
      expected_returns_by_run_by_mfn_simulated;
  thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
      expected_returns_by_mfn_simulated_for_expected_run;
};

HistoricalReturnsCudaProcessed cuda_process_historical_returns(
    const SamplingCudaProcessed &sampling,
    const uint32_t num_months_to_simulate,
    const thrust::device_vector<HistoricalReturnsCuda>
        &historical_returns_series,
    const StocksAndBondsFLOAT &expected_returns);

#endif // CUDA_PROCESS_HISTORICAL_RETURNS_H
