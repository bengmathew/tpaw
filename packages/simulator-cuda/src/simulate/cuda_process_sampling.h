#ifndef CUDA_PROCESS_SAMPLING_H
#define CUDA_PROCESS_SAMPLING_H

#include "src/public_headers/plan_params_cuda.h"
#include <thrust/device_vector.h>

struct SamplingCudaProcessed {
  uint32_t num_runs;
  thrust::device_vector<uint32_t> index_by_run_by_mfn_simulated;
};

SamplingCudaProcessed
cuda_process_sampling(const PlanParamsCuda::Advanced::Sampling &sampling,
                      const uint32_t historical_returns_series_len,
                      const uint32_t num_months,
                      const uint32_t num_months_to_simulate);

#endif // CUDA_PROCESS_SAMPLING_H
