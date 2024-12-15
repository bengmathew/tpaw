#ifndef SIMULATE_H
#define SIMULATE_H

#include "src/public_headers/historical_returns_cuda.h"
#include "src/public_headers/plan_params_cuda.h"
#include "src/public_headers/result_cuda.h"
#include "src/simulate/plan_params_cuda_vectors.h"
#include <vector>

namespace simulate {
  ResultCudaNotArrays
  simulate(const uint32_t num_months_to_simulate,
           const CURRENCY current_portfolio_balance,
           const PlanParamsCuda &plan_params,
           const PlanParamsCuda_Vectors &plan_params_vectors,
           ResultCudaArrays &out);
}

#endif // SIMULATE_H
