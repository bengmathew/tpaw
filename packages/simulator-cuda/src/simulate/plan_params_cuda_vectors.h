#ifndef PLAN_PARAMS_CUDA_CPP_VECTORS_H
#define PLAN_PARAMS_CUDA_CPP_VECTORS_H

#include "src/public_headers/numeric_types.h"
#include "src/public_headers/plan_params_cuda.h"
#include <vector>

struct PlanParamsCuda_Vectors {
  std::vector<CURRENCY> income_combined_by_mfn;
  std::vector<CURRENCY> essential_expenses_by_mfn;
  std::vector<CURRENCY> discretionary_expenses_by_mfn;
  std::vector<FLOAT> tpaw_rra_including_pos_infinity_by_mfn;
  std::vector<FLOAT> spaw_spending_tilt_by_mfn;
  std::vector<FLOAT> spaw_and_swr_stock_allocation_savings_portfolio_by_mfn;
  std::vector<uint32_t> percentiles;
  std::vector<HistoricalReturnsCuda> historical_returns_series;
};

PlanParamsCuda_Vectors
get_plan_params_cuda_vectors(const PlanParamsCuda_C_Arrays &c_arrays,
                             const uint32_t num_months);

#endif // PLAN_PARAMS_CUDA_CPP_VECTORS_H
