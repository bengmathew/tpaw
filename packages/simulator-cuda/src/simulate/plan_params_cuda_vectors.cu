#include "plan_params_cuda_vectors.h"
#include "src/utils/c_array_to_std_vector.h"
#include <cstdint>

PlanParamsCuda_Vectors
get_plan_params_cuda_vectors(const PlanParamsCuda_C_Arrays &c_arrays,
                             const uint32_t num_months) {

  auto future_savings_by_mfn =
      c_array_to_std_vector(c_arrays.future_savings_by_mfn, num_months);
  auto income_during_retirement_by_mfn = c_array_to_std_vector(
      c_arrays.income_during_retirement_by_mfn, num_months);
  std::vector<CURRENCY> income_combined_by_mfn;
  for (uint32_t i = 0; i < num_months; i++) {
    income_combined_by_mfn.push_back(future_savings_by_mfn[i] +
                                     income_during_retirement_by_mfn[i]);
  }
  return PlanParamsCuda_Vectors{
      .income_combined_by_mfn = income_combined_by_mfn,
      .essential_expenses_by_mfn =
          c_array_to_std_vector(c_arrays.essential_expenses_by_mfn, num_months),
      .discretionary_expenses_by_mfn = c_array_to_std_vector(
          c_arrays.discretionary_expenses_by_mfn, num_months),
      .tpaw_rra_including_pos_infinity_by_mfn = c_array_to_std_vector(
          c_arrays.tpaw_rra_including_pos_infinity_by_mfn, num_months),
      .spaw_spending_tilt_by_mfn =
          c_array_to_std_vector(c_arrays.spaw_spending_tilt_by_mfn, num_months),
      .spaw_and_swr_stock_allocation_savings_portfolio_by_mfn =
          c_array_to_std_vector(
              c_arrays.spaw_and_swr_stock_allocation_savings_portfolio_by_mfn,
              num_months),
      .percentiles =
          c_array_to_std_vector(c_arrays.percentiles, c_arrays.num_percentiles),
      .historical_returns_series =
          c_array_to_std_vector(c_arrays.historical_returns_series,
                                c_arrays.historical_returns_series_len),
  };
}
