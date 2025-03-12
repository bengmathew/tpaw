
#include "public_headers/simulator_cuda.h"
#include "simulate/simulate.h"
#include "src/basic_types/stocks_and_bonds.h"
#include "src/get_annualized_stats.h"
#include "src/public_headers/result_cuda.h"
#include "src/simulate/plan_params_cuda_vectors.h"
#include "src/utils/c_array_to_std_vector.h"

extern "C" {

ResultCudaNotArrays
cuda_simulate(const uint32_t num_months_to_simulate,
              const CURRENCY current_portfolio_balance,
              const struct PlanParamsCuda *const plan_params,
              const struct PlanParamsCuda_C_Arrays *const plan_params_c_arrays,
              struct ResultCudaArrays *const out) {
  return simulate::simulate(num_months_to_simulate,
                            current_portfolio_balance,
                            *plan_params,
                            get_plan_params_cuda_vectors(
                                *plan_params_c_arrays,
                                plan_params->ages.simulation_months.num_months),
                            *out);
}

struct StocksAndBondsFLOAT cuda_get_empirical_annual_non_log_mean(
    const uint64_t seed,
    const uint32_t num_runs,
    const uint32_t num_months,
    const FLOAT *const historical_monthly_log_returns_stocks,
    const FLOAT *const historical_monthly_log_returns_bonds,
    const uint32_t historical_returns_series_len) {

  thrust::device_vector<FLOAT> historical_monthly_log_returns_stocks_vec =
      c_array_to_std_vector(historical_monthly_log_returns_stocks,
                            historical_returns_series_len);
  thrust::device_vector<FLOAT> historical_monthly_log_returns_bonds_vec =
      c_array_to_std_vector(historical_monthly_log_returns_bonds,
                            historical_returns_series_len);

  const StocksAndBonds<FLOAT> result = get_empirical_annual_non_log_mean(
      seed,
      num_runs,
      num_months,
      historical_monthly_log_returns_stocks_vec,
      historical_monthly_log_returns_bonds_vec);
  return {
      .stocks = result.stocks,
      .bonds = result.bonds,
  };
}
}
