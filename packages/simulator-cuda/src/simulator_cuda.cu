
#include "public_headers/simulator_cuda.h"
#include "simulate/simulate.h"
#include "src/public_headers/result_cuda.h"
#include "src/simulate/plan_params_cuda_vectors.h"

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
}
