#ifndef SIMULATOR_CUDA_H
#define SIMULATOR_CUDA_H

#include "numeric_types.h"
#include "plan_params_cuda.h"
#include "result_cuda.h"
#include <stdint.h> // bindgen cannot find cstdint (needed for sized number types).

// bindgen (to generate rust bindings) has trouble with extern "C"
// because it is expecting C not C++ and extern "C" is a C++ thing.
#ifdef __cplusplus
extern "C" {
#endif

struct ResultCudaNotArrays
cuda_simulate(const uint32_t num_months_to_simulate,
              const CURRENCY current_portfolio_balance,
              const struct PlanParamsCuda *const plan_params,
              const struct PlanParamsCuda_C_Arrays *const plan_params_c_arrays,
              struct ResultCudaArrays *const out);

#ifdef __cplusplus
}
#endif

#endif // SIMULATOR_CUDA_H
