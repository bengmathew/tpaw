#ifndef GET_RESULT_CUDA_NOT_ARRAY_H
#define GET_RESULT_CUDA_NOT_ARRAY_H

#include "src/public_headers/result_cuda.h"
#include "src/simulate/run/run_result_padded.h"

ResultCudaNotArrays
get_result_cuda_not_array(const uint32_t num_runs,
                          const RunResultPadded &run_result_padded_not_sorted,
                          const CURRENCY tpaw_net_present_value_exact_month_0_legacy);

#endif // GET_RESULT_CUDA_NOT_ARRAY_H
