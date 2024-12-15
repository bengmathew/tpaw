
#ifndef PICK_PERCENTILES_H
#define PICK_PERCENTILES_H

#include "src/public_headers/result_cuda.h"
#include "src/simulate/run/run_result_padded.h"

void pick_percentiles(const RunResultPadded &run_result_padded_sorted,
                      const uint32_t num_runs,
                      const std::vector<uint32_t> &percentiles,
                      const uint32_t num_months_simulated,
                      ResultCudaArrays &result_cuda);

#endif // PICK_PERCENTILES_H
