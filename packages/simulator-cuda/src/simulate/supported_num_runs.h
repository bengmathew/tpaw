#ifndef SUPPORTED_NUM_RUNS_H
#define SUPPORTED_NUM_RUNS_H

// This should match block_sort::get_fn Keep this list small, it affects
// compilation time for the templatized block_sort.

constexpr uint32_t SUPPORTED_NUM_RUNS[] = {
    1, 20, 500, 1000, 2000, 5000,
    // NOTE: Unable to add 25000 and 50000 due block_sort::fn templating
    // failure. Compilation takes a long time and fails with "nvlink error, uses
    // too much shared data (for sm_80 target)".
};

constexpr uint32_t get_padded_num_runs_for_block_size(uint32_t num_runs,
                                                      uint32_t block_size) {
  // Round up to the nearest SUPPORTED_NUM_RUNS. This is to support historical
  // simulation. Those don't align exactly with SUPPORTED_NUM_RUNS.
  uint32_t num_runs_rounded_up = 0;
  for (uint32_t supported_num_runs : SUPPORTED_NUM_RUNS) {
    if (supported_num_runs >= num_runs) {
      num_runs_rounded_up = supported_num_runs;
      break;
    }
  }
  return ((num_runs_rounded_up + block_size - 1) / block_size) * block_size;
}

constexpr uint32_t BLOCK_SORT_BLOCK_SIZE = 32;
constexpr uint32_t get_padded_num_runs(uint32_t num_runs) {
  return get_padded_num_runs_for_block_size(num_runs, BLOCK_SORT_BLOCK_SIZE);
}

#endif // SUPPORTED_NUM_RUNS_H
