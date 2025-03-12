
#include "extern/nanobench.h"
#include "sort_run_result_padded.h"
#include "src/simulate/run/run_result_padded.h"
#include "src/simulate/supported_num_runs.h"
#include "src/utils/bench_utils.h"
#include "src/utils/cuda_utils.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <extern/doctest.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <utility>

// From: https://nvidia.github.io/cccl/cub/index.html
//
// keys are split into sections of BLOCK_SIZE * KEYS_PER_THREAD and each section
// is sorted separately. BLOCK_SIZE should match kernel launch configuration
//
// BlockRadixSort works with float: From doc: BlockRadixSort can sort all of the
// built-in C++ numeric primitive types (``unsigned char``, ``int``, ``double``,
// etc.) as well as CUDA's ``__half`` half-precision floating-point type.
template <int BLOCK_SIZE, // aka THREADS_PER_BLOCK
          int KEYS_PER_THREAD,
          typename T>
__global__ void _kernel(T *const keys) {
  // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
  using BlockLoadT =
      cub::BlockLoad<T, BLOCK_SIZE, KEYS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = cub::
      BlockStore<T, BLOCK_SIZE, KEYS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE>;
  using BlockRadixSortT = cub::BlockRadixSort<T, BLOCK_SIZE, KEYS_PER_THREAD>;

  // Allocate type-safe, repurposable shared memory for collectives
  __shared__ union {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockRadixSortT::TempStorage sort;
  } temp_storage;

  // Obtain this block's segment of consecutive keys (blocked across threads)
  T thread_keys[KEYS_PER_THREAD];
  int block_offset = blockIdx.x * (BLOCK_SIZE * KEYS_PER_THREAD);
  BlockLoadT(temp_storage.load).Load(keys + block_offset, thread_keys);
  __syncthreads(); // Barrier for smem reuse

  // Collectively sort the keys
  BlockRadixSortT(temp_storage.sort).Sort(thread_keys);
  __syncthreads(); // Barrier for smem reuse

  // Store the sorted segment
  BlockStoreT(temp_storage.store).Store(keys + block_offset, thread_keys);
}

template <uint32_t BLOCK_SIZE, uint32_t KEYS_PER_THREAD, typename T>
void _sort_single(thrust::device_vector<T> &data_month_major,
                  uint32_t num_months_to_simulate) {
  assert(data_month_major.size() % num_months_to_simulate == 0);
  const uint32_t num_runs = data_month_major.size() / num_months_to_simulate;
  assert(num_runs == BLOCK_SIZE * KEYS_PER_THREAD);

  // thrust::device_vector<T> out(data_month_major.size());

  _kernel<BLOCK_SIZE, KEYS_PER_THREAD, T>
      <<<num_months_to_simulate, BLOCK_SIZE>>>(data_month_major.data().get());
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  // return out;
}
template <typename T>
using FnType = void (*)(thrust::device_vector<T> &data_month_major,
                        uint32_t num_months_to_simulate);

template <typename T, uint32_t BLOCK_SIZE>
FnType<T> _get_sort_single_for_num_runs(const uint32_t num_runs) {
  // This function should match SUPPORTED_NUM_RUNS.
  assert(SUPPORTED_NUM_RUNS[0] == 1);
  assert(SUPPORTED_NUM_RUNS[1] == 20);
  assert(SUPPORTED_NUM_RUNS[2] == 500);
  assert(SUPPORTED_NUM_RUNS[3] == 1000);
  assert(SUPPORTED_NUM_RUNS[4] == 2000);
  assert(SUPPORTED_NUM_RUNS[5] == 5000);

  // NOTE:  Possibly a surprisingly large impact on compilation times for each
  // of the template instantiations, so this has to be kept limited. Due to
  // BlockStore<> BlockLoad<> BlockRadixSort<> instantiations being very
  // expensive?
  if (num_runs <= 1) {
    return &_sort_single<BLOCK_SIZE, (1 + BLOCK_SIZE - 1) / BLOCK_SIZE, T>;
  } else if (num_runs <= 20) {
    return &_sort_single<BLOCK_SIZE, (20 + BLOCK_SIZE - 1) / BLOCK_SIZE, T>;
  } else if (num_runs <= 500) {
    return &_sort_single<BLOCK_SIZE, (500 + BLOCK_SIZE - 1) / BLOCK_SIZE, T>;
  } else if (num_runs <= 1000) {
    return &_sort_single<BLOCK_SIZE, (1000 + BLOCK_SIZE - 1) / BLOCK_SIZE, T>;
  } else if (num_runs <= 2000) {
    return &_sort_single<BLOCK_SIZE, (2000 + BLOCK_SIZE - 1) / BLOCK_SIZE, T>;
  } else if (num_runs <= 5000) {
    return &_sort_single<BLOCK_SIZE, (5000 + BLOCK_SIZE - 1) / BLOCK_SIZE, T>;
  }
  // NOTE: Unable to add 25000 and 50000 due block_sort::fn templating
  // failure. Compilation takes a long time and fails with "nvlink error, uses
  // too much shared data (for sm_80 target)".
  //
  /// else if (num_runs == 25000) {
  //   return &fn<BLOCK_SIZE, (25000 + BLOCK_SIZE - 1) / BLOCK_SIZE, T>;
  // }
  // else if (num_runs == 50000) {
  // return &fn<BLOCK_SIZE, (50000 + BLOCK_SIZE - 1) / BLOCK_SIZE, T>;
  // }
  throw std::runtime_error("Should never reach here");
}

TEST_CASE("block_sort") {
  constexpr uint32_t BLOCK_SIZE = 32;
  constexpr uint32_t KEYS_PER_THREAD = 1;
  const uint32_t num_runs = BLOCK_SIZE * KEYS_PER_THREAD;
  const uint32_t num_months = 2;

  auto data_by_run_by_mfn_month_major_host =
      RunResultPadded::get_test_data_single<CURRENCY>(num_runs, num_months);
  thrust::device_vector<CURRENCY> data_by_run_by_mfn_month_major_device(
      data_by_run_by_mfn_month_major_host);

  _sort_single<BLOCK_SIZE, KEYS_PER_THREAD, CURRENCY>(
      data_by_run_by_mfn_month_major_device, num_months);

  std::vector<CURRENCY> sorted_host =
      device_vector_to_host(data_by_run_by_mfn_month_major_device);

  std::vector<CURRENCY> expected_sorted_host(
      static_cast<size_t>(num_runs * num_months));
  for (uint32_t month_index = 0; month_index < num_months; month_index++) {
    CURRENCY curr = static_cast<CURRENCY>(month_index + 1) * 10000;
    for (uint32_t run_index = 0; run_index < num_runs; run_index++) {
      curr++;
      expected_sorted_host[month_index * num_runs + run_index] = curr;
    }
  }

  CHECK(sorted_host == expected_sorted_host);
}

TEST_CASE("block_sort::double") {
  constexpr uint32_t BLOCK_SIZE = 32;
  constexpr uint32_t KEYS_PER_THREAD = 1;
  const uint32_t num_runs = BLOCK_SIZE * KEYS_PER_THREAD;
  const uint32_t num_months = 1;

  std::vector<double> data_by_run_by_mfn_month_major_host{
      1.0, -0.024, 0.024, 0.0, -3.24, 5.0};
  while (data_by_run_by_mfn_month_major_host.size() <
         static_cast<size_t>(num_runs * num_months)) {
    data_by_run_by_mfn_month_major_host.push_back(1.7976931348623157E+308);
  }
  // std::cout << "Input data: ";
  // for (const auto &value : data_by_run_by_mfn_month_major_host) {
  //   std::cout << value << " ";
  // }
  // std::cout << std::endl;

  thrust::device_vector<double> data_by_run_by_mfn_month_major_device(
      data_by_run_by_mfn_month_major_host);

  // const auto sorted_gpu =
  _sort_single<BLOCK_SIZE, KEYS_PER_THREAD, double>(
      data_by_run_by_mfn_month_major_device, num_months);

  std::vector<double> sorted_host =
      device_vector_to_host(data_by_run_by_mfn_month_major_device);

  std::vector<double> expected_sorted_host{-3.24, -0.024, 0.0, 0.024, 1.0, 5.0};
  while (expected_sorted_host.size() <
         static_cast<size_t>(num_runs * num_months)) {
    expected_sorted_host.push_back(1.7976931348623157E+308);
  }
  // std::cout << "Input data: ";
  // for (const auto &value : sorted_host) {
  //   std::cout << value << " ";
  // }
  // std::cout << std::endl;
  CHECK(sorted_host == expected_sorted_host);
}

TEST_CASE("bench::block_sort") {
  auto do_bench =
      []<typename T, uint32_t BLOCK_SIZE>(std::string name) -> void {
    const auto do_bench =
        [](std::string name, uint32_t num_runs, uint32_t num_months) -> void {
      auto data_host = RunResultPadded::get_test_data_single<T>(
          get_padded_num_runs_for_block_size(num_runs, BLOCK_SIZE), num_months);
      thrust::device_vector<T> data_device(data_host);
      ankerl::nanobench::Bench()
          .timeUnit(std::chrono::milliseconds{1}, "ms")
          .run(name.c_str(), [&]() {
            _get_sort_single_for_num_runs<T, BLOCK_SIZE>(num_runs)(data_device,
                                                                   num_months);
          });
    };

    for (auto num_runs : bench_num_runs_vec) {
      for (auto num_years : bench_num_years_vec) {
        do_bench((name + std::to_string(num_runs) + " x " +
                  std::to_string(num_years))
                     .c_str(),
                 num_runs,
                 num_years * 12);
      }
    }
  };

  do_bench.template operator()<int64_t, BLOCK_SORT_BLOCK_SIZE>(
      "block_sort:: default x int64_t x ");
  do_bench.template operator()<int32_t, BLOCK_SORT_BLOCK_SIZE>(
      "block_sort:: default x int32_t x ");
  do_bench.template operator()<double, BLOCK_SORT_BLOCK_SIZE>(
      "block_sort:: default x double x ");

  // Comment out what is not tested, this creates a lot of templates that
  // slows down compilation.
  //
  // do_bench.template operator()<int64_t, 32>("block_sort::32 x int64_t x ");
  // do_bench.template operator()<int64_t, 64>("block_sort::64 x int64_t x ");
  // do_bench.template operator()<int64_t, 128>("block_sort::128 x int64_t x
  // "); do_bench.template operator()<int64_t, 256>("block_sort::256 x int64_t
  // x "); do_bench.template operator()<int64_t, 512>("block_sort::512 x
  // int64_t x ");

  // do_bench.template operator()<int32_t, 32>("block_sort::32 x int32_t x ");
  // do_bench.template operator()<int32_t, 64>("block_sort::64 x int32_t x ");
  // do_bench.template operator()<int32_t, 128>("block_sort::128 x int32_t x
  // "); do_bench.template operator()<int32_t, 256>("block_sort::256 x int32_t
  // x "); do_bench.template operator()<int32_t, 512>("block_sort::512 x
  // int32_t x ");
}

void sort_run_result_padded(RunResultPadded &run_result_padded,
                            const uint32_t num_runs,
                            const uint32_t num_months_to_simulate) {

  const auto fn_full =
      [&]<typename T>(thrust::device_vector<T> &data_month_major) {
        _get_sort_single_for_num_runs<T, BLOCK_SORT_BLOCK_SIZE>(num_runs)(
            data_month_major, num_months_to_simulate);
      };

  const auto fn_single =
      [&]<typename T>(thrust::device_vector<T> &data_month_major) {
        _get_sort_single_for_num_runs<T, BLOCK_SORT_BLOCK_SIZE>(num_runs)(
            data_month_major, 1);
      };

  fn_full(run_result_padded.by_run_by_mfn_simulated_month_major_balance_start);
  fn_full(run_result_padded
              .by_run_by_mfn_simulated_month_major_withdrawals_essential);
  fn_full(run_result_padded
              .by_run_by_mfn_simulated_month_major_withdrawals_discretionary);
  fn_full(run_result_padded
              .by_run_by_mfn_simulated_month_major_withdrawals_general);
  fn_full(
      run_result_padded.by_run_by_mfn_simulated_month_major_withdrawals_total);
  fn_full(
      run_result_padded
          .by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate);
  fn_full(
      run_result_padded
          .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio);
  fn_full(
      run_result_padded
          .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth);
  fn_full(
      run_result_padded.tpaw_by_run_by_mfn_simulated_month_major_spending_tilt);
  fn_single(run_result_padded.by_run_ending_balance);
}

TEST_CASE("bench::sort_run_result_padded") {
  auto do_bench = []<uint32_t BLOCK_SIZE>(std::string name) {
    const auto do_bench =
        [](std::string name, uint32_t num_runs, uint32_t num_months) {
          RunResultPadded run_result_padded =
              RunResultPadded::get_test_data_for_block_size(
                  num_runs, num_months, BLOCK_SIZE);

          ankerl::nanobench::Bench()
              .timeUnit(std::chrono::milliseconds{1}, "ms")
              .run(name.c_str(), [&]() {
                sort_run_result_padded(run_result_padded, num_runs, num_months);
              });
        };

    for (auto num_runs : bench_num_runs_vec) {
      for (auto num_years : bench_num_years_vec) {
        do_bench((name + std::to_string(num_runs) + " x " +
                  std::to_string(num_years))
                     .c_str(),
                 num_runs,
                 num_years * 12);
      }
    }
  };

  do_bench.template operator()<BLOCK_SORT_BLOCK_SIZE>(
      "sort_run_result_padded:: default x int64_t x ");
}
