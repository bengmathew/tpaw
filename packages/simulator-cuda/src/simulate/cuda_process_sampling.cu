#include "curand_kernel.h"
#include "extern/doctest.h"
#include "src/device_utils/get_random_index.h"
#include "src/simulate/cuda_process_sampling.h"
#include "src/utils/bench_utils.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/run_mfn_indexing.h"
#include <extern/nanobench.h>

// Note, it will be more efficient to skip the intermediate
// SamplingCudaProcessed and compute the indices directly during
// cuda_process_historical_returns. But breaking it up keeps the code simpler
// and easier to test. The overhead is not trivial, but tolerable.

namespace {

  struct _MonteCarloBlockState {
    curandStateXORWOW_t curand_state;
    uint32_t next_index;
    uint32_t last_value;
  };

  __device__ __forceinline__ uint32_t
  _next_monte_carlo_index(const uint32_t max_index,
                          const uint32_t block_size,
                          _MonteCarloBlockState &block_state) {
    block_state.last_value =
        block_state.next_index % block_size == 0
            ? get_random_index(max_index, &block_state.curand_state)
            : (block_state.last_value + 1) % max_index;
    block_state.next_index = block_state.next_index + 1;
    return block_state.last_value;
  }

  __global__ void
  _monte_carlo_kernel(const uint32_t num_months_to_simulate,
                      const PlanParamsCuda::Advanced::Sampling::MonteCarlo spec,
                      const uint32_t historical_returns_series_len,
                      uint32_t *index_by_run_by_mfn_simulated) {

    uint32_t run_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (run_index >= spec.num_runs)
      return;

    _MonteCarloBlockState block_state{};
    curand_init(spec.seed, run_index, 0, &block_state.curand_state);
    block_state.next_index = 0;
    block_state.last_value = 0;

    uint32_t stagger =
        spec.stagger_run_starts ? run_index % spec.block_size : 0;
    for (uint32_t month_index = 0; month_index < stagger; month_index++) {
      _next_monte_carlo_index(
          historical_returns_series_len, spec.block_size, block_state);
    }

    for (uint32_t month_index = 0; month_index < num_months_to_simulate;
         month_index++) {
      // Most of computation time is spent in writing to memory here.
      // MONTH_MAJOR is twice as fast as RUN_MAJOR.
      index_by_run_by_mfn_simulated[get_run_by_mfn_index(
          spec.num_runs, num_months_to_simulate, run_index, month_index)] =
          _next_monte_carlo_index(
              historical_returns_series_len, spec.block_size, block_state);
    }
  }

  SamplingCudaProcessed
  _monte_carlo(const PlanParamsCuda::Advanced::Sampling::MonteCarlo &spec,
               const uint32_t historical_returns_series_len,
               const uint32_t num_months_to_simulate) {

    thrust::device_vector<uint32_t> index_by_run_by_mfn_simulated(
        static_cast<size_t>(spec.num_runs * num_months_to_simulate));

    const uint32_t block_size{32};
    _monte_carlo_kernel<<<(spec.num_runs + block_size - 1) / block_size,
                          block_size>>>(
        num_months_to_simulate,
        spec,
        historical_returns_series_len,
        index_by_run_by_mfn_simulated.data().get());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return SamplingCudaProcessed{
        .num_runs = spec.num_runs,
        .index_by_run_by_mfn_simulated =
            std::move(index_by_run_by_mfn_simulated),
    };
  }

  TEST_CASE("bench::cuda_process_sampling") {

    auto do_bench = [&](const uint32_t num_runs, const uint32_t num_months) {
      ankerl::nanobench::Bench()
          .timeUnit(std::chrono::milliseconds{1}, "ms")
          .run("cuda_process_sampling::_monte_carlo::" +
                   std::to_string(num_runs) + "x" + std::to_string(num_months),
               [&]() {
                 _monte_carlo(
                     PlanParamsCuda::Advanced::Sampling::MonteCarlo{
                         .seed = 1234,
                         .num_runs = num_runs,
                         .block_size = 5 * 12,
                         .stagger_run_starts = true,
                     },
                     1000,
                     num_months);
               });
    };
    // for (const auto num_runs : bench_num_runs_vec) {

    for (uint32_t num_runs : {500, 1000, 2000, 20000}) {
      for (const auto num_years : bench_num_years_vec) {
        do_bench(num_runs, num_years * 12);
      }
    }
  }

  TEST_CASE("monte_carlo::block_size=1") {
    const uint32_t num_runs = 200000;
    const uint32_t num_months = 50 * 12;
    const uint32_t historical_returns_series_len = 10;
    const PlanParamsCuda::Advanced::Sampling::MonteCarlo spec{
        .seed = 0,
        .num_runs = num_runs,
        .block_size = 1,
        .stagger_run_starts = false,
    };
    const SamplingCudaProcessed result =
        _monte_carlo(spec, historical_returns_series_len, num_months);

    std::vector<uint32_t> host_result =
        device_vector_to_host(result.index_by_run_by_mfn_simulated);

    std::vector<uint32_t> counts(historical_returns_series_len);
    for (uint32_t i : host_result) {
      counts[i]++;
    }
    std::vector<double> probabilities(historical_returns_series_len);
    for (uint32_t i = 0; i < historical_returns_series_len; ++i) {
      probabilities[i] = static_cast<double>(counts[i]) /
                         static_cast<double>(num_runs * num_months);
    }
    for (uint32_t i = 0; i < historical_returns_series_len; ++i) {
      printf("%.8f\n ",
             abs(1 / static_cast<double>(historical_returns_series_len) -
                 probabilities[i]));
    }
  }

  TEST_CASE("monte_carlo") {
    auto do_test = [&](const bool stagger_run_starts,
                       const std::vector<std::vector<uint32_t>> &truth) {
      const PlanParamsCuda::Advanced::Sampling::MonteCarlo spec{
          .seed = 1234,
          .num_runs = 4,
          .block_size = 3,
          .stagger_run_starts = stagger_run_starts,
      };
      const uint32_t historical_returns_series_len = 1000;
      const uint32_t num_months_to_simulate = 10;
      const SamplingCudaProcessed result = _monte_carlo(
          spec, historical_returns_series_len, num_months_to_simulate);

      std::vector<uint32_t> host_result =
          device_vector_to_host(result.index_by_run_by_mfn_simulated);

      for (uint32_t run = 0; run < result.num_runs; ++run) {
        for (uint32_t month = 0; month < num_months_to_simulate; ++month) {
          CHECK(host_result[get_run_by_mfn_index(
                    result.num_runs, num_months_to_simulate, run, month)] ==
                truth[run][month]);
        }
      }
    };
    SUBCASE("monte_carlo::stagger") {
      do_test(false,
              {
                  {773, 774, 775, 844, 845, 846, 282, 283, 284, 316},
                  {202, 203, 204, 785, 786, 787, 705, 706, 707, 676},
                  {744, 745, 746, 504, 505, 506, 60, 61, 62, 224},
                  {439, 440, 441, 58, 59, 60, 990, 991, 992, 836},
              });
    }
    SUBCASE("monte_carlo::stagger") {
      do_test(true,
              {
                  {773, 774, 775, 844, 845, 846, 282, 283, 284, 316},
                  {203, 204, 785, 786, 787, 705, 706, 707, 676, 677},
                  {746, 504, 505, 506, 60, 61, 62, 224, 225, 226},
                  {439, 440, 441, 58, 59, 60, 990, 991, 992, 836},
              });
    }
  }

  __global__ void _historical_kernel(const uint32_t num_runs,
                                     const uint32_t num_months_to_simulate,
                                     uint32_t *index_by_run_by_mfn_simulated) {
    uint32_t run_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (run_index >= num_runs)
      return;
    for (uint32_t month = 0; month < num_months_to_simulate; ++month) {
      index_by_run_by_mfn_simulated[get_run_by_mfn_index(
          num_runs, num_months_to_simulate, run_index, month)] =
          run_index + month;
    }
  }

  SamplingCudaProcessed
  _historical(const uint32_t historical_returns_series_len,
              const uint32_t num_months,
              const uint32_t num_months_to_simulate) {
    uint32_t num_runs = historical_returns_series_len - (num_months - 1);
    thrust::device_vector<uint32_t> index_by_run_by_mfn_simulated(
        static_cast<size_t>(num_runs * num_months_to_simulate));
    const uint32_t block_size{32};
    _historical_kernel<<<(num_runs + block_size - 1) / block_size,
                         block_size>>>(
        num_runs,
        num_months_to_simulate,
        index_by_run_by_mfn_simulated.data().get());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return SamplingCudaProcessed{
        .num_runs = num_runs,
        .index_by_run_by_mfn_simulated =
            std::move(index_by_run_by_mfn_simulated),
    };
  }

  TEST_CASE("bench::cuda_process_sampling") {
    ankerl::nanobench::Bench()
        .timeUnit(std::chrono::milliseconds{1}, "ms")
        .run("cuda_process_sampling::_historical::1800x120x120",
             [&]() { _historical(1800, 120 * 12, 120 * 12); })
        .run("cuda_process_sampling::_historical::1800x60x60",
             [&]() { _historical(1800, 60 * 12, 60 * 12); });
  }
  TEST_CASE("_historical") {
    uint32_t num_months = 5;
    uint32_t num_months_to_simulate = 2;
    uint32_t historical_returns_series_len = 6;
    auto result = _historical(
        historical_returns_series_len, num_months, num_months_to_simulate);
    std::vector<uint32_t> host_result =
        device_vector_to_host(result.index_by_run_by_mfn_simulated);

    CHECK(result.num_runs == 2);
    std::vector<std::vector<uint32_t>> truth{{0, 1}, {1, 2}};
    for (uint32_t run = 0; run < result.num_runs; ++run) {
      for (uint32_t month = 0; month < num_months_to_simulate; ++month) {
        CHECK(host_result[get_run_by_mfn_index(
                  result.num_runs, num_months_to_simulate, run, month)] ==
              truth[run][month]);
      }
    }
  }
} // namespace

SamplingCudaProcessed
cuda_process_sampling(const PlanParamsCuda::Advanced::Sampling &sampling,
                      const uint32_t historical_returns_series_len,
                      const uint32_t num_months,
                      const uint32_t num_months_to_simulate) {
  if (sampling.type ==
      PlanParamsCuda::Advanced::Sampling::Type::MonteCarloSampling) {

    return _monte_carlo(sampling.monte_carlo_or_historical.monte_carlo,
                        historical_returns_series_len,
                        num_months_to_simulate);
  } else if (sampling.type ==
             PlanParamsCuda::Advanced::Sampling::Type::HistoricalSampling) {
    return _historical(
        historical_returns_series_len, num_months, num_months_to_simulate);
  } else {
    throw std::runtime_error("Unknown sampling type");
  }
}
