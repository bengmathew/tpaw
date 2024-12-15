#include "cuda_process_historical_returns.h"
#include "extern/doctest.h"
#include "extern/nanobench.h"
#include "src/device_utils/monthly_to_annual_rate_device.h"
#include "src/public_headers/historical_returns_cuda.h"
#include "src/public_headers/numeric_types.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include "src/utils/annual_to_monthly_returns.h"
#include "src/utils/bench_utils.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/monthly_to_annual_returns.h"
#include "src/utils/print_public_types.h"
#include <cstdint>
#include <cstdio>
#include <cuda/std/array>

template <typename T>
__device__ __host__ void
MonthlyAndAnnual<T>::print(const uint32_t num_tabs) const {
  printf("%*smonthly:\n", num_tabs * 4, "");
  print_stocks_and_bonds_float(monthly, num_tabs + 1);
  printf("%*sannual:\n", num_tabs * 4, "");
  print_stocks_and_bonds_float(annual, num_tabs + 1);
}

namespace {
  __global__ void
  _kernel(const uint32_t num_runs,
          const uint32_t num_months_to_simulate,
          const uint32_t *index_by_run_by_mfn_simulated,
          const HistoricalReturnsCuda *historical_returns_series,
          StocksAndBondsFLOAT expected_returns_at_month_0,
          // out
          StocksAndBondsFLOAT *historical_returns_by_run_by_mfn_simulated,
          MonthlyAndAnnual<StocksAndBondsFLOAT>
              *expected_returns_by_run_by_mfn_simulated,
          MonthlyAndAnnual<StocksAndBondsFLOAT>
              *expected_returns_by_mfn_simulated_for_expected_run) {

    uint32_t run_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (run_index >= num_runs)
      return;

    StocksAndBondsFLOAT curr_expected_returns{expected_returns_at_month_0};
    const MonthlyAndAnnual<StocksAndBondsFLOAT>
        annual_and_monthly_expected_returns_at_month_0{
            .monthly = expected_returns_at_month_0,
            .annual =
                StocksAndBondsFLOAT{
                    .stocks = monthly_to_annual_rate_device(
                        expected_returns_at_month_0.stocks),
                    .bonds = monthly_to_annual_rate_device(
                        expected_returns_at_month_0.bonds),
                },
        };

    for (uint32_t month_index = 0; month_index < num_months_to_simulate;
         month_index++) {
      uint32_t index = get_run_by_mfn_index(
          num_runs, num_months_to_simulate, run_index, month_index);

      uint32_t index_into_historical_returns =
          index_by_run_by_mfn_simulated[index];
      HistoricalReturnsCuda historical_returns_for_index =
          historical_returns_series[index_into_historical_returns];

      historical_returns_by_run_by_mfn_simulated[index] = StocksAndBondsFLOAT{
          .stocks = historical_returns_for_index.stocks.returns,
          .bonds = historical_returns_for_index.bonds.returns};

      curr_expected_returns.stocks +=
          historical_returns_for_index.stocks.expected_return_change;
      curr_expected_returns.bonds +=
          historical_returns_for_index.bonds.expected_return_change;
      expected_returns_by_run_by_mfn_simulated[index] =
          MonthlyAndAnnual<StocksAndBondsFLOAT>{
              .monthly = curr_expected_returns,
              .annual =
                  StocksAndBondsFLOAT{
                      .stocks = monthly_to_annual_rate_device(
                          curr_expected_returns.stocks),
                      .bonds = monthly_to_annual_rate_device(
                          curr_expected_returns.bonds),
                  },
          };
      if (run_index == 0) {
        expected_returns_by_mfn_simulated_for_expected_run[month_index] =
            annual_and_monthly_expected_returns_at_month_0;
      }
    }
  }

  std::vector<HistoricalReturnsCuda>
  _get_test_historical_returns_series(const uint32_t size) {
    std::vector<HistoricalReturnsCuda> result(
        size,
        HistoricalReturnsCuda{
            .stocks =
                HistoricalReturnsCuda::Part{
                    .returns = FLOAT_L(0.0),
                    .expected_return_change = FLOAT_L(0.0),
                },
            .bonds =
                HistoricalReturnsCuda::Part{
                    .returns = FLOAT_L(0.0),
                    .expected_return_change = FLOAT_L(0.0),
                },
        });

    for (uint32_t i = 0; i < result.size(); i++) {
      result[i] = HistoricalReturnsCuda{
          .stocks =
              HistoricalReturnsCuda::Part{
                  .returns = static_cast<FLOAT>(i) + FLOAT_L(0.0),
                  .expected_return_change =
                      static_cast<FLOAT>(i) + FLOAT_L(1.0),
              },
          .bonds =
              HistoricalReturnsCuda::Part{
                  .returns = static_cast<FLOAT>(i) + FLOAT_L(2.0),
                  .expected_return_change =
                      static_cast<FLOAT>(i) + FLOAT_L(3.0),
              },
      };
    }
    return result;
  }

} // namespace

HistoricalReturnsCudaProcessed cuda_process_historical_returns(
    const SamplingCudaProcessed &sampling,
    const uint32_t num_months_to_simulate,
    const thrust::device_vector<HistoricalReturnsCuda>
        &historical_returns_series,
    const StocksAndBondsFLOAT &expected_returns_at_month_0) {

  HistoricalReturnsCudaProcessed result{
      .historical_returns_by_run_by_mfn_simulated =
          thrust::device_vector<StocksAndBondsFLOAT>(
              static_cast<size_t>(sampling.num_runs * num_months_to_simulate)),
      // TODO: When implementing duration matching consider if this is expected
      // return after or during the month at index and think carefully about the
      // value at the start of each run.
      .expected_returns_by_run_by_mfn_simulated =
          thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>(
              static_cast<size_t>(sampling.num_runs * num_months_to_simulate)),
      .expected_returns_by_mfn_simulated_for_expected_run =
          thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>(
              static_cast<size_t>(num_months_to_simulate)),
  };

  const uint32_t block_size{32};
  _kernel<<<(sampling.num_runs + block_size - 1) / block_size, block_size>>>(
      sampling.num_runs,
      num_months_to_simulate,
      sampling.index_by_run_by_mfn_simulated.data().get(),
      historical_returns_series.data().get(),
      expected_returns_at_month_0,
      result.historical_returns_by_run_by_mfn_simulated.data().get(),
      result.expected_returns_by_run_by_mfn_simulated.data().get(),
      result.expected_returns_by_mfn_simulated_for_expected_run.data().get());
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return result;
}

namespace {

  TEST_CASE("cuda_process_historical_returns") {
    const uint32_t num_runs = 3;
    const uint32_t num_months_to_simulate = 4;
    const MonthlyAndAnnual<StocksAndBondsFLOAT> expected_returns{
        .monthly = {.stocks = annual_to_monthly_return(FLOAT_L(0.05)),
                    .bonds = annual_to_monthly_return(FLOAT_L(0.03))},
        .annual = {.stocks = FLOAT_L(0.05), .bonds = FLOAT_L(0.03)},
    };

    std::vector<uint32_t> index_by_run_by_mfn_host(
        static_cast<size_t>(num_runs * num_months_to_simulate));
    {
      const auto _set = [&](const uint32_t run_index,
                            const uint32_t month_index,
                            const uint32_t value) {
        index_by_run_by_mfn_host[get_run_by_mfn_index(
            num_runs, num_months_to_simulate, run_index, month_index)] = value;
      };
      _set(0, 0, 0);
      _set(0, 1, 1);
      _set(0, 2, 2);
      _set(1, 0, 3);
      _set(1, 1, 4);
      _set(1, 2, 5);
      _set(2, 0, 6);
      _set(2, 1, 7);
      _set(2, 2, 8);
    }
    SamplingCudaProcessed sampling{
        .num_runs = num_runs,
        .index_by_run_by_mfn_simulated = index_by_run_by_mfn_host,
    };

    std::vector<HistoricalReturnsCuda> historical_returns_series_host =
        _get_test_historical_returns_series(9);
    thrust::device_vector<HistoricalReturnsCuda> historical_returns_series_gpu =
        historical_returns_series_host;

    HistoricalReturnsCudaProcessed processed =
        cuda_process_historical_returns(sampling,
                                        num_months_to_simulate,
                                        historical_returns_series_gpu,
                                        expected_returns.monthly);
    std::vector<StocksAndBondsFLOAT>
        historical_returns_by_run_by_mfn_simulated = device_vector_to_host(
            processed.historical_returns_by_run_by_mfn_simulated);
    std::vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        expected_returns_by_run_by_mfn_simulated = device_vector_to_host(
            processed.expected_returns_by_run_by_mfn_simulated);
    std::vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        expected_returns_by_mfn_simulated_for_expected_run =
            device_vector_to_host(
                processed.expected_returns_by_mfn_simulated_for_expected_run);

    for (uint32_t run_index = 0; run_index < num_runs; run_index++) {
      StocksAndBondsFLOAT curr_expected_returns{expected_returns.monthly};
      for (uint32_t month_index = 0; month_index < num_months_to_simulate;
           month_index++) {

        const HistoricalReturnsCuda curr_historical_returns =
            historical_returns_series_host
                [index_by_run_by_mfn_host[get_run_by_mfn_index(
                    num_runs, num_months_to_simulate, run_index, month_index)]];

        curr_expected_returns.stocks +=
            curr_historical_returns.stocks.expected_return_change;
        curr_expected_returns.bonds +=
            curr_historical_returns.bonds.expected_return_change;
        const uint32_t index_in_result = get_run_by_mfn_index(
            num_runs, num_months_to_simulate, run_index, month_index);
        CHECK(historical_returns_by_run_by_mfn_simulated[index_in_result]
                  .stocks == curr_historical_returns.stocks.returns);
        CHECK(
            historical_returns_by_run_by_mfn_simulated[index_in_result].bonds ==
            curr_historical_returns.bonds.returns);
        CHECK(expected_returns_by_run_by_mfn_simulated[index_in_result]
                  .monthly.stocks == curr_expected_returns.stocks);
        CHECK(expected_returns_by_run_by_mfn_simulated[index_in_result]
                  .monthly.bonds == curr_expected_returns.bonds);
        CHECK(expected_returns_by_run_by_mfn_simulated[index_in_result]
                  .annual.stocks == doctest::Approx(monthly_to_annual_return(
                                        curr_expected_returns.stocks)));
        CHECK(expected_returns_by_run_by_mfn_simulated[index_in_result]
                  .annual.bonds == doctest::Approx(monthly_to_annual_return(
                                       curr_expected_returns.bonds)));
        CHECK(expected_returns_by_mfn_simulated_for_expected_run[month_index]
                  .monthly.stocks == expected_returns.monthly.stocks);
        CHECK(expected_returns_by_mfn_simulated_for_expected_run[month_index]
                  .monthly.bonds == expected_returns.monthly.bonds);
        CHECK(expected_returns_by_mfn_simulated_for_expected_run[month_index]
                  .annual.stocks == doctest::Approx(monthly_to_annual_return(
                                        expected_returns.monthly.stocks)));
        CHECK(expected_returns_by_mfn_simulated_for_expected_run[month_index]
                  .annual.bonds == doctest::Approx(monthly_to_annual_return(
                                       expected_returns.monthly.bonds)));
      }
    }
  }

  TEST_CASE("bench::cuda_process_historical_returns") {
    auto do_bench = [&](const uint32_t num_runs,
                        const uint32_t num_months_to_simulate) {
      const StocksAndBondsFLOAT expected_returns{
          .stocks = annual_to_monthly_return(FLOAT_L(0.05)),
          .bonds = annual_to_monthly_return(FLOAT_L(0.03)),
      };

      std::vector<HistoricalReturnsCuda> historical_returns_series_host =
          _get_test_historical_returns_series(2000);
      thrust::device_vector<HistoricalReturnsCuda>
          historical_returns_series_gpu = historical_returns_series_host;

      SamplingCudaProcessed sampling = cuda_process_sampling(
          PlanParamsCuda::Advanced::Sampling{
              .type =
                  PlanParamsCuda::Advanced::Sampling::Type::MonteCarloSampling,
              .monte_carlo_or_historical =
                  {PlanParamsCuda::Advanced::Sampling::MonteCarlo{
                      .seed = 1234,
                      .num_runs = num_runs,
                      .block_size = 5 * 12,
                      .stagger_run_starts = true,
                  }}},
          historical_returns_series_host.size(),
          num_months_to_simulate,
          num_months_to_simulate);

      ankerl::nanobench::Bench()
          .timeUnit(std::chrono::milliseconds{1}, "ms")
          .run("cuda_process_historical_returns::" + std::to_string(num_runs) +
                   "x" + std::to_string(num_months_to_simulate),
               [&]() {
                 cuda_process_historical_returns(sampling,
                                                 num_months_to_simulate,
                                                 historical_returns_series_gpu,
                                                 expected_returns);
               });
    };
    for (const auto num_runs : bench_num_runs_vec) {
      for (const auto num_years : bench_num_years_vec) {
        do_bench(num_runs, num_years * 12);
      }
    }
  }
} // namespace