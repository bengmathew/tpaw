// #include "extern/doctest.h"
// #include "extern/nanobench.h"
// #include "src/public_headers/returns_stats.h"
// #include "src/public_headers/stocks_and_bonds_float.h"
// #include "src/utils/bench_utils.h"
// #include "src/utils/cuda_utils.h"
// #include "src/utils/print_public_types.h"
// #include <cstdint>
// #include <cuda/std/array>

// namespace {
//   template <bool for_stocks>
//   __global__ void _stats_kernel(
//       const uint32_t num_runs,
//       const uint32_t num_months_to_simulate,
//       const StocksAndBondsFLOAT
//           *const monthly_non_log_returns_by_run_by_mfn_simulated,
//       // This is just memory for each thread to use for intermediate values.
//       // There is probably a better way to do this. FEATURE: Optimize.
//       FLOAT2 *const mem_by_run_by_annualized_mfn_simulated,
//       // out
//       ReturnsStats::LogAndNonLogStats *annualized_stats_by_run) {

//     uint32_t run_index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (run_index >= num_runs)
//       return;

//     const uint32_t num_years_from_annualization = num_months_to_simulate -
//     11;

//     cuda::std::array<FLOAT, 12> last_12_monthly_log_returns{};

//     FLOAT annualized_log_return{FLOAT_L(0.0)};
//     FLOAT annualized_log_return_sum{FLOAT_L(0.0)};
//     FLOAT annualized_non_log_return_sum{FLOAT_L(0.0)};

//     for (uint32_t month_index = 0; month_index < num_months_to_simulate;
//          month_index++) {
//       const uint32_t index = get_run_by_mfn_index(
//           num_runs, num_months_to_simulate, run_index, month_index);

//       const uint32_t index_into_last_12_monthly_log_returns = month_index %
//       12; if (month_index >= 12) {
//         annualized_log_return -=
//             last_12_monthly_log_returns[index_into_last_12_monthly_log_returns];
//       }

//       const FLOAT monthly_non_log_return =
//           for_stocks
//               ? monthly_non_log_returns_by_run_by_mfn_simulated[index].stocks
//               : monthly_non_log_returns_by_run_by_mfn_simulated[index].bonds;
//       const FLOAT monthly_log_return =
//           __FLOAT_LOG(monthly_non_log_return + FLOAT_L(1.0));

//       last_12_monthly_log_returns[index_into_last_12_monthly_log_returns] =
//           monthly_log_return;
//       annualized_log_return += monthly_log_return;

//       if (month_index >= 11) {
//         const uint32_t annualized_index =
//             get_run_by_mfn_index(num_runs,
//                                  num_years_from_annualization,
//                                  run_index,
//                                  month_index - 11);

//         const FLOAT annualized_non_log_return =
//             __FLOAT_EXP(annualized_log_return) - FLOAT_L(1.0);

//         mem_by_run_by_annualized_mfn_simulated[annualized_index] =
//             FLOAT2{annualized_log_return, annualized_non_log_return};

//         annualized_log_return_sum += annualized_log_return;
//         annualized_non_log_return_sum += annualized_non_log_return;
//       }
//     }

//     const FLOAT annualized_log_return_mean =
//         __FLOAT_DIVIDE(annualized_log_return_sum,
//                        static_cast<FLOAT>(num_years_from_annualization));
//     const FLOAT annualized_non_log_return_mean =
//         __FLOAT_DIVIDE(annualized_non_log_return_sum,
//                        static_cast<FLOAT>(num_years_from_annualization));

//     for (uint32_t month_index = 0; month_index < num_months_to_simulate - 11;
//          month_index++) {
//       const uint32_t annualized_index = get_run_by_mfn_index(
//           num_runs, num_years_from_annualization, run_index, month_index);
//       const FLOAT2 mem =
//           mem_by_run_by_annualized_mfn_simulated[annualized_index];
//       const FLOAT annualized_log_return = mem.x;
//       const FLOAT annualized_non_log_return = mem.y;

//       annualized_log_return_variance_sum += __FLOAT_POWF(
//           annualized_log_return - annualized_log_return_mean, FLOAT_L(2.0));
//       annualized_non_log_return_variance_sum += __FLOAT_POWF(
//           annualized_non_log_return - annualized_non_log_return_mean,
//           FLOAT_L(2.0));
//     }

//     annualized_stats_by_run[run_index] = {
//         .log = {.n = num_years_from_annualization,
//                 .mean = annualized_log_return_mean,
//                 .variance = __FLOAT_DIVIDE(
//                     annualized_log_return_variance_sum,
//                     static_cast<FLOAT>(num_years_from_annualization - 1))},
//         .non_log = {.n = num_years_from_annualization,
//                     .mean = annualized_non_log_return_mean,
//                     .variance = __FLOAT_DIVIDE(
//                         annualized_non_log_return_variance_sum,
//                         static_cast<FLOAT>(num_years_from_annualization -
//                         1))}};
//   }

//   ReturnsStats::LogAndNonLogStats
//   _combine_stats(thrust::device_vector<ReturnsStats::LogAndNonLogStats>
//                      &stats_by_run_device) {
//     ReturnsStats::LogAndNonLogStats result{
//         .log = {.n = 0, .mean = FLOAT_L(0.0), .variance = FLOAT_L(0.0)},
//         .non_log = {.n = 0, .mean = FLOAT_L(0.0), .variance = FLOAT_L(0.0)}};
//     std::vector<ReturnsStats::LogAndNonLogStats> stats_by_run_host(
//         stats_by_run_device.size());
//     thrust::copy(stats_by_run_device.begin(),
//                  stats_by_run_device.end(),
//                  stats_by_run_host.begin());
//     result.log.n = stats_by_run_host[0].log.n;
//     result.non_log.n = stats_by_run_host[0].non_log.n;
//     for (const auto &stats : stats_by_run_host) {
//       result.log.mean += stats.log.mean;
//       result.non_log.mean += stats.non_log.mean;
//       result.log.variance += stats.log.variance;
//       result.non_log.variance += stats.non_log.variance;
//     }
//     result.log.mean =
//         result.log.mean / static_cast<FLOAT>(stats_by_run_host.size());
//     result.non_log.mean =
//         result.non_log.mean / static_cast<FLOAT>(stats_by_run_host.size());
//     result.log.variance =
//         result.log.variance / static_cast<FLOAT>(stats_by_run_host.size());
//     result.non_log.variance =
//         result.non_log.variance /
//         static_cast<FLOAT>(stats_by_run_host.size());
//     return result;
//   }

// } // namespace

// ReturnsStats get_historical_returns_annualized_stats(
//     const uint32_t num_runs,
//     const uint32_t num_months_to_simulate,
//     const thrust_device_vector_no_init<StocksAndBondsFLOAT>
//         &historical_returns_by_run_by_mfn_simulated) {
//   const auto _helper =
//       [num_months_to_simulate,
//        num_runs,
//        &historical_returns_by_run_by_mfn_simulated]<bool for_stocks>() {
//         thrust::device_vector<FLOAT2> mem_by_run_by_annualized_mfn_simulated(
//             static_cast<size_t>(num_runs * (num_months_to_simulate - 11)));

//         thrust::device_vector<ReturnsStats::LogAndNonLogStats>
//             annualized_stats_by_run(static_cast<size_t>(num_runs));
//         const uint32_t block_size{32};
//         _stats_kernel<for_stocks>
//             <<<(num_runs + block_size - 1) / block_size, block_size>>>(
//                 num_runs,
//                 num_months_to_simulate,
//                 historical_returns_by_run_by_mfn_simulated.data().get(),
//                 mem_by_run_by_annualized_mfn_simulated.data().get(),
//                 annualized_stats_by_run.data().get());
//         return _combine_stats(annualized_stats_by_run);
//       };
//   return {
//       .stocks = _helper.operator()<true>(),
//       //   .bonds = _helper.operator()<false>(),
//       // TODO:
//       .bonds = {
//           .log = {.n = 0, .mean = FLOAT_L(0.0), .variance = FLOAT_L(0.0)},
//           .non_log = {.n = 0, .mean = FLOAT_L(0.0), .variance =
//           FLOAT_L(0.0)}}};
// }

// namespace {
//   TEST_CASE("get_historical_returns_annualized_stats") {
//     std::vector<StocksAndBondsFLOAT>
//         monthly_non_log_returns_by_run_by_mfn_simulated_host(13 * 2);
//     auto set = [&](const uint32_t run_index, FLOAT stocks, FLOAT bonds) {
//       for (uint32_t month_index = 0; month_index < 13; month_index++) {
//         const uint32_t index =
//             get_run_by_mfn_index(2, 13, run_index, month_index);
//         const FLOAT offset = static_cast<FLOAT>(month_index) / 1000.0;
//         monthly_non_log_returns_by_run_by_mfn_simulated_host[index] =
//             StocksAndBondsFLOAT{.stocks = stocks + offset,
//                                 .bonds = bonds + offset};
//       }
//     };
//     set(0, FLOAT_L(0.0), FLOAT_L(0.1));
//     set(1, FLOAT_L(0.2), FLOAT_L(0.3));

//     const auto stats = get_historical_returns_annualized_stats(
//         2, 13, monthly_non_log_returns_by_run_by_mfn_simulated_host);

//     print_returns_stats(stats, 0);
//   }

//   TEST_CASE("STAR::bench::get_historical_returns_annualized_stats") {
//     auto do_bench = [&](const uint32_t num_runs,
//                         const uint32_t num_months_to_simulate) {
//       std::vector<StocksAndBondsFLOAT>
//           monthly_non_log_returns_by_run_by_mfn_simulated_host(
//               static_cast<size_t>(num_runs * num_months_to_simulate),
//               {
//                   .stocks = FLOAT_L(0.005),
//                   .bonds = FLOAT_L(0.003),
//               });

//       thrust::device_vector<StocksAndBondsFLOAT>
//           monthly_non_log_returns_by_run_by_mfn_simulated_device =
//               monthly_non_log_returns_by_run_by_mfn_simulated_host;

//       ankerl::nanobench::Bench()
//           .timeUnit(std::chrono::milliseconds{1}, "ms")
//           .run("get_historical_returns_annualized_stats::" +
//                    std::to_string(num_runs) + "x" +
//                    std::to_string(num_months_to_simulate),
//                [&]() {
//                  get_historical_returns_annualized_stats(
//                      num_runs,
//                      num_months_to_simulate,
//                      monthly_non_log_returns_by_run_by_mfn_simulated_device);
//                });
//     };
//     for (const auto num_runs : bench_num_runs_vec) {
//       for (const auto num_years : bench_num_years_vec) {
//         do_bench(num_runs, num_years * 12);
//       }
//     }
//   }
// } // namespace