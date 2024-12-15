#include "extern/doctest.h"
#include "extern/nanobench.h"
#include "simulate.h"
#include "src/public_headers/numeric_types.h"
#include "src/public_headers/plan_params_cuda.h"
#include "src/simulate/cuda_process_historical_returns.h"
#include "src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/cuda_process_spaw_run_x_mfn_simulated_x_mfn.h"
#include "src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/cuda_process_swr_run_x_mfn_simulated_x_mfn.h"
#include "src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/cuda_process_tpaw_run_x_mfn_simulated_x_mfn.h"
#include "src/simulate/cuda_process_sampling.h"
#include "src/simulate/get_result_cuda_not_array.h"
#include "src/simulate/pick_percentiles.h"
#include "src/simulate/run/run_spaw.h"
#include "src/simulate/run/run_swr.h"
#include "src/simulate/run/run_tpaw.h"
#include "src/simulate/sort_run_result_padded.h"
#include "src/utils/annual_to_monthly_returns.h"
#include "src/utils/bench_utils.h"
#include "src/utils/get_result_cuda_for_testing.h"
#include "thrust/device_vector.h"
#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

ResultCudaNotArrays
simulate::simulate(const uint32_t num_months_to_simulate,
                   const CURRENCY current_portfolio_balance,
                   const PlanParamsCuda &plan_params,
                   const PlanParamsCuda_Vectors &plan_params_vectors,
                   ResultCudaArrays &out) {

  const thrust::device_vector<CURRENCY> income_by_mfn_gpu =
      plan_params_vectors.income_combined_by_mfn;
  const thrust::device_vector<CURRENCY> essential_expense_by_mfn_gpu =
      plan_params_vectors.essential_expenses_by_mfn;
  const thrust::device_vector<CURRENCY> discretionary_expense_by_mfn_gpu =
      plan_params_vectors.discretionary_expenses_by_mfn;
  const thrust::device_vector<FLOAT>
      stock_allocation_savings_portfolio_by_mfn_gpu =
          plan_params_vectors
              .spaw_and_swr_stock_allocation_savings_portfolio_by_mfn;

  const thrust::device_vector<HistoricalReturnsCuda>
      historical_returns_series_gpu =
          plan_params_vectors.historical_returns_series;

  const SamplingCudaProcessed sampling_cuda_processed =
      cuda_process_sampling(plan_params.advanced.sampling,
                            historical_returns_series_gpu.size(),
                            plan_params.ages.simulation_months.num_months,
                            num_months_to_simulate);

  const HistoricalReturnsCudaProcessed historical_returns_cuda_processed =
      cuda_process_historical_returns(
          sampling_cuda_processed,
          num_months_to_simulate,
          historical_returns_series_gpu,
          plan_params.advanced.return_stats_for_planning
              .expected_returns_at_month_0);

  const auto simulate_tpaw = [&]() {
    const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN
        cuda_processed_run_x_mfn_simulated_x_mfn =
            cuda_process_tpaw_run_x_mfn_simulated_x_mfn(
                sampling_cuda_processed.num_runs,
                num_months_to_simulate,
                plan_params.ages.simulation_months.num_months,
                plan_params_vectors.income_combined_by_mfn,
                plan_params_vectors.essential_expenses_by_mfn,
                plan_params_vectors.discretionary_expenses_by_mfn,
                plan_params_vectors.tpaw_rra_including_pos_infinity_by_mfn,
                plan_params.advanced.return_stats_for_planning
                    .annual_empirical_log_variance_stocks,
                plan_params.risk.tpaw.time_preference,
                plan_params.risk.tpaw.annual_additional_spending_tilt,
                plan_params.risk.tpaw.legacy_rra_including_pos_infinity,
                plan_params.adjustments_to_spending.tpaw_and_spaw.legacy,
                historical_returns_cuda_processed
                    .expected_returns_by_mfn_simulated_for_expected_run,
                historical_returns_cuda_processed
                    .expected_returns_by_run_by_mfn_simulated);

    thrust::copy(
        cuda_processed_run_x_mfn_simulated_x_mfn
            .tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn.begin(),
        cuda_processed_run_x_mfn_simulated_x_mfn
            .tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn.end(),
        out.tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn);

    RunResultPadded run_result_padded = tpaw::run(
        sampling_cuda_processed.num_runs,
        num_months_to_simulate,
        current_portfolio_balance,
        plan_params.ages.simulation_months.withdrawal_start_month,
        plan_params.adjustments_to_spending.tpaw_and_spaw.spending_ceiling,
        plan_params.adjustments_to_spending.tpaw_and_spaw.spending_floor,
        plan_params.advanced.return_stats_for_planning
            .expected_returns_at_month_0,
        income_by_mfn_gpu,
        essential_expense_by_mfn_gpu,
        discretionary_expense_by_mfn_gpu,
        historical_returns_cuda_processed
            .historical_returns_by_run_by_mfn_simulated,
        cuda_processed_run_x_mfn_simulated_x_mfn);

    Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry first_entry =
        cuda_processed_run_x_mfn_simulated_x_mfn.for_expected_run[0];

    return std::make_pair(std::move(run_result_padded),
                          first_entry.npv_approx.legacy_exact);
  };

  const auto simulate_spaw = [&]() {
    const Cuda_Processed_SPAW_Run_x_MFNSimulated_x_MFN
        cuda_processed_run_x_mfn_simulated_x_mfn =
            cuda_process_spaw_run_x_mfn_simulated_x_mfn(
                sampling_cuda_processed.num_runs,
                num_months_to_simulate,
                plan_params.ages.simulation_months.num_months,
                plan_params_vectors.income_combined_by_mfn,
                plan_params_vectors.essential_expenses_by_mfn,
                plan_params_vectors.discretionary_expenses_by_mfn,
                plan_params_vectors
                    .spaw_and_swr_stock_allocation_savings_portfolio_by_mfn,
                plan_params_vectors.spaw_spending_tilt_by_mfn,
                plan_params.adjustments_to_spending.tpaw_and_spaw.legacy,
                historical_returns_cuda_processed
                    .expected_returns_by_mfn_simulated_for_expected_run,
                historical_returns_cuda_processed
                    .expected_returns_by_run_by_mfn_simulated);

    RunResultPadded run_result_padded = spaw::run(
        sampling_cuda_processed.num_runs,
        num_months_to_simulate,
        current_portfolio_balance,
        plan_params.ages.simulation_months.withdrawal_start_month,
        plan_params.adjustments_to_spending.tpaw_and_spaw.spending_ceiling,
        plan_params.adjustments_to_spending.tpaw_and_spaw.spending_floor,
        plan_params.advanced.return_stats_for_planning
            .expected_returns_at_month_0,
        income_by_mfn_gpu,
        essential_expense_by_mfn_gpu,
        discretionary_expense_by_mfn_gpu,
        stock_allocation_savings_portfolio_by_mfn_gpu,
        historical_returns_cuda_processed
            .historical_returns_by_run_by_mfn_simulated,
        cuda_processed_run_x_mfn_simulated_x_mfn);

    return std::make_pair(std::move(run_result_padded), 0.0);
  };

  const auto simulate_swr = [&]() {
    const Cuda_Processed_SWR_Run_x_MFNSimulated_x_MFN
        cuda_processed_run_x_mfn_simulated_x_mfn =
            cuda_process_swr_run_x_mfn_simulated_x_mfn(
                sampling_cuda_processed.num_runs,
                num_months_to_simulate,
                plan_params.ages.simulation_months.num_months,
                plan_params_vectors.income_combined_by_mfn,
                historical_returns_cuda_processed
                    .expected_returns_by_run_by_mfn_simulated);

    RunResultPadded run_result_padded =
        swr::run(sampling_cuda_processed.num_runs,
                 num_months_to_simulate,
                 current_portfolio_balance,
                 plan_params.ages.simulation_months.withdrawal_start_month,
                 plan_params.risk.swr.withdrawal_type,
                 plan_params.risk.swr.withdrawal_as_percent_or_amount,
                 income_by_mfn_gpu,
                 essential_expense_by_mfn_gpu,
                 discretionary_expense_by_mfn_gpu,
                 stock_allocation_savings_portfolio_by_mfn_gpu,
                 historical_returns_cuda_processed
                     .historical_returns_by_run_by_mfn_simulated,
                 cuda_processed_run_x_mfn_simulated_x_mfn);

    return std::make_pair(std::move(run_result_padded), 0.0);
  };

  auto [run_result_padded, tpaw_net_present_value_exact_month_0_legacy] =
      plan_params.advanced.strategy == PlanParamsCuda::Advanced::Strategy_TPAW
          ? simulate_tpaw()
      : plan_params.advanced.strategy == PlanParamsCuda::Advanced::Strategy_SPAW
          ? simulate_spaw()
      : plan_params.advanced.strategy == PlanParamsCuda::Advanced::Strategy_SWR
          ? simulate_swr()
          : throw std::runtime_error(
                "simulate::simulate::unsupported_strategy");

  // This should only pull values that are identical for each run, so it does
  // not technically matter that this is done before sort, but we call it
  // before sort for semantic clarity.
  const ResultCudaNotArrays result_cuda_not_array =
      get_result_cuda_not_array(sampling_cuda_processed.num_runs,
                                run_result_padded,
                                tpaw_net_present_value_exact_month_0_legacy);

  sort_run_result_padded(run_result_padded,
                         sampling_cuda_processed.num_runs,
                         num_months_to_simulate);

  pick_percentiles(run_result_padded,
                   sampling_cuda_processed.num_runs,
                   plan_params_vectors.percentiles,
                   num_months_to_simulate,
                   out);

  return result_cuda_not_array;
}

const PlanParamsCuda
get_test_plan_params(const uint32_t num_runs,
                     const uint32_t num_months,
                     const PlanParamsCuda::Advanced::Strategy strategy) {

  return PlanParamsCuda{
      .ages = PlanParamsCuda::Ages{.simulation_months =
                                       PlanParamsCuda::Ages::SimulationMonths{
                                           .num_months = num_months,
                                           .withdrawal_start_month =
                                               num_months / 2}},
      .adjustments_to_spending =
          PlanParamsCuda::AdjustmentsToSpending{
              .tpaw_and_spaw =
                  PlanParamsCuda::AdjustmentsToSpending::TPAWAndSPAW{
                      .spending_ceiling = OptCURRENCY{0, 0.0},
                      .spending_floor = OptCURRENCY{0, 0.0},
                      .legacy = 1000.0}},

      .risk =
          PlanParamsCuda::Risk{
              .tpaw =
                  PlanParamsCuda::Risk::TPAW{
                      .time_preference = 0.0,
                      .annual_additional_spending_tilt = 0.0,
                      .legacy_rra_including_pos_infinity = 5.0},
              .swr =
                  PlanParamsCuda::Risk::SWR{
                      .withdrawal_type =
                          PlanParamsCuda::Risk::SWR::WithdrawalType::Percent,
                      .withdrawal_as_percent_or_amount = 0.004}},

      .advanced =
          PlanParamsCuda::Advanced{
              .return_stats_for_planning =
                  PlanParamsCuda::Advanced::ReturnStatsForPlanning{
                      .expected_returns_at_month_0 =
                          StocksAndBondsFLOAT{
                              .stocks = static_cast<float>(
                                  annual_to_monthly_return(0.05)),
                              .bonds = static_cast<float>(
                                  annual_to_monthly_return(0.02))},
                      .annual_empirical_log_variance_stocks = 0.1},
              .sampling =
                  PlanParamsCuda::Advanced::Sampling{
                      .type = PlanParamsCuda::Advanced::Sampling::Type::
                          MonteCarloSampling,
                      .monte_carlo_or_historical =
                          {PlanParamsCuda::Advanced::Sampling::MonteCarlo{
                              .seed = 1234,
                              .num_runs = num_runs,
                              .block_size = 5 * 12,
                              .stagger_run_starts = true}}},
              .strategy = strategy}

  };
}

std::tuple<PlanParamsCuda, PlanParamsCuda_Vectors>
get_test_params(const uint32_t num_runs,
                const uint32_t num_months,
                const uint32_t history_len,
                const PlanParamsCuda::Advanced::Strategy strategy) {

  std::vector<HistoricalReturnsCuda> historical_returns_series(
      history_len,
      HistoricalReturnsCuda{
          .stocks =
              HistoricalReturnsCuda::Part{
                  .returns = annual_to_monthly_return(0.05),
                  .expected_return_change = 0.0,
              },
          .bonds =
              HistoricalReturnsCuda::Part{
                  .returns = annual_to_monthly_return(0.02),
                  .expected_return_change = 0.0,
              },
      });

  const PlanParamsCuda plan_params =
      get_test_plan_params(num_runs, num_months, strategy);

  PlanParamsCuda_Vectors plan_params_vectors{
      .income_combined_by_mfn = std::vector<CURRENCY>(num_months, 1000.0),
      .essential_expenses_by_mfn = std::vector<CURRENCY>(num_months, 30.0),
      .discretionary_expenses_by_mfn = std::vector<CURRENCY>(num_months, 40.0),
      .tpaw_rra_including_pos_infinity_by_mfn =
          std::vector<FLOAT>(num_months, 2.0),
      .spaw_spending_tilt_by_mfn = std::vector<FLOAT>(num_months, 0.0),
      .spaw_and_swr_stock_allocation_savings_portfolio_by_mfn =
          std::vector<FLOAT>(num_months, 0.5),
      .percentiles = {5, 50, 95},
      .historical_returns_series = historical_returns_series,
  };

  return std::make_tuple(plan_params, std::move(plan_params_vectors));
}

TEST_CASE("STAR::simulate") {

  const uint32_t num_runs = 1000;
  const uint32_t num_months = 120 * 12;

  const std::vector<PlanParamsCuda::Advanced::Strategy> strategies = {
      PlanParamsCuda::Advanced::Strategy_TPAW,
      PlanParamsCuda::Advanced::Strategy_SPAW,
      PlanParamsCuda::Advanced::Strategy_SWR};

  for (const auto strategy : strategies) {
    auto [plan_params, plan_params_vectors] =
        get_test_params(num_runs, num_months, 2000, strategy);
    auto [out, out_raii] = get_result_cuda_for_testing(
        plan_params_vectors.percentiles.size(), num_months);

    simulate::simulate(
        num_months, 100000.0, plan_params, plan_params_vectors, out);
  }
}

TEST_CASE("STAR::bench::simulate") {
  const auto do_bench = [](std::string name,
                           uint32_t num_runs,
                           uint32_t num_months,
                           const PlanParamsCuda::Advanced::Strategy strategy) {
    auto [plan_params, plan_params_vectors] =
        get_test_params(num_runs, num_months, 2000, strategy);
    auto [out, raii] = get_result_cuda_for_testing(
        plan_params_vectors.percentiles.size(), num_months);

    const std::string strategy_str =
        strategy == PlanParamsCuda::Advanced::Strategy_TPAW   ? "TPAW"
        : strategy == PlanParamsCuda::Advanced::Strategy_SPAW ? "SPAW"
        : strategy == PlanParamsCuda::Advanced::Strategy_SWR
            ? "SWR"
            : throw std::runtime_error("unsupported_strategy");

    ankerl::nanobench::Bench()
        .timeUnit(std::chrono::milliseconds{1}, "ms")
        .run((name + "::" + strategy_str + "::" + std::to_string(num_runs) +
              " x " + std::to_string(num_months))
                 .c_str(),
             [&]() {
               simulate::simulate(
                   num_months, 100000.0, plan_params, plan_params_vectors, out);
             });
  };

  const std::vector<PlanParamsCuda::Advanced::Strategy> strategies = {
      PlanParamsCuda::Advanced::Strategy_TPAW,
      PlanParamsCuda::Advanced::Strategy_SPAW,
      PlanParamsCuda::Advanced::Strategy_SWR};
  for (const auto strategy : strategies) {
    for (auto num_runs : bench_num_runs_vec) {
      for (auto num_years : bench_num_years_vec) {

        do_bench("simulate", num_runs, num_years * 12, strategy);
      }
    }
  }
  for (const auto strategy : strategies) {
    for (auto num_runs : bench_num_runs_vec) {
      do_bench(("simulate::stock_allocation_only " + std::to_string(num_runs))
                   .c_str(),
               num_runs,
               1,
               strategy);
    }
  }
}
