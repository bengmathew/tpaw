#include "cuda_process_tpaw_run_x_mfn_simulated_x_mfn.h"
#include "extern/doctest.h"
#include "extern/nanobench.h"
#include "mertons_formula.h"
#include "src/public_headers/numeric_types.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include "src/simulate/cuda_process_historical_returns.h"
#include "src/utils/annual_to_monthly_returns.h"
#include "src/utils/bench_utils.h"
#include "src/utils/cuda_utils.h"
#include <cstdint>
#include <sys/types.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <utility>

__device__ __host__ void
Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry::ApproxNPV::print(
    uint32_t num_tabs) const {
  const auto indent = static_cast<int32_t>(num_tabs * 4);
  printf("%*sincome_without_current_month: %.65f\n",
         indent,
         "",
         static_cast<CURRENCY>(income_without_current_month));
  printf("%*sessential_expenses_without_current_month: %.65f\n",
         indent,
         "",
         static_cast<CURRENCY>(essential_expenses_without_current_month));
  printf("%*sdiscretionary_expenses_without_current_month: %.65f\n",
         indent,
         "",
         static_cast<CURRENCY>(discretionary_expenses_without_current_month));
  printf("%*slegacy_exact: %.65f\n",
         indent,
         "",
         static_cast<CURRENCY>(legacy_exact));
}

__device__ __host__ void
Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry::print(
    uint32_t num_tabs) const {
  const auto indent = static_cast<int32_t>(num_tabs * 4);
  printf("%*snpv_approx:\n", indent, "");
  npv_approx.print(num_tabs + 1);
  printf("%*sstock_allocation_total_portfolio: %.65f\n",
         indent,
         "",
         stock_allocation_total_portfolio);
  printf("%*slegacy_stock_allocation: %.65f\n",
         indent,
         "",
         legacy_stock_allocation);
  printf("%*sspending_tilt: %.65f\n", indent, "", spending_tilt);
  printf("%*scumulative_1_plus_g_over_1_plus_r: %.65f\n",
         indent,
         "",
         cumulative_1_plus_g_over_1_plus_r);
}

// PlanParamsCuda_ByMFN but performant for kernal use.

struct __align__(32) _PlanParamsByMFN {
  CURRENCY_NPV income;
  CURRENCY_NPV essential_expense;
  CURRENCY_NPV discretionary_expense;
  FLOAT rra_including_pos_infinity;
};

__global__ void
_kernel(uint32_t num_runs,
        uint32_t num_months_to_simulate,
        uint32_t num_months,
        const _PlanParamsByMFN *params_mfn_struct,
        const FLOAT annual_empirical_log_variance_stocks,
        const FLOAT time_preference,
        const FLOAT annual_additional_spending_tilt,
        const FLOAT legacy_rra_including_pos_infinity,
        const CURRENCY legacy,
        const MonthlyAndAnnual<StocksAndBondsFLOAT>
            *expected_returns_by_run_by_mfn_simulated,
        FLOAT *stock_allocation_total_portfolio_expected_run_by_mfn,
        Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry *result) {

  const uint32_t index{blockIdx.x * blockDim.x + threadIdx.x};
  const uint32_t run_index{index / num_months_to_simulate};
  if (run_index >= num_runs)
    return;
  const uint32_t month_index{index % num_months_to_simulate};

  //   const bool debug = run_index == 0 && month_index == 0;

  const uint32_t run_by_mfn_index = get_run_by_mfn_index(
      num_runs, num_months_to_simulate, run_index, month_index);

  CURRENCY_NPV npv_income_with{CURRENCY_NPV_L(0.0)};
  CURRENCY_NPV npv_essential_expense_with{CURRENCY_NPV_L(0.0)};
  CURRENCY_NPV npv_discretionary_expense_with{CURRENCY_NPV_L(0.0)};
  FLOAT cumulative_1_plus_g_over_1_plus_r{FLOAT_L(0.0)};

  const MonthlyAndAnnual<StocksAndBondsFLOAT> expected_returns =
      expected_returns_by_run_by_mfn_simulated[run_by_mfn_index];
  const FLOAT monthly_equity_premium =
      expected_returns.monthly.stocks - expected_returns.monthly.bonds;
  const FLOAT annual_equity_premium =
      expected_returns.annual.stocks - expected_returns.annual.bonds;

  const FLOAT one_over_1p_bonds = __FLOAT_DIVIDE(
      FLOAT_L(1.0), (FLOAT_L(1.0) + expected_returns.monthly.bonds));

  const EffectiveMertonsFormulaClosure effective_mertons_formula_closure =
      get_effective_mertons_formula_closure(
          expected_returns.annual.bonds,
          annual_equity_premium,
          annual_empirical_log_variance_stocks,
          time_preference,
          annual_additional_spending_tilt);

  if (run_index == 0 && month_index == 0) {
    for (uint32_t ii = 0; ii < num_months; ii++) {
      stock_allocation_total_portfolio_expected_run_by_mfn[ii] =
          effective_mertons_formula(
              effective_mertons_formula_closure,
              params_mfn_struct[ii].rra_including_pos_infinity)
              .stock_allocation;
    }
  }

  // uint32_t cannot be used when counting down if the test condition is
  // >=0 , because 0 will wrap around instead of decrementing.
  auto num_months_int32 = static_cast<int32_t>(num_months);
  auto month_index_int32 = static_cast<int32_t>(month_index);
  for (int32_t ii = num_months_int32 - 1; ii >= month_index_int32; ii--) {

    const _PlanParamsByMFN curr_month_params = params_mfn_struct[ii];

    // Apply Merton's formula.
    const MertonsFormulaResult merton_result =
        effective_mertons_formula(effective_mertons_formula_closure,
                                  curr_month_params.rra_including_pos_infinity);

    const FLOAT one_plus_r_portfolio =
        FLOAT(1.0) + FLOAT_MA(monthly_equity_premium,
                              merton_result.stock_allocation,
                              expected_returns.monthly.bonds);
    const FLOAT one_over_one_plus_r_portfolio =
        __FLOAT_DIVIDE(FLOAT_L(1.0), one_plus_r_portfolio);

    npv_income_with =
        CURRENCY_NPV_MA(npv_income_with,
                        static_cast<CURRENCY_NPV>(one_over_1p_bonds),
                        curr_month_params.income);
    npv_essential_expense_with =
        CURRENCY_NPV_MA(npv_essential_expense_with,
                        static_cast<CURRENCY_NPV>(one_over_1p_bonds),
                        curr_month_params.essential_expense);
    npv_discretionary_expense_with = CURRENCY_NPV_MA(
        npv_discretionary_expense_with,
        static_cast<CURRENCY_NPV>(one_over_one_plus_r_portfolio),
        curr_month_params.discretionary_expense);

    FLOAT one_plus_g_over_1_plus_r = __FLOAT_DIVIDE(
        merton_result.spending_tilt + FLOAT(1.0), one_plus_r_portfolio);
    cumulative_1_plus_g_over_1_plus_r =
        FLOAT_MA(cumulative_1_plus_g_over_1_plus_r,
                 one_plus_g_over_1_plus_r,
                 FLOAT_L(1.0));
  }

  _PlanParamsByMFN curr_month_params = params_mfn_struct[month_index];

  const MertonsFormulaResult merton_result =
      effective_mertons_formula(effective_mertons_formula_closure,
                                curr_month_params.rra_including_pos_infinity);

  const FLOAT legacy_stock_allocation =
      effective_mertons_formula_stock_allocation_only(
          annual_equity_premium,
          annual_empirical_log_variance_stocks,
          legacy_rra_including_pos_infinity);

  const FLOAT r_of_legacy_portfolio = FLOAT_MA(monthly_equity_premium,
                                               legacy_stock_allocation,
                                               expected_returns.monthly.bonds);

  const uint32_t num_months_left = num_months - month_index - 1;
  const FLOAT legacy_npv_denominator =
      FLOAT_POWF(FLOAT_L(1.0) + r_of_legacy_portfolio,
                 static_cast<FLOAT>(num_months_left + 1));
  const CURRENCY legacy_npv =
      legacy / static_cast<CURRENCY>(legacy_npv_denominator);

  result[run_by_mfn_index] = {
      .npv_approx =
          {
              .income_without_current_month = static_cast<CURRENCY>(
                  npv_income_with - curr_month_params.income),
              .essential_expenses_without_current_month =
                  static_cast<CURRENCY>(npv_essential_expense_with -
                                        curr_month_params.essential_expense),
              .discretionary_expenses_without_current_month =
                  static_cast<CURRENCY>(
                      npv_discretionary_expense_with -
                      curr_month_params.discretionary_expense),
              .legacy_exact = static_cast<CURRENCY>(legacy_npv),
          },
      .stock_allocation_total_portfolio = merton_result.stock_allocation,
      .legacy_stock_allocation = legacy_stock_allocation,
      .spending_tilt = merton_result.spending_tilt,
      .cumulative_1_plus_g_over_1_plus_r = cumulative_1_plus_g_over_1_plus_r};
}

thrust::device_vector<_PlanParamsByMFN> _get_plan_params_mfn_optimized_gpu(
    const std::vector<CURRENCY> &income_by_mfn,
    const std::vector<CURRENCY> &essential_expense_by_mfn,
    const std::vector<CURRENCY> &discretionary_expense_by_mfn,
    const std::vector<FLOAT> &rra_including_pos_infinity_by_mfn) {
  const uint32_t num_months = income_by_mfn.size();
  std::vector<_PlanParamsByMFN> plan_params_mfn_optimized_host(num_months);
  for (uint32_t i{0}; i < num_months; i++) {
    plan_params_mfn_optimized_host[i] = _PlanParamsByMFN{
        .income = static_cast<CURRENCY_NPV>(income_by_mfn[i]),
        .essential_expense =
            static_cast<CURRENCY_NPV>(essential_expense_by_mfn[i]),
        .discretionary_expense =
            static_cast<CURRENCY_NPV>(discretionary_expense_by_mfn[i]),
        .rra_including_pos_infinity = rra_including_pos_infinity_by_mfn[i],
    };
  }
  const thrust::device_vector<_PlanParamsByMFN> result(
      plan_params_mfn_optimized_host);
  return result;
}

Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN
cuda_process_tpaw_run_x_mfn_simulated_x_mfn(
    const uint32_t num_runs,
    const uint32_t num_months_to_simulate,
    const uint32_t num_months,
    const std::vector<CURRENCY> &income_by_mfn,
    const std::vector<CURRENCY> &essential_expense_by_mfn,
    const std::vector<CURRENCY> &discretionary_expense_by_mfn,
    const std::vector<FLOAT> &rra_including_pos_infinity_by_mfn,
    const FLOAT annual_empirical_log_variance_stocks,
    const FLOAT time_preference,
    const FLOAT annual_additional_spending_tilt,
    const FLOAT legacy_stock_allocation,
    const CURRENCY legacy,
    const thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        &expected_returns_by_mfn_simulated_for_expected_run,
    const thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        &expected_returns_by_run_by_mfn_simulated) {

  const thrust::device_vector<_PlanParamsByMFN> plan_params_mfn_optimized_gpu =
      _get_plan_params_mfn_optimized_gpu(income_by_mfn,
                                         essential_expense_by_mfn,
                                         discretionary_expense_by_mfn,
                                         rra_including_pos_infinity_by_mfn);

  // ---- Helper to call kernel.
  auto call_kernel = [&](const uint32_t num_runs,
                         const thrust::device_vector<
                             MonthlyAndAnnual<StocksAndBondsFLOAT>>
                             &expected_returns_by_run_by_mfn_simulated) {
    thrust::device_vector<Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry>
        result(static_cast<size_t>(num_runs * num_months_to_simulate));
    thrust::device_vector<FLOAT>
        stock_allocation_total_portfolio_expected_run_by_mfn(num_months,
                                                             FLOAT_L(0.0));
    const uint32_t block_size{64};
    _kernel<<<(num_runs * num_months_to_simulate + block_size - 1) / block_size,
              block_size>>>(
        num_runs,
        num_months_to_simulate,
        num_months,
        plan_params_mfn_optimized_gpu.data().get(),
        annual_empirical_log_variance_stocks,
        time_preference,
        annual_additional_spending_tilt,
        legacy_stock_allocation,
        legacy,
        expected_returns_by_run_by_mfn_simulated.data().get(),
        stock_allocation_total_portfolio_expected_run_by_mfn.data().get(),
        result.data().get());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return std::make_pair(
        std::move(result),
        std::move(stock_allocation_total_portfolio_expected_run_by_mfn));
  };

  auto [result_for_expected_run,
        stock_allocation_total_portfolio_for_month_0_by_mfn_device] =
      call_kernel(1, expected_returns_by_mfn_simulated_for_expected_run);
  auto [result_for_normal_run, _ignore] =
      call_kernel(num_runs, expected_returns_by_run_by_mfn_simulated);

  std::vector<FLOAT> stock_allocation_total_portfolio_for_month_0_by_mfn_host =
      device_vector_to_host(
          stock_allocation_total_portfolio_for_month_0_by_mfn_device);

  const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN result{
      .tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn =
          std::move(stock_allocation_total_portfolio_for_month_0_by_mfn_device),
      .for_expected_run = std::move(result_for_expected_run),
      .for_normal_run = std::move(result_for_normal_run),
  };
  return result;
}

// *****************************************************************************
// TESTS
// *****************************************************************************

TEST_CASE("cuda_process_tpaw_run_x_mfn_simulated_x_mfn") {
  const uint32_t num_runs = 5;
  const uint32_t num_months = 30 * 12;
  const uint32_t num_months_to_simulate = num_months;
  std::vector<_PlanParamsByMFN> plan_params_mfn_host(num_months);
  for (uint32_t i{0}; i < num_months; i++) {
    plan_params_mfn_host[i] = _PlanParamsByMFN{
        .income = 0.0,
        .essential_expense = 0.0,
        .discretionary_expense = 0.0,
        .rra_including_pos_infinity = 4.0,
    };
  }
  thrust::device_vector<_PlanParamsByMFN> plan_params_mfn_gpu(
      plan_params_mfn_host);

  std::vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
      expected_returns_by_run_by_mfn_simulated_host(
          static_cast<size_t>(num_runs * num_months_to_simulate),
          {.monthly =
               StocksAndBondsFLOAT{
                   .stocks = annual_to_monthly_return(0.05),
                   .bonds = annual_to_monthly_return(0.03),
               },
           .annual = {
               .stocks = 0.05,
               .bonds = 0.03,
           }});
  thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
      expected_returns_by_run_by_mfn_simulated_gpu(
          expected_returns_by_run_by_mfn_simulated_host);

  thrust::device_vector<FLOAT>
      stock_allocation_total_portfolio_expected_run_by_mfn_device(num_months,
                                                                  FLOAT_L(0.0));
  thrust::device_vector<Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry>
      result_device(static_cast<size_t>(num_runs * num_months_to_simulate));

  const uint32_t block_size{64};
  _kernel<<<(num_runs * num_months_to_simulate + block_size - 1) / block_size,
            block_size>>>(
      num_runs,
      num_months_to_simulate,
      num_months,
      plan_params_mfn_gpu.data().get(),
      0.01, // annual_empirical_log_variance_stocks
      0.01, // time_preference
      0.0,  // annual_additional_spending_tilt
      4.0,  // legacy_rra_including_pos_infinity
      0.0,  // legacy
      expected_returns_by_run_by_mfn_simulated_gpu.data().get(),
      stock_allocation_total_portfolio_expected_run_by_mfn_device.data().get(),
      result_device.data().get());
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  std::vector<Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry> result_host =
      device_vector_to_host(result_device);

  // Print out test_out_host with each run on a separate line
  //   for (uint32_t run = 0; run < num_runs; ++run) {
  //     std::cout << "Run " << run << ": ";
  //     for (uint32_t month = 0; month < num_months_to_simulate; ++month) {
  //       uint32_t index =
  //           get_run_by_mfn_index(num_runs, num_months_to_simulate, run,
  //           month);
  //       std::cout << result_host[index].stock_allocation << " ";
  //     }
  //     std::cout << std::endl;
  //   }
};

// *****************************************************************************
// BENCHES
// *****************************************************************************

TEST_CASE("bench::cuda_process_tpaw_run_x_mfn_simulated_x_mfn") {

  const auto do_bench = [](const char *name,
                           const uint32_t num_runs,
                           const uint32_t num_months,
                           const bool direct_kernel_call) {
    const uint32_t num_months_to_simulate = num_months;

    std::vector<CURRENCY> income_by_mfn(num_months, 1000.0);
    std::vector<CURRENCY> essential_expense_by_mfn(num_months, 400.0);
    std::vector<CURRENCY> discretionary_expense_by_mfn(num_months, 9000.0);
    std::vector<FLOAT> rra_including_pos_infinity_by_mfn(num_months,
                                                         FLOAT_L(4.0));

    thrust::device_vector<_PlanParamsByMFN> plan_params_mfn_optimized_gpu =
        _get_plan_params_mfn_optimized_gpu(income_by_mfn,
                                           essential_expense_by_mfn,
                                           discretionary_expense_by_mfn,
                                           rra_including_pos_infinity_by_mfn);

    const StocksAndBondsFLOAT current_annual_expected_returns = {
        .stocks = (0.05),
        .bonds = (0.02),
    };
    const MonthlyAndAnnual<StocksAndBondsFLOAT> current_expected_returns = {
        .monthly =
            {
                .stocks = annual_to_monthly_return(
                    current_annual_expected_returns.stocks),
                .bonds = annual_to_monthly_return(
                    current_annual_expected_returns.bonds),
            },
        .annual = current_annual_expected_returns,
    };
    thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        expected_returns_by_mfn_simulated_for_expected_run_device(
            static_cast<size_t>(num_months_to_simulate),
            current_expected_returns);
    thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        expected_returns_by_run_by_mfn_simulated_device(
            static_cast<size_t>(num_runs * num_months_to_simulate),
            current_expected_returns);

    ankerl::nanobench::Bench()
        .timeUnit(std::chrono::milliseconds{1}, "ms")
        .run(name, [&]() {
          thrust::device_vector<FLOAT>
              stock_allocation_total_portfolio_expected_run_by_mfn_device(
                  num_months, FLOAT_L(0.0));
          thrust::device_vector<
              Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry>
              result(static_cast<size_t>(num_runs * num_months_to_simulate));

          if (direct_kernel_call) {
            const uint32_t block_size{64};
            _kernel<<<(num_runs * num_months_to_simulate + block_size - 1) /
                          block_size,
                      block_size>>>(
                num_runs,
                num_months_to_simulate,
                num_months,
                plan_params_mfn_optimized_gpu.data().get(),
                0.01,   // annual_empirical_log_variance_stocks
                0.0,    // time_preference
                0.0,    // annual_additional_spending_tilt
                0.5,    // legacy_stock_allocation
                1000.0, // legacy
                expected_returns_by_run_by_mfn_simulated_device.data().get(),
                stock_allocation_total_portfolio_expected_run_by_mfn_device
                    .data()
                    .get(),
                result.data().get());
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
          } else {
            cuda_process_tpaw_run_x_mfn_simulated_x_mfn(
                num_runs,
                num_months_to_simulate,
                num_months,
                income_by_mfn,
                essential_expense_by_mfn,
                discretionary_expense_by_mfn,
                rra_including_pos_infinity_by_mfn,
                0.01,   // empirical_log_variance_stocks
                0.0,    // time_preference
                1.0,    // additional_spending_tilt_plus_one
                0.5,    // legacy_stock_allocation
                1000.0, // legacy
                expected_returns_by_mfn_simulated_for_expected_run_device,
                expected_returns_by_run_by_mfn_simulated_device);
          }
        });
  };

  for (const auto &direct_kernel_call : std::vector<bool>{true, false}) {
    for (const auto &num_runs : bench_num_runs_vec) {
      for (const auto &num_years : bench_num_years_vec) {
        do_bench((std::string("cuda_process_tpaw_run_x_mfn_simulated_x_mfn::") +
                  std::to_string(num_runs) + "x" + std::to_string(num_years) +
                  (direct_kernel_call ? "xdirect" : "xindirect"))
                     .c_str(),
                 num_runs,
                 num_years * 12,
                 direct_kernel_call);
      }
    }
  }
}

TEST_CASE("bench::cuda_process_tpaw_run_x_mfn_simulated_x_mfn::optimize_plan_"
          "params_mfn") {
  const auto do_bench = [](const char *name,
                           const uint32_t num_runs,
                           const uint32_t num_months) {
    const uint32_t num_months_to_simulate = num_months;
    std::vector<CURRENCY> income_by_mfn(num_months, 1000.0);
    std::vector<CURRENCY> essential_expense_by_mfn(num_months, 400.0);
    std::vector<CURRENCY> discretionary_expense_by_mfn(num_months, 9000.0);
    std::vector<FLOAT> rra_including_pos_infinity_by_mfn(num_months,
                                                         FLOAT_L(4.0));

    ankerl::nanobench::Bench()
        .timeUnit(std::chrono::milliseconds{1}, "ms")
        .run(name, [&]() {
          thrust::device_vector<
              Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry>
              result(static_cast<size_t>(num_runs * num_months_to_simulate));
          _get_plan_params_mfn_optimized_gpu(income_by_mfn,
                                             essential_expense_by_mfn,
                                             discretionary_expense_by_mfn,
                                             rra_including_pos_infinity_by_mfn);
        });
  };

  for (const auto &num_runs : bench_num_runs_vec) {
    for (const auto &num_years : bench_num_years_vec) {
      do_bench((std::string("cuda_process_tpaw_run_x_mfn_simulated_x_mfn::"
                            "optimize_plan_params_mfn::") +
                std::to_string(num_runs) + "x" + std::to_string(num_years))
                   .c_str(),
               num_runs,
               num_years * 12);
    }
  }
}