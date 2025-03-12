#include "extern/doctest.h"
#include "extern/nanobench.h"
#include "run_tpaw.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include "src/simulate/cuda_process_historical_returns.h"
#include "src/simulate/run/run_common.h"
#include "src/simulate/run/run_result_padded.h"
#include "src/utils/annual_to_monthly_returns.h"
#include "src/utils/bench_utils.h"
#include "src/utils/monthly_to_annual_returns.h"
#include "src/utils/print_public_types.h"
#include "src/utils/run_mfn_indexing.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace tpaw {
  struct _ExpectedRunData {
    CURRENCY wealth_starting;
    FLOAT elasticity_of_extra_withdrawal_goals_wrt_wealth;
    FLOAT elasticity_of_legacy_goals_wrt_wealth;

    __device__ __host__ void print(uint32_t num_tabs) const;
  };

  __device__ __host__ void _ExpectedRunData::print(uint32_t num_tabs) const {
    const auto i = static_cast<int32_t>(num_tabs * 4);
    printf("%*swealth_starting: %.65f\n", i, "", wealth_starting);
    printf("%*selasticity_of_extra_withdrawal_goals_wrt_wealth: %.65f\n",
           i,
           "",
           elasticity_of_extra_withdrawal_goals_wrt_wealth);
    printf("%*selast  icity_of_legacy_goals_wrt_wealth: %.65f\n",
           i,
           "",
           elasticity_of_legacy_goals_wrt_wealth);
  }

  // *****************************************************************************
  // _get_precomputation_at_start
  // *****************************************************************************
  struct _NPVSpendingScaledAndConstrainedToWealth {
    CURRENCY discretionary;
    CURRENCY legacy;
    CURRENCY general;
    __host__ __device__ void print(uint32_t num_tabs) const {
      const auto i = static_cast<int32_t>(num_tabs * 4);
      printf("%*s"
             "discretionary: %.65f\n",
             i,
             "",
             discretionary);
      printf("%*s"
             "legacy: %.65f\n",
             i,
             "",
             legacy);
      printf("%*s"
             "general: %.65f\n",
             i,
             "",
             general);
    }
  };

  struct _ExpensesScale {
    FLOAT discretionary;
    FLOAT legacy;
    __host__ __device__ void print(uint32_t num_tabs) const {
      const auto i = static_cast<int32_t>(num_tabs * 4);
      printf("%*s"
             "discretionary: %.65f\n",
             i,
             "",
             discretionary);
      printf("%*s"
             "legacy: %.65f\n",
             i,
             "",
             legacy);
    }
  };

  struct _PrecomputationAtStart {
    CURRENCY wealth;
    _ExpensesScale expenses_scale;
    _NPVSpendingScaledAndConstrainedToWealth
        npv_approx_spending_scaled_and_constrained_to_wealth;

    __host__ __device__ void print(uint32_t num_tabs) const {
      const auto i = static_cast<int32_t>(num_tabs * 4);
      printf("%*s"
             "wealth: %.65f\n",
             i,
             "",
             wealth);
      printf("%*s"
             "expenses_scale: \n",
             i,
             "");
      expenses_scale.print(num_tabs + 1);
      printf("%*s"
             "npv_approx_spending_scaled_and_constrained_to_wealth: \n",
             i,
             "");
      npv_approx_spending_scaled_and_constrained_to_wealth.print(num_tabs + 1);
    }
  };

  namespace _get_precomputation_at_start {
    template <bool is_expected_run>
    _PrecomputationAtStart __device__ __forceinline__
    fn(const CURRENCY balance_starting,
       const CURRENCY npv_approx_income_without_current_month,
       const CURRENCY npv_approx_essential_expenses_without_current_month,
       const CURRENCY npv_approx_discretionary_expenses_without_current_month,
       const CURRENCY npv_approx_legacy,
       const CURRENCY current_month_income,
       const CURRENCY current_month_essential_expense,
       const CURRENCY current_month_discretionary_expense,
       const _ExpectedRunData *expected_run_data_if_normal_run) {

      const CURRENCY wealth = balance_starting +
                              npv_approx_income_without_current_month +
                              current_month_income;
      _ExpensesScale expenses_scale =
        is_expected_run
            ? _ExpensesScale{
                  .discretionary = FLOAT_L(1.0),
                  .legacy = FLOAT_L(1.0),
              }
            : ({
      const FLOAT percentage_increase_in_wealth_over_scheduled =
          expected_run_data_if_normal_run->wealth_starting == FLOAT_L(0.0)
              ? FLOAT_L(0.0)
              : static_cast<FLOAT>(
                    wealth / expected_run_data_if_normal_run->wealth_starting) -
                    FLOAT_L(1.0);
      _ExpensesScale {
          .discretionary = FLOAT_MAX(
              FLOAT_MA(percentage_increase_in_wealth_over_scheduled,
                       expected_run_data_if_normal_run
                           ->elasticity_of_extra_withdrawal_goals_wrt_wealth,
                       FLOAT_L(1.0)),
              FLOAT_L(0.0)),
          .legacy =
              FLOAT_MAX(FLOAT_MA(percentage_increase_in_wealth_over_scheduled,
                                 expected_run_data_if_normal_run
                                     ->elasticity_of_legacy_goals_wrt_wealth,
                                 FLOAT_L(1.0)),
                        FLOAT_L(0.0)),
      };
    });

      AccountForWithdrawal account(wealth);
      account.withdraw(npv_approx_essential_expenses_without_current_month +
                       current_month_essential_expense);

      const _NPVSpendingScaledAndConstrainedToWealth
          npv_spending_scaled_and_constrained_to_wealth{
              .discretionary = account.withdraw(
                  (npv_approx_discretionary_expenses_without_current_month +
                   current_month_discretionary_expense) *
                  expenses_scale.discretionary),
              .legacy =
                  account.withdraw(npv_approx_legacy * expenses_scale.legacy),
              .general = account.balance,
          };

      return _PrecomputationAtStart{
          .wealth = wealth,
          .expenses_scale = expenses_scale,
          .npv_approx_spending_scaled_and_constrained_to_wealth =
              npv_spending_scaled_and_constrained_to_wealth,
      };
    }
    template <bool is_expected_run>
    __global__ void _test_kernel(
        const CURRENCY balance_starting,
        const CURRENCY npv_approx_income_with_current_month,
        const CURRENCY npv_approx_essential_expenses_with_current_month,
        const CURRENCY npv_approx_discretionary_expenses_with_current_month,
        const CURRENCY npv_approx_legacy,
        const CURRENCY current_month_income,
        const CURRENCY current_month_essential_expense,
        const CURRENCY current_month_discretionary_expense,
        const _ExpectedRunData expected_run_data_if_normal_run,
        _PrecomputationAtStart *result) {

      *result = fn<is_expected_run>(
          balance_starting,
          npv_approx_income_with_current_month,
          npv_approx_essential_expenses_with_current_month,
          npv_approx_discretionary_expenses_with_current_month,
          npv_approx_legacy,
          current_month_income,
          current_month_essential_expense,
          current_month_discretionary_expense,
          is_expected_run ? nullptr : &expected_run_data_if_normal_run);
    }

    template <bool is_expected_run>
    _PrecomputationAtStart _test_fn(
        const CURRENCY balance_starting,
        const CURRENCY npv_approx_income_with_current_month,
        const CURRENCY npv_approx_essential_expenses_with_current_month,
        const CURRENCY npv_approx_discretionary_expenses_with_current_month,
        const CURRENCY npv_approx_legacy,
        const CURRENCY current_month_income,
        const CURRENCY current_month_essential_expense,
        const CURRENCY current_month_discretionary_expense,
        const _ExpectedRunData &expected_run_data_if_normal_run) {
      thrust::device_vector<_PrecomputationAtStart> result_vec(1);
      _test_kernel<is_expected_run>
          <<<1, 1>>>(balance_starting,
                     npv_approx_income_with_current_month,
                     npv_approx_essential_expenses_with_current_month,
                     npv_approx_discretionary_expenses_with_current_month,
                     npv_approx_legacy,
                     current_month_income,
                     current_month_essential_expense,
                     current_month_discretionary_expense,
                     expected_run_data_if_normal_run,
                     result_vec.data().get());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      return result_vec[0];
    };

    TEST_CASE("tpaw::_get_precomputation_at_start") {
      SUBCASE("wealth") {
        const _PrecomputationAtStart result = _test_fn<true>(
            100000.0,
            40000.0,
            0.0,
            0.0,
            0.0,
            10000.0,
            0.0,
            0.0,
            _ExpectedRunData{
                .wealth_starting = 0.0,
                .elasticity_of_extra_withdrawal_goals_wrt_wealth = FLOAT_L(0.0),
                .elasticity_of_legacy_goals_wrt_wealth = FLOAT_L(0.0),
            });

        CHECK(result.wealth == 100000.0 + 50000.0);
      }

      SUBCASE("expenses_scale") {
        SUBCASE("expected_run") {
          const _PrecomputationAtStart result = _test_fn<true>(
              100000.0,
              40000.0,
              0.0,
              0.0,
              0.0,
              10000.0,
              0.0,
              0.0,
              _ExpectedRunData{
                  .wealth_starting = 90000.0 + 50000.0,
                  .elasticity_of_extra_withdrawal_goals_wrt_wealth =
                      FLOAT_L(0.2),
                  .elasticity_of_legacy_goals_wrt_wealth = FLOAT_L(0.5),
              });

          CHECK(result.expenses_scale.discretionary ==
                doctest::Approx(FLOAT_L(1.0)));
          CHECK(result.expenses_scale.legacy == doctest::Approx(FLOAT_L(1.0)));
        }

        SUBCASE("more wealth") {
          const _PrecomputationAtStart result = _test_fn<false>(
              100000.0,
              40000.0,
              0.0,
              0.0,
              0.0,
              10000.0,
              0.0,
              0.0,
              _ExpectedRunData{
                  .wealth_starting = 90000.0 + 50000.0,
                  .elasticity_of_extra_withdrawal_goals_wrt_wealth =
                      FLOAT_L(0.2),
                  .elasticity_of_legacy_goals_wrt_wealth = FLOAT_L(0.5),
              });
          const FLOAT p_increase_in_wealth = (10000.0) / (90000.0 + 50000.0);
          CHECK(result.expenses_scale.discretionary ==
                doctest::Approx(FLOAT_L(p_increase_in_wealth * 0.2 + 1.0)));
          CHECK(result.expenses_scale.legacy ==
                doctest::Approx(FLOAT_L(p_increase_in_wealth * 0.5 + 1.0)));
        }

        SUBCASE("less wealth") {
          const _PrecomputationAtStart result = _test_fn<false>(
              80000.0,
              40000.0,
              0.0,
              0.0,
              0.0,
              10000.0,
              0.0,
              0.0,
              _ExpectedRunData{
                  .wealth_starting = 90000.0 + 50000.0,
                  .elasticity_of_extra_withdrawal_goals_wrt_wealth =
                      FLOAT_L(0.2),
                  .elasticity_of_legacy_goals_wrt_wealth = FLOAT_L(0.5),
              });
          const FLOAT p_increase_in_wealth = (-10000.0) / (90000.0 + 50000.0);
          CHECK(result.expenses_scale.discretionary ==
                doctest::Approx(FLOAT_L(p_increase_in_wealth * 0.2 + 1.0)));
          CHECK(result.expenses_scale.legacy ==
                doctest::Approx(FLOAT_L(p_increase_in_wealth * 0.5 + 1.0)));
        }
      }
    }
  } // namespace _get_precomputation_at_start

  // *****************************************************************************
  // _get_expected_run_data
  // *****************************************************************************
  namespace _get_expected_run_data {
    _ExpectedRunData __device__ __forceinline__
    fn(const FLOAT stock_allocation,
       const FLOAT legacy_stock_allocation,
       const CURRENCY wealth_at_start,
       const _NPVSpendingScaledAndConstrainedToWealth
           npv_approx_at_start_spending_scaled_and_constrained_to_wealth) {

      // Effectively the percentage of wealth that is in stocks.
      const FLOAT elasticity_of_wealth_wrt_stocks =
          wealth_at_start == 0.0
              ? __FLOAT_DIVIDE(legacy_stock_allocation + stock_allocation +
                                   stock_allocation,
                               FLOAT_L(3.0))
              : ({
                  const CURRENCY legacy_in_stocks =
                      (npv_approx_at_start_spending_scaled_and_constrained_to_wealth
                           .legacy *
                       static_cast<CURRENCY>(legacy_stock_allocation));
                  const auto discretionary_and_legacy_in_stocks = fma(
                      npv_approx_at_start_spending_scaled_and_constrained_to_wealth
                          .discretionary,
                      static_cast<CURRENCY>(stock_allocation),
                      legacy_in_stocks);
                  const auto all_in_stocks = fma(
                      npv_approx_at_start_spending_scaled_and_constrained_to_wealth
                          .general,
                      static_cast<CURRENCY>(stock_allocation),
                      discretionary_and_legacy_in_stocks);
                  static_cast<FLOAT>(all_in_stocks / wealth_at_start);
                });

      const FLOAT elasticity_of_extra_withdrawal_goals_wrt_stocks =
          stock_allocation;
      const FLOAT elasticity_of_extra_withdrawal_goals_wrt_wealth =
          elasticity_of_wealth_wrt_stocks == FLOAT_L(0.0)
              ? FLOAT_L(0.0)
              : __FLOAT_DIVIDE(elasticity_of_extra_withdrawal_goals_wrt_stocks,
                               elasticity_of_wealth_wrt_stocks);

      const FLOAT elasticity_of_legacy_goals_wrt_stocks =
          legacy_stock_allocation;
      const FLOAT elasticity_of_legacy_goals_wrt_wealth =
          elasticity_of_wealth_wrt_stocks == FLOAT_L(0.0)
              ? FLOAT_L(0.0)
              : __FLOAT_DIVIDE(elasticity_of_legacy_goals_wrt_stocks,
                               elasticity_of_wealth_wrt_stocks);

      return _ExpectedRunData{
          .wealth_starting = wealth_at_start,
          .elasticity_of_extra_withdrawal_goals_wrt_wealth =
              elasticity_of_extra_withdrawal_goals_wrt_wealth,
          .elasticity_of_legacy_goals_wrt_wealth =
              elasticity_of_legacy_goals_wrt_wealth,
      };
    }
    __global__ void _test_kernel(
        FLOAT stock_allocation,
        FLOAT legacy_stock_allocation,
        CURRENCY wealth_at_start,
        _NPVSpendingScaledAndConstrainedToWealth
            npv_approx_at_start_spending_scaled_and_constrained_to_wealth,
        _ExpectedRunData *result) {
      *result =
          fn(stock_allocation,
             legacy_stock_allocation,
             wealth_at_start,
             npv_approx_at_start_spending_scaled_and_constrained_to_wealth);
    }

    _ExpectedRunData _test_fn(
        FLOAT stock_allocation,
        FLOAT legacy_stock_allocation,
        CURRENCY wealth_at_start,
        _NPVSpendingScaledAndConstrainedToWealth
            npv_approx_at_start_spending_scaled_and_constrained_to_wealth) {
      thrust::device_vector<_ExpectedRunData> result_vec(1);
      _test_kernel<<<1, 1>>>(
          stock_allocation,
          legacy_stock_allocation,
          wealth_at_start,
          npv_approx_at_start_spending_scaled_and_constrained_to_wealth,
          result_vec.data().get());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      return result_vec[0];
    };

    TEST_CASE("tpaw::_get_expected_run_data") {
      SUBCASE("basic_case") {
        const _ExpectedRunData result =
            _test_fn(FLOAT_L(0.6),
                     FLOAT_L(0.4),
                     100000.0,
                     _NPVSpendingScaledAndConstrainedToWealth{
                         .discretionary = 30000.0,
                         .legacy = 20000.0,
                         .general = 40000.0,
                     });
        const FLOAT elasticity_of_wealth_wrt_stocks =
            (30000.0 * 0.6 + 20000.0 * 0.4 + 40000.0 * 0.6) / 100000.0;

        CHECK(result.wealth_starting == 100000.0);
        CHECK(result.elasticity_of_extra_withdrawal_goals_wrt_wealth ==
              doctest::Approx(0.6 / elasticity_of_wealth_wrt_stocks));
        CHECK(result.elasticity_of_legacy_goals_wrt_wealth ==
              doctest::Approx(0.4 / elasticity_of_wealth_wrt_stocks));
      }
    }
  } // namespace _get_expected_run_data

  // *****************************************************************************
  // _get_target_withdrawals_assuming_no_ceiling_or_floor
  // *****************************************************************************
  namespace _get_target_withdrawals_assuming_no_ceiling_or_floor {
    __device__ __forceinline__ run_common::TargetWithdrawals
    fn(const bool withdrawal_started,
       const CURRENCY curr_month_essential_expense,
       const CURRENCY curr_month_discretionary_expense,
       const FLOAT curr_month_cumulative_1_plus_g_over_1_plus_r,
       const FLOAT expenses_scale_discretionary,
       const CURRENCY
           npv_approx_at_start_general_expenses_scaled_and_constrained_to_wealth) {

      return run_common::TargetWithdrawals{
          .essential = curr_month_essential_expense,
          .discretionary =
              curr_month_discretionary_expense * expenses_scale_discretionary,
          .general =
              !withdrawal_started
                  ? 0.0
                  : npv_approx_at_start_general_expenses_scaled_and_constrained_to_wealth /
                        static_cast<CURRENCY>(
                            curr_month_cumulative_1_plus_g_over_1_plus_r)};
    }

    __global__ void _test_kernel(
        bool withdrawal_started,
        CURRENCY curr_month_essential_expense,
        CURRENCY curr_month_discretionary_expense,
        FLOAT curr_month_cumulative_1_plus_g_over_1_plus_r,
        FLOAT expenses_scale_discretionary,
        CURRENCY
            npv_approx_at_start_general_expenses_scaled_and_constrained_to_wealth,
        run_common::TargetWithdrawals *result) {
      *result = fn(
          withdrawal_started,
          curr_month_essential_expense,
          curr_month_discretionary_expense,
          curr_month_cumulative_1_plus_g_over_1_plus_r,
          expenses_scale_discretionary,
          npv_approx_at_start_general_expenses_scaled_and_constrained_to_wealth);
    }

    run_common::TargetWithdrawals _test_fn(
        bool withdrawal_started,
        CURRENCY curr_month_essential_expense,
        CURRENCY curr_month_discretionary_expense,
        FLOAT curr_month_cumulative_1_plus_g_over_1_plus_r,
        FLOAT expenses_scale_discretionary,
        CURRENCY
            npv_approx_at_start_general_expenses_scaled_and_constrained_to_wealth) {
      thrust::device_vector<run_common::TargetWithdrawals> result_vec(1);

      _test_kernel<<<1, 1>>>(
          withdrawal_started,
          curr_month_essential_expense,
          curr_month_discretionary_expense,
          curr_month_cumulative_1_plus_g_over_1_plus_r,
          expenses_scale_discretionary,
          npv_approx_at_start_general_expenses_scaled_and_constrained_to_wealth,
          result_vec.data().get());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      return result_vec[0];
    }

    TEST_CASE("tpaw::_get_target_withdrawals_assuming_no_ceiling_or_floor") {
      SUBCASE("withdrawal_started") {
        const run_common::TargetWithdrawals result = _test_fn(
            true, 10000.0, 20000.0, FLOAT_L(0.05), FLOAT_L(1.2), 50000.0);
        CHECK(result.essential == doctest::Approx(10000.0));
        CHECK(result.discretionary == doctest::Approx(20000.0 * 1.2));
        CHECK(result.general == doctest::Approx(50000.0 / 0.05));
      }
    }
  } // namespace _get_target_withdrawals_assuming_no_ceiling_or_floor

  // *****************************************************************************
  // _get_stock_allocation
  // *******************************************************
  // *****************************************************************************

  namespace _get_stock_allocation {
    __device__ __forceinline__ FLOAT
    fn(const FLOAT stock_allocation_total_portfolio,
       const FLOAT legacy_stock_allocation,
       const CURRENCY npv_approx_income_without_current_month,
       const CURRENCY npv_approx_essential_expenses_without_current_month,
       const CURRENCY npv_approx_discretionary_expenses_without_current_month,
       const CURRENCY npv_approx_legacy,
       const _ExpensesScale expenses_scale,
       const CURRENCY balance_after_contributions_and_withdrawals,
       const bool debug = false) {

      // If savings portfolio balance is 0, we approximate the limit as it
      // goes to 0.
      const CURRENCY savings_portfolio_balance =
          fmax(balance_after_contributions_and_withdrawals, 0.00001);
      AccountForWithdrawal account(savings_portfolio_balance +
                                   npv_approx_income_without_current_month);

      account.withdraw(npv_approx_essential_expenses_without_current_month);

      const _NPVSpendingScaledAndConstrainedToWealth
          npv_spending_scaled_and_constrained_to_wealth{
              .discretionary = account.withdraw(
                  npv_approx_discretionary_expenses_without_current_month *
                  expenses_scale.discretionary),
              .legacy =
                  account.withdraw(npv_approx_legacy * expenses_scale.legacy),
              .general = account.balance,
          };

      const CURRENCY stocks_target =
          npv_spending_scaled_and_constrained_to_wealth.legacy *
              legacy_stock_allocation +
          npv_spending_scaled_and_constrained_to_wealth.discretionary *
              stock_allocation_total_portfolio +
          npv_spending_scaled_and_constrained_to_wealth.general *
              stock_allocation_total_portfolio;

      if (debug) {
        printf("==== expecses scale\n");
        expenses_scale.print(1);
        printf("==== npv_approx_discretionary_expenses_without_current_month: "
               "%.65f\n",
               npv_approx_discretionary_expenses_without_current_month);
        printf(
            "==== npv_approx_essential_expenses_without_current_month: %.65f\n",
            npv_approx_essential_expenses_without_current_month);
        printf("==== npv_approx_income_without_current_month: %.65f\n",
               npv_approx_income_without_current_month);
        printf("==== npv_approx_legacy: %.65f\n", npv_approx_legacy);
        printf("==== legacy_stock_allocation: %.65f\n",
               legacy_stock_allocation);
        printf("==== stock_allocation_total_portfolio: %.65f\n",
               stock_allocation_total_portfolio);
        printf("=====npv_spending_scaled_and_constrained_to_wealth\n");
        npv_spending_scaled_and_constrained_to_wealth.print(1);
        printf("=====stocks_target: %.65f\n", stocks_target);
        printf("=====savings_portfolio_balance: %.65f\n",
               savings_portfolio_balance);
      }
      const FLOAT result = __FLOAT_SATURATE(
          static_cast<FLOAT>(stocks_target / savings_portfolio_balance));
      return result;
    }

    __global__ void _test_kernel(
        FLOAT stock_allocation_total_portfolio,
        FLOAT legacy_stock_allocation,
        CURRENCY npv_approx_income_without_current_month,
        CURRENCY npv_approx_essential_expenses_without_current_month,
        CURRENCY npv_approx_discretionary_expenses_without_current_month,
        CURRENCY npv_approx_legacy,
        _ExpensesScale expenses_scale,
        CURRENCY balance_after_contributions_and_withdrawals,
        FLOAT *result) {
      result[0] = fn(stock_allocation_total_portfolio,
                     legacy_stock_allocation,
                     npv_approx_income_without_current_month,
                     npv_approx_essential_expenses_without_current_month,
                     npv_approx_discretionary_expenses_without_current_month,
                     npv_approx_legacy,
                     expenses_scale,
                     balance_after_contributions_and_withdrawals);
    }

    FLOAT
    _test_fn(FLOAT stock_allocation_total_portfolio,
             FLOAT legacy_stock_allocation,
             CURRENCY npv_approx_income_without_current_month,
             CURRENCY npv_approx_essential_expenses_without_current_month,
             CURRENCY npv_approx_discretionary_expenses_without_current_month,
             CURRENCY npv_approx_legacy,
             _ExpensesScale expenses_scale,
             CURRENCY balance_after_contributions_and_withdrawals) {
      thrust::device_vector<FLOAT> result_vec(1);
      _test_kernel<<<1, 1>>>(
          stock_allocation_total_portfolio,
          legacy_stock_allocation,
          npv_approx_income_without_current_month,
          npv_approx_essential_expenses_without_current_month,
          npv_approx_discretionary_expenses_without_current_month,
          npv_approx_legacy,
          expenses_scale,
          balance_after_contributions_and_withdrawals,
          result_vec.data().get());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      return result_vec[0];
    }

    TEST_CASE("tpaw::_get_stock_allocation") {
      SUBCASE("basic_case") {
        const FLOAT result = _test_fn(
            FLOAT_L(0.6),
            FLOAT_L(0.4),
            10000.0, // npv_approx_income_without_current_month
            10000.0, // npv_approx_essential_expenses_without_current_month
            30000.0, // npv_approx_discretionary_expenses_without_current_month
            20000.0, // npv_approx_legacy
            _ExpensesScale{.discretionary = FLOAT_L(1.2),
                           .legacy = FLOAT_L(0.8)},
            90000.0); // balance_after_contributions_and_withdrawals
        const CURRENCY legacy_scaled = 20000.0 * 0.8;
        const CURRENCY discretionary_scaled = 30000.0 * 1.2;
        const CURRENCY stocks_target = discretionary_scaled * 0.6 +
                                       legacy_scaled * 0.4 +
                                       (10000.0 + 90000.0 - 10000.0 -
                                        discretionary_scaled - legacy_scaled) *
                                           0.6;
        CHECK(result == doctest::Approx((stocks_target / 90000.0)));
      }
    }
  } // namespace _get_stock_allocation

  // *****************************************************************************
  // _single_month
  // *****************************************************************************
  namespace _single_month {
    struct _ReturnValue {
      FLOAT withdrawals_from_savings_portfolio_rate;
      CURRENCY balance_ending;

      __host__ __device__ void print(uint32_t num_tabs) const {
        const auto i = static_cast<int32_t>(num_tabs * 4);
        printf("%*s"
               "withdrawals_from_savings_portfolio_rate: %.65f\n",
               i,
               "",
               withdrawals_from_savings_portfolio_rate);
        printf("%*s"
               "balance_ending: %.65f\n",
               i,
               "",
               balance_ending);
      }
    };

    template <bool write_result, bool is_expected_run>
    __device__ __forceinline__ _ReturnValue
    fn(const uint32_t run_index,
       const uint32_t month_index,
       const CURRENCY balance_starting,
       const uint32_t withdrawal_start_month,
       const OptCURRENCY spending_ceiling,
       const OptCURRENCY spending_floor,
       const CURRENCY current_month_income,
       const CURRENCY current_month_essential_expense,
       const CURRENCY current_month_discretionary_expense,
       const StocksAndBondsFLOAT &returns,
       const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry
           &cuda_processed_run_x_mfn_simulated_x_mfn,
       _ExpectedRunData *expected_run_result_by_mfn,
       RunResultPadded_GPU *run_result_padded) {
      //   const bool debug = run_index == 0 && month_index == 432;
      constexpr bool debug = false;

      if (debug) {
        printf("\n----is_expected_run: %s\n",
               is_expected_run ? "true" : "false");
        printf("\n----write_result: %s\n", write_result ? "true" : "false");
        printf("\n----month_index: %u\n", month_index);
        printf("\n----run_index: %u\n", run_index);
        printf("\n----balance_starting: %.65f\n", balance_starting);
        printf("\n----cuda_processed_run_x_mfn_simulated_x_mfn: \n");
        cuda_processed_run_x_mfn_simulated_x_mfn.print(0);
      }

      const bool withdrawal_started = month_index >= withdrawal_start_month;

      // Step 1: Precomputation at start
      const _PrecomputationAtStart precomputation_at_start =
          _get_precomputation_at_start::fn<is_expected_run>(
              balance_starting,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .income_without_current_month,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .essential_expenses_without_current_month,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .discretionary_expenses_without_current_month,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx.legacy_exact,
              current_month_income,
              current_month_essential_expense,
              current_month_discretionary_expense,
              is_expected_run ? nullptr
                              : &expected_run_result_by_mfn[month_index]);
      if (debug) {
        printf("\n----precomputation_at_start:\n");
        precomputation_at_start.print(0);
      }

      // Step 2: Data for expected run (if needed).
      if (is_expected_run && write_result) {
        expected_run_result_by_mfn[month_index] = _get_expected_run_data::fn(
            cuda_processed_run_x_mfn_simulated_x_mfn
                .stock_allocation_total_portfolio,
            cuda_processed_run_x_mfn_simulated_x_mfn.legacy_stock_allocation,
            precomputation_at_start.wealth,
            precomputation_at_start
                .npv_approx_spending_scaled_and_constrained_to_wealth);
        if (debug) {
          printf("\n----expected_run_result_by_mfn:\n");
          expected_run_result_by_mfn[month_index].print(0);
        }
      }

      // Step 3: Calculate target withdrawals before ceiling and floor (note
      // this  happens before contributions are applied).
      const run_common::TargetWithdrawals
          target_withdrawals_before_ceiling_and_floor =
              _get_target_withdrawals_assuming_no_ceiling_or_floor::fn(
                  withdrawal_started,
                  current_month_essential_expense,
                  current_month_discretionary_expense,
                  cuda_processed_run_x_mfn_simulated_x_mfn
                      .cumulative_1_plus_g_over_1_plus_r,
                  precomputation_at_start.expenses_scale.discretionary,
                  precomputation_at_start
                      .npv_approx_spending_scaled_and_constrained_to_wealth
                      .general);
      if (debug) {
        printf("\n----target_withdrawals_before_ceiling_and_floor:\n");
        target_withdrawals_before_ceiling_and_floor.print(0);
      }

      // Step 4: Apply ceiling and floor to target withdrawals
      const run_common::TargetWithdrawals target_withdrawals =
          run_common::apply_withdrawal_ceiling_and_floor(
              target_withdrawals_before_ceiling_and_floor,
              // Note: this should NOT be the scaled discretionary expense.
              current_month_discretionary_expense,
              spending_ceiling,
              spending_floor,
              withdrawal_started,
              debug);
      if (debug) {
        printf("\n----target_withdrawals:\n");
        target_withdrawals.print(0);
      }

      // Step 5: Apply contributions and withdrawals
      const run_common::AfterContributionsAndWithdrawals
          savings_portfolio_after_contributions_and_withdrawals =
              run_common::apply_contributions_and_withdrawals(
                  balance_starting,
                  current_month_income,
                  target_withdrawals,
                  debug);
      //   if (run_index == 0 &&
      //       savings_portfolio_after_contributions_and_withdrawals
      //           .insufficent_funds) {
      //     printf("insuffient month: %u\n", month_index);
      //   }
      if (debug) {
        printf(
            "\n----savings_portfolio_after_contributions_and_withdrawals:\n");
        savings_portfolio_after_contributions_and_withdrawals.print(0);
      }

      // Step 5.5: Handle NaN withdrawals from savings portfolio rate.
      const FLOAT withdrawals_from_savings_portfolio_rate =
          (!isfinite(savings_portfolio_after_contributions_and_withdrawals
                         .withdrawals.from_savings_portfolio_rate_or_nan_or_inf)
               ? ({
                   fn<false, is_expected_run>(
                       run_index,
                       month_index,
                       0.000001,
                       withdrawal_start_month,
                       spending_ceiling,
                       spending_floor,
                       current_month_income,
                       current_month_essential_expense,
                       current_month_discretionary_expense,
                       returns,
                       cuda_processed_run_x_mfn_simulated_x_mfn,
                       expected_run_result_by_mfn,
                       run_result_padded)
                       .withdrawals_from_savings_portfolio_rate;
                 })
               : savings_portfolio_after_contributions_and_withdrawals
                     .withdrawals.from_savings_portfolio_rate_or_nan_or_inf);

      // Step 6: Calculate stock allocation
      const FLOAT stock_allocation = _get_stock_allocation::fn(
          cuda_processed_run_x_mfn_simulated_x_mfn
              .stock_allocation_total_portfolio,
          cuda_processed_run_x_mfn_simulated_x_mfn.legacy_stock_allocation,
          cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
              .income_without_current_month,
          cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
              .essential_expenses_without_current_month,
          cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
              .discretionary_expenses_without_current_month,
          cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx.legacy_exact,
          precomputation_at_start.expenses_scale,
          savings_portfolio_after_contributions_and_withdrawals.balance,
          debug);
      if (debug) {
        printf("\n----stock_allocation:\n");
        printf("    %.65f\n", stock_allocation);
      }

      // Step 7: Apply allocation
      const run_common::End savings_portfolio_at_end =
          run_common::apply_allocation(
              stock_allocation,
              returns,
              savings_portfolio_after_contributions_and_withdrawals.balance,
              cuda_processed_run_x_mfn_simulated_x_mfn.npv_approx
                  .income_without_current_month);
      if (debug) {
        printf("\n----savings_portfolio_at_end:\n");
        savings_portfolio_at_end.print(0);
      }

      if (debug) {
        printf("\n----important as array:\n");
        printf("%.1f,\n", static_cast<double>(run_index));
        printf("%.1f,\n", static_cast<double>(month_index));
        printf("%.65f,\n", balance_starting);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals
                   .contributions);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals.withdrawals
                   .essential);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals.withdrawals
                   .discretionary);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals.withdrawals
                   .general);
        printf("%.65f,\n",
               savings_portfolio_after_contributions_and_withdrawals.withdrawals
                   .total);
        printf("%.65f,\n", withdrawals_from_savings_portfolio_rate);
        printf("%.65f,\n",
               savings_portfolio_at_end
                   .stock_allocation_on_total_portfolio_or_zero_if_no_wealth);
        printf("%.65f,\n", stock_allocation);
        printf("%.65f,\n", savings_portfolio_at_end.balance);
      }

      // Step 8: Write results (in write_result)
      if (write_result && !is_expected_run) {
        run_common::write_result(
            balance_starting,
            savings_portfolio_after_contributions_and_withdrawals,
            withdrawals_from_savings_portfolio_rate,
            savings_portfolio_at_end,
            cuda_processed_run_x_mfn_simulated_x_mfn.spending_tilt,
            *run_result_padded,
            run_index,
            month_index);
      }

      // Step 9 :Return
      return _ReturnValue{
          .withdrawals_from_savings_portfolio_rate =
              withdrawals_from_savings_portfolio_rate,
          .balance_ending = savings_portfolio_at_end.balance,
      };
    }

    template <bool write_result, bool is_expected_run>
    __global__ void
    _test_kernel(const uint32_t run_index,
                 const uint32_t month_index,
                 const CURRENCY balance_starting,
                 const uint32_t withdrawal_start_month,
                 const OptCURRENCY spending_ceiling,
                 const OptCURRENCY spending_floor,
                 const CURRENCY current_month_income,
                 const CURRENCY current_month_essential_expense,
                 const CURRENCY current_month_discretionary_expense,
                 const StocksAndBondsFLOAT returns,
                 const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry
                     cuda_processed_run_x_mfn_simulated_x_mfn,
                 _ExpectedRunData *const expected_run_result_by_mfn,
                 RunResultPadded_GPU *const run_result_padded,
                 _ReturnValue *const result) {
      result[0] = fn<write_result, is_expected_run>(
          run_index,
          month_index,
          balance_starting,
          withdrawal_start_month,
          spending_ceiling,
          spending_floor,
          current_month_income,
          current_month_essential_expense,
          current_month_discretionary_expense,
          returns,
          cuda_processed_run_x_mfn_simulated_x_mfn,
          expected_run_result_by_mfn,
          run_result_padded);
    }

    template <bool write_result, bool is_expected_run>
    std::tuple<_ReturnValue, _ExpectedRunData, RunResult_Single>
    _test_fn(const uint32_t month_index,
             const CURRENCY balance_starting,
             const uint32_t withdrawal_start_month,
             const OptCURRENCY spending_ceiling,
             const OptCURRENCY spending_floor,
             const CURRENCY current_month_income,
             const CURRENCY current_month_essential_expense,
             const CURRENCY current_month_discretionary_expense,
             const StocksAndBondsFLOAT &returns,
             const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry
                 &cuda_processed_run_x_mfn_simulated_x_mfn,
             const _ExpectedRunData &expected_run_data) {
      thrust::device_vector<_ExpectedRunData> d_expected_run_result_by_mfn(
          1, expected_run_data);

      RunResultPadded host_struct_of_device_vectors =
          RunResultPadded::make(1, month_index + 1);
      unique_ptr_gpu<RunResultPadded_GPU> run_result_padded =
          host_struct_of_device_vectors.copy_to_gpu();

      thrust::device_vector<_ReturnValue> result_vec(1);
      _test_kernel<write_result, is_expected_run>
          <<<1, 1>>>(0,
                     month_index,
                     balance_starting,
                     withdrawal_start_month,
                     spending_ceiling,
                     spending_floor,
                     current_month_income,
                     current_month_essential_expense,
                     current_month_discretionary_expense,
                     returns,
                     cuda_processed_run_x_mfn_simulated_x_mfn,
                     d_expected_run_result_by_mfn.data().get(),
                     run_result_padded.get(),
                     result_vec.data().get());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      return std::make_tuple(
          result_vec[0],
          d_expected_run_result_by_mfn[0],
          host_struct_of_device_vectors.get_single(0, month_index));
    }

    TEST_CASE("tpaw::_single_month") {
      const _ExpectedRunData expected_run_data = {
          .wealth_starting = 100000.0,
          .elasticity_of_extra_withdrawal_goals_wrt_wealth = 0,
          .elasticity_of_legacy_goals_wrt_wealth = 1.1319,
      };

      // auto [return_value, expectedRunData, runResult] =
      _test_fn<true, false>(
          0,                   // month_index
          100000.0,            // balance_starting
          0,                   // withdrawal_start_month
          OptCURRENCY{0, 0.0}, // spending_ceiling
          OptCURRENCY{0, 0.0}, // spending_floor
          0.0,                 // current_month_income
          0.0,                 // current_month_essential_expense
          0.0,                 // current_month_discretionary_expense
          StocksAndBondsFLOAT{
              .stocks = 0.00327374,
              .bonds = 0.00165158,
          },
          Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry{
              .npv_approx =
                  {
                      .income_without_current_month = 0.0,
                      .essential_expenses_without_current_month = 0.0,
                      .discretionary_expenses_without_current_month = 0.0,
                      .legacy_exact = 88347.3,
                  },
              .stock_allocation_total_portfolio = 0,
              .legacy_stock_allocation = 0.889091,
              .spending_tilt = 0.0,
              .cumulative_1_plus_g_over_1_plus_r = 5.97532,
          },
          expected_run_data);
    }

  } // namespace _single_month

  // *****************************************************************************
  // _kernel
  // *****************************************************************************

  template <bool is_expected_run>
  __global__ void
  _kernel(const uint32_t num_runs,
          const uint32_t num_months_to_simulate,
          const CURRENCY current_portfolio_balance,
          const uint32_t withdrawal_start_month,
          const OptCURRENCY spending_ceiling,
          const OptCURRENCY spending_floor,
          const StocksAndBondsFLOAT expected_returns_at_month_0,
          const CURRENCY *const income_by_mfn,
          const CURRENCY *const essential_expense_by_mfn,
          const CURRENCY *const discretionary_expense_by_mfn,
          const StocksAndBondsFLOAT
              *const historical_returns_by_run_by_mfn_simulated,
          const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry
              *const cuda_processed_run_x_mfn_simulated_x_mfn,
          _ExpectedRunData *const expected_run_result_by_mfn,
          RunResultPadded_GPU *const run_result_padded) {
    const uint32_t run_index = (threadIdx.x + blockIdx.x * blockDim.x);
    if (run_index >= num_runs)
      return;

    CURRENCY balance_starting = current_portfolio_balance;
    for (uint32_t month_index = 0; month_index < num_months_to_simulate;
         month_index++) {
      const uint32_t run_by_mfn_simulated_index = get_run_by_mfn_index(
          num_runs, num_months_to_simulate, run_index, month_index);
      balance_starting =
          _single_month::fn<true, is_expected_run>(
              run_index,
              month_index,
              balance_starting,
              withdrawal_start_month,
              spending_ceiling,
              spending_floor,
              income_by_mfn[month_index],
              essential_expense_by_mfn[month_index],
              discretionary_expense_by_mfn[month_index],
              is_expected_run ? expected_returns_at_month_0
                              : historical_returns_by_run_by_mfn_simulated
                                    [run_by_mfn_simulated_index],
              cuda_processed_run_x_mfn_simulated_x_mfn
                  [run_by_mfn_simulated_index],
              expected_run_result_by_mfn,
              run_result_padded)
              .balance_ending;
    }
  }

  // *****************************************************************************
  // run
  // *****************************************************************************

  RunResultPadded
  run(const uint32_t num_runs_in,
      const uint32_t num_months_to_simulate,
      const CURRENCY current_portfolio_balance,
      const uint32_t withdrawal_start_month,
      const OptCURRENCY spending_ceiling,
      const OptCURRENCY spending_floor,
      const StocksAndBondsFLOAT expected_returns_at_month_0,
      const thrust::device_vector<CURRENCY> &income_by_mfn,
      const thrust::device_vector<CURRENCY> &essential_expense_by_mfn,
      const thrust::device_vector<CURRENCY> &discretionary_expense_by_mfn,
      const thrust::device_vector<StocksAndBondsFLOAT>
          &historical_returns_by_run_by_mfn_simulated,
      const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN
          &cuda_processed_run_x_mfn_simulated_x_mfn) {

    thrust::device_vector<_ExpectedRunData> expected_run_result_by_mfn(
        num_months_to_simulate);
    RunResultPadded run_result_padded_host_struct =
        RunResultPadded::make(num_runs_in, num_months_to_simulate);
    unique_ptr_gpu<RunResultPadded_GPU> run_result_padded_device_struct =
        run_result_padded_host_struct.copy_to_gpu();

    const auto do_run = [&]<bool is_expected_run>() {
      const int32_t num_runs = is_expected_run ? 1 : num_runs_in;
      const int32_t block_size = 64;
      _kernel<is_expected_run>
          <<<(num_runs + block_size - 1) / block_size, block_size>>>(
              num_runs,
              num_months_to_simulate,
              current_portfolio_balance,
              withdrawal_start_month,
              spending_ceiling,
              spending_floor,
              expected_returns_at_month_0,
              income_by_mfn.data().get(),
              essential_expense_by_mfn.data().get(),
              discretionary_expense_by_mfn.data().get(),
              historical_returns_by_run_by_mfn_simulated.data().get(),
              is_expected_run
                  ? cuda_processed_run_x_mfn_simulated_x_mfn.for_expected_run
                        .data()
                        .get()
                  : cuda_processed_run_x_mfn_simulated_x_mfn.for_normal_run
                        .data()
                        .get(),
              expected_run_result_by_mfn.data().get(),
              run_result_padded_device_struct.get());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
    };

    do_run.operator()<true>();
    do_run.operator()<false>();

    return run_result_padded_host_struct;
  }

  TEST_CASE("run_tpaw") {
    const uint32_t num_runs = 1;
    const uint32_t num_months = 6;
    const uint32_t num_months_to_simulate = num_months;
    const FLOAT legacy_rra_including_pos_infinity = 0.675854;
    const FLOAT legacy = 90000.0;
    const FLOAT empirical_log_variance_stocks = FLOAT_L(0.00269957);
    const FLOAT time_preference = FLOAT_L(0.00);
    const FLOAT additional_spending_tilt_plus_one = FLOAT_L(1.0);
    const CURRENCY current_portfolio_balance = 100000.0;
    const uint32_t withdrawal_start_month = 0;
    const OptCURRENCY spending_ceiling = OptCURRENCY{0, 0.0};
    const OptCURRENCY spending_floor = OptCURRENCY{0, 0.0};

    const StocksAndBondsFLOAT current_monthly_expected_returns = {
        //   .stocks = annual_to_monthly_return(0.05),
        //   .bonds = annual_to_monthly_return(0.00),
        .stocks = 0.00327374,
        .bonds = 0.00165158,
    };
    const auto current_expected_returns = MonthlyAndAnnual<StocksAndBondsFLOAT>{
        .monthly = current_monthly_expected_returns,
        .annual = {
            .stocks = monthly_to_annual_return(
                current_monthly_expected_returns.stocks),
            .bonds = monthly_to_annual_return(
                current_monthly_expected_returns.bonds),
        }};

    std::vector<CURRENCY> income_by_mfn_cpu(num_months, 0.0);
    std::vector<CURRENCY> essential_expense_by_mfn_cpu(num_months, 0.0);
    std::vector<CURRENCY> discretionary_expense_by_mfn_cpu(num_months, 0.0);
    std::vector<FLOAT> rra_including_pos_infinity_by_mfn_cpu(num_months,
                                                             INFINITY);
    thrust::device_vector<CURRENCY> income_by_mfn_gpu(income_by_mfn_cpu);
    thrust::device_vector<CURRENCY> essential_expense_by_mfn_gpu(
        essential_expense_by_mfn_cpu);
    thrust::device_vector<CURRENCY> discretionary_expense_by_mfn_gpu(
        discretionary_expense_by_mfn_cpu);

    thrust::device_vector<StocksAndBondsFLOAT>
        historical_returns_by_run_by_mfn_simulated_gpu(
            static_cast<size_t>(num_runs * num_months_to_simulate),
            current_monthly_expected_returns);

    thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        expected_returns_by_run_by_mfn_simulated_gpu(
            static_cast<size_t>(num_runs * num_months_to_simulate),
            current_expected_returns);

    thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
        expected_returns_by_mfn_simulated_for_expected_run_gpu(
            static_cast<size_t>(num_months_to_simulate),
            current_expected_returns);

    const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN
        cuda_processed_run_x_mfn_simulated_x_mfn =
            cuda_process_tpaw_run_x_mfn_simulated_x_mfn(
                num_runs,
                num_months_to_simulate,
                num_months,
                income_by_mfn_cpu,
                essential_expense_by_mfn_cpu,
                discretionary_expense_by_mfn_cpu,
                rra_including_pos_infinity_by_mfn_cpu,
                empirical_log_variance_stocks,
                time_preference,
                additional_spending_tilt_plus_one,
                legacy_rra_including_pos_infinity,
                legacy,
                expected_returns_by_mfn_simulated_for_expected_run_gpu,
                expected_returns_by_run_by_mfn_simulated_gpu);

    std::cout << "cuda_processed_run_x_mfn_simulated_x_mfn:\n";

    auto run_result_padded = run(num_runs,
                                 num_months_to_simulate,
                                 current_portfolio_balance,
                                 withdrawal_start_month,
                                 spending_ceiling,
                                 spending_floor,
                                 current_monthly_expected_returns,
                                 income_by_mfn_gpu,
                                 essential_expense_by_mfn_gpu,
                                 discretionary_expense_by_mfn_gpu,
                                 historical_returns_by_run_by_mfn_simulated_gpu,
                                 cuda_processed_run_x_mfn_simulated_x_mfn);

    {
      std::vector<Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry>
          for_expected_run_host = device_vector_to_host(
              cuda_processed_run_x_mfn_simulated_x_mfn.for_expected_run);

      std::vector<Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN::Entry>
          for_normal_run_host = device_vector_to_host(
              cuda_processed_run_x_mfn_simulated_x_mfn.for_normal_run);
    }
  }

  // *****************************************************************************
  // BENCHES
  // *********************************************************************
  // *****************************************************************************

  TEST_CASE("bench::run_tpaw") {

    const auto do_bench = [](const char *name,
                             const uint32_t num_runs,
                             const uint32_t num_months,
                             const bool direct_kernel_call) {
      const uint32_t num_months_to_simulate = num_months;

      const StocksAndBondsFLOAT current_expected_returns = {
          .stocks = annual_to_monthly_return(0.05),
          .bonds = annual_to_monthly_return(0.02),
      };

      const auto current_expected_returns_by_mfn_simulated =
          MonthlyAndAnnual<StocksAndBondsFLOAT>{
              .monthly = current_expected_returns,
              .annual = {
                  .stocks =
                      monthly_to_annual_return(current_expected_returns.stocks),
                  .bonds =
                      monthly_to_annual_return(current_expected_returns.bonds),
              }};

      std::vector<CURRENCY> income_by_mfn_cpu(num_months, 1000.0);
      std::vector<CURRENCY> essential_expense_by_mfn_cpu(num_months, 400.0);
      std::vector<CURRENCY> discretionary_expense_by_mfn_cpu(num_months,
                                                             9000.0);
      std::vector<FLOAT> rra_including_pos_infinity_by_mfn_cpu(num_months,
                                                               FLOAT_L(4.0));
      thrust::device_vector<CURRENCY> income_by_mfn_gpu(income_by_mfn_cpu);
      thrust::device_vector<CURRENCY> essential_expense_by_mfn_gpu(
          essential_expense_by_mfn_cpu);
      thrust::device_vector<CURRENCY> discretionary_expense_by_mfn_gpu(
          discretionary_expense_by_mfn_cpu);

      thrust::device_vector<StocksAndBondsFLOAT>
          historical_returns_by_run_by_mfn_simulated_gpu(
              static_cast<size_t>(num_runs * num_months_to_simulate),
              current_expected_returns);

      thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
          expected_returns_by_mfn_simulated_for_expected_run_gpu(
              static_cast<size_t>(num_months_to_simulate),
              current_expected_returns_by_mfn_simulated);

      thrust::device_vector<MonthlyAndAnnual<StocksAndBondsFLOAT>>
          expected_returns_by_run_by_mfn_simulated_gpu(
              static_cast<size_t>(num_runs * num_months_to_simulate),
              MonthlyAndAnnual<StocksAndBondsFLOAT>{
                  .monthly = current_expected_returns,
                  .annual = {
                      .stocks = monthly_to_annual_return(
                          current_expected_returns.stocks),
                      .bonds = monthly_to_annual_return(
                          current_expected_returns.bonds),
                  }});

      const Cuda_Processed_TPAW_Run_x_MFNSimulated_x_MFN
          cuda_processed_run_x_mfn_simulated_x_mfn =
              cuda_process_tpaw_run_x_mfn_simulated_x_mfn(
                  num_runs,
                  num_months_to_simulate,
                  num_months,
                  income_by_mfn_cpu,
                  essential_expense_by_mfn_cpu,
                  discretionary_expense_by_mfn_cpu,
                  rra_including_pos_infinity_by_mfn_cpu,
                  FLOAT_L(0.01), // empirical_log_variance_stocks
                  FLOAT_L(0.01), // time_preference
                  FLOAT_L(1.0),  // additional_spending_tilt_plus_one
                  5.0,           // legacy_rra_including_pos_infinity
                  1000.0,
                  expected_returns_by_mfn_simulated_for_expected_run_gpu,
                  expected_returns_by_run_by_mfn_simulated_gpu);

      ankerl::nanobench::Bench()
          .timeUnit(std::chrono::milliseconds{1}, "ms")
          .run(name, [&]() {
            const std::vector<_ExpectedRunData> expected_run_result_host(
                num_months_to_simulate,
                _ExpectedRunData{
                    .wealth_starting = 100000.0,
                    .elasticity_of_extra_withdrawal_goals_wrt_wealth = 1.0,
                    .elasticity_of_legacy_goals_wrt_wealth = 1.0,
                });
            thrust::device_vector<_ExpectedRunData> expected_run_result_gpu(
                expected_run_result_host);

            RunResultPadded run_result_host_struct_of_device_vectors =
                RunResultPadded::make(num_runs, num_months_to_simulate);

            unique_ptr_gpu<RunResultPadded_GPU>
                run_result_device_struct_of_device_pointers =
                    run_result_host_struct_of_device_vectors.copy_to_gpu();

            if (direct_kernel_call) {
              const uint32_t block_size{64};
              _kernel<false>
                  <<<(num_runs + block_size - 1) / block_size, block_size>>>(
                      num_runs,
                      num_months_to_simulate,
                      100000.0,                // current_portfolio_balance
                      num_months / 2,          // withdrawal_start_month
                      OptCURRENCY{1, 10000.0}, // spending_ceiling
                      OptCURRENCY{0, 0.0},     // spending_floor
                      current_expected_returns,
                      income_by_mfn_gpu.data().get(),
                      essential_expense_by_mfn_gpu.data().get(),
                      discretionary_expense_by_mfn_gpu.data().get(),
                      historical_returns_by_run_by_mfn_simulated_gpu.data()
                          .get(),
                      cuda_processed_run_x_mfn_simulated_x_mfn.for_normal_run
                          .data()
                          .get(),
                      expected_run_result_gpu.data().get(),
                      run_result_device_struct_of_device_pointers.get());
              gpuErrchk(cudaPeekAtLastError());
              gpuErrchk(cudaDeviceSynchronize());
            } else {
              run(num_runs,
                  num_months_to_simulate,
                  100000.0,
                  num_months / 2,
                  OptCURRENCY{1, 10000.0},
                  OptCURRENCY{0, 0.0},
                  current_expected_returns,
                  income_by_mfn_gpu,
                  essential_expense_by_mfn_gpu,
                  discretionary_expense_by_mfn_gpu,
                  historical_returns_by_run_by_mfn_simulated_gpu,
                  cuda_processed_run_x_mfn_simulated_x_mfn);
            }
          });
    };

    //   do_bench("expected", 1, 120 * 12, true)
    for (const auto &direct_kernel_call : std::vector<bool>{true, false}) {
      for (const auto &num_runs : bench_num_runs_vec) {
        for (const auto &num_years : bench_num_years_vec) {
          do_bench((std::string("run_tpaw::") + std::to_string(num_runs) + "x" +
                    std::to_string(num_years) +
                    (direct_kernel_call ? "xdirect" : "xindirect"))
                       .c_str(),
                   num_runs,
                   num_years * 12,
                   direct_kernel_call);
        }
      }
    }
  }

} // namespace tpaw