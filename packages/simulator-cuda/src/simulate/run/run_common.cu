#include "extern/doctest.h"
#include "run_common.h"
#include <thrust/device_vector.h>
#include <vector>

namespace run_common {
  __host__ __device__ void AfterContributionsAndWithdrawals::Withdrawals::print(
      uint32_t num_tabs) const {
    const uint32_t num_spaces = num_tabs * 4;
    printf("%*sessential: %.65f\n", num_spaces, "", essential);
    printf("%*sdiscretionary: %.65f\n", num_spaces, "", discretionary);
    printf("%*sgeneral: %.65f\n", num_spaces, "", general);
    printf("%*stotal: %.65f\n", num_spaces, "", total);
    printf("%*sfrom_savings_portfolio_rate_or_nan_or_inf: %.65f\n",
           num_spaces,
           "",
           from_savings_portfolio_rate_or_nan_or_inf);
  }

  __host__ __device__ void
  AfterContributionsAndWithdrawals::print(uint32_t num_tabs) const {
    const uint32_t num_spaces = num_tabs * 4;
    printf("%*scontributions: %.65f\n", num_spaces, "", contributions);
    printf("%*swithdrawals:\n", num_spaces, "");
    withdrawals.print(num_tabs + 1);
    printf("%*sbalance: %.65f\n", num_spaces, "", balance);
    printf("%*sinsufficient_funds: %s\n",
           num_spaces,
           "",
           (insufficent_funds ? "true" : "false"));
  }

  __host__ __device__ void TargetWithdrawals::print(uint32_t num_tabs) const {
    const uint32_t num_spaces = num_tabs * 4;
    printf("%*sessential: %.65f\n", num_spaces, "", essential);
    printf("%*sdiscretionary: %.65f\n", num_spaces, "", discretionary);
    printf("%*sgeneral: %.65f\n", num_spaces, "", general);
  }

  __host__ __device__ void End::print(uint32_t num_tabs) const {
    const uint32_t num_spaces = num_tabs * 4;
    printf("%*sstock_allocation_percent_before_returns: %.65f\n",
           num_spaces,
           "",
           stock_allocation_percent_before_returns);
    printf("%*sstock_allocation_amount_before_returns: %.65f\n",
           num_spaces,
           "",
           stock_allocation_amount_before_returns);
    printf("%*sbalance: %.65f\n", num_spaces, "", balance);
    printf(
        "%*sstock_allocation_on_total_portfolio_or_zero_if_no_wealth: %.65f\n",
        num_spaces,
        "",
        stock_allocation_on_total_portfolio_or_zero_if_no_wealth);
  }

  // ***************************************************************************
  // apply_withdrawal_ceiling_and_floor ****************************************
  // ***************************************************************************

  namespace _test_apply_withdrawal_ceiling_and_floor {
    __global__ void
    _test_kernel(const TargetWithdrawals target,
                 const CURRENCY curr_month_discretionary_expense,
                 const OptCURRENCY spending_ceiling,
                 const OptCURRENCY spending_floor,
                 const bool withdrawal_started,
                 TargetWithdrawals *result) {
      result[0] =
          apply_withdrawal_ceiling_and_floor(target,
                                             curr_month_discretionary_expense,
                                             spending_ceiling,
                                             spending_floor,
                                             withdrawal_started);
    }

    TargetWithdrawals _test_fn(const TargetWithdrawals &target,
                               const CURRENCY curr_month_discretionary_expense,
                               const OptCURRENCY spending_ceiling,
                               const OptCURRENCY spending_floor,
                               const bool withdrawal_started) {

      thrust::device_vector<TargetWithdrawals> d_result(1);

      _test_kernel<<<1, 1>>>(target,
                             curr_month_discretionary_expense,
                             spending_ceiling,
                             spending_floor,
                             withdrawal_started,
                             d_result.data().get());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      std::vector<TargetWithdrawals> h_result = device_vector_to_host(d_result);
      return h_result[0];
    }

    TEST_CASE("run_common::apply_withdrawal_ceiling_and_floor") {
      SUBCASE("none") {
        const TargetWithdrawals result = _test_fn(
            TargetWithdrawals{
                .essential = 1000.0,
                .discretionary = 2000.0,
                .general = 3000.0,
            },
            500.0,
            OptCURRENCY(false, 300.0),
            OptCURRENCY(false, 500.0),
            true);
        CHECK(result.essential == 1000.0);
        CHECK(result.discretionary == 2000.0);
        CHECK(result.general == 3000.0);
      }

      SUBCASE("ceiling") {
        const TargetWithdrawals result = _test_fn(
            TargetWithdrawals{
                .essential = 1000.0,
                .discretionary = 2000.0,
                .general = 3000.0,
            },
            500.0,
            OptCURRENCY(true, 600.0),
            OptCURRENCY(false, 300.0),
            true);
        CHECK(result.essential == 1000.0);
        CHECK(result.discretionary == 500.0);
        CHECK(result.general == 600.0);
      }

      SUBCASE("floor_withdrawal_started") {
        const TargetWithdrawals result = _test_fn(
            TargetWithdrawals{
                .essential = 1000.0,
                .discretionary = 2000.0,
                .general = 3000.0,
            },
            500.0,
            OptCURRENCY(false, 600.0),
            OptCURRENCY(true, 3010.0),
            true);
        CHECK(result.essential == 1000.0);
        CHECK(result.discretionary == 2000.0);
        CHECK(result.general == 3010.0);
      }
      SUBCASE("floor_withdrawal_not_started") {
        const TargetWithdrawals result = _test_fn(
            TargetWithdrawals{
                .essential = 1000.0,
                .discretionary = 2000.0,
                .general = 0.0,
            },
            500.0,
            OptCURRENCY(false, 600.0),
            OptCURRENCY(true, 3010.0),
            false);
        CHECK(result.essential == 1000.0);
        CHECK(result.discretionary == 2000.0);
        CHECK(result.general == 0.0);
      }
    }
  } // namespace _test_apply_withdrawal_ceiling_and_floor

  // ***************************************************************************
  // apply_contributions_and_withdrawals ***************************************
  // ***************************************************************************

  namespace _test_apply_contributions_and_withdrawals {
    __global__ void _test_kernel(const CURRENCY balance_starting,
                                 const CURRENCY contributions,
                                 const TargetWithdrawals target_withdrawals,
                                 AfterContributionsAndWithdrawals *result) {
      result[0] = apply_contributions_and_withdrawals(
          balance_starting, contributions, target_withdrawals);
    }

    AfterContributionsAndWithdrawals
    _test_fn(const CURRENCY balance_starting,
             const CURRENCY contributions,
             const TargetWithdrawals &target_withdrawals) {

      thrust::device_vector<AfterContributionsAndWithdrawals> d_result(1);

      _test_kernel<<<1, 1>>>(balance_starting,
                             contributions,
                             target_withdrawals,
                             d_result.data().get());

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      std::vector<AfterContributionsAndWithdrawals> h_result =
          device_vector_to_host(d_result);
      return h_result[0];
    }

    TEST_CASE("run_common::apply_contributions_and_withdrawals") {
      SUBCASE("sufficient_funds") {
        const AfterContributionsAndWithdrawals result =
            _test_fn(10000.0,
                     1000.0,
                     TargetWithdrawals{
                         .essential = 1000.0,
                         .discretionary = 2000.0,
                         .general = 3000.0,
                     });
        struct Withdrawals {
          CURRENCY essential;
          CURRENCY discretionary;
          CURRENCY general;
          CURRENCY total;
          FLOAT from_savings_portfolio_rate_or_nan_or_inf;
        };

        CHECK(result.contributions == 1000.0);
        CHECK(result.withdrawals.essential == 1000.0);
        CHECK(result.withdrawals.discretionary == 2000.0);
        CHECK(result.withdrawals.general == 3000.0);
        CHECK(result.withdrawals.total == 6000.0);
        CHECK(result.withdrawals.from_savings_portfolio_rate_or_nan_or_inf ==
              doctest::Approx((6000.0 - 1000.0) / 10000.0));
        CHECK(result.balance == 10000.0 + 1000.0 - 6000.0);
        CHECK(result.insufficent_funds == false);
      }

      SUBCASE("insufficient_funds") {
        const AfterContributionsAndWithdrawals result =
            _test_fn(3000.0,
                     1000.0,
                     TargetWithdrawals{
                         .essential = 1000.0,
                         .discretionary = 2000.0,
                         .general = 3000.0,
                     });
        struct Withdrawals {
          CURRENCY essential;
          CURRENCY discretionary;
          CURRENCY general;
          CURRENCY total;
          FLOAT from_savings_portfolio_rate_or_nan_or_inf;
        };

        CHECK(result.contributions == 1000.0);
        CHECK(result.withdrawals.essential == 1000.0);
        CHECK(result.withdrawals.discretionary == 2000.0);
        CHECK(result.withdrawals.general == 1000.0);
        CHECK(result.withdrawals.total == 4000.0);
        CHECK(result.withdrawals.from_savings_portfolio_rate_or_nan_or_inf ==
              doctest::Approx((4000.0 - 1000.0) / 3000.0));
        CHECK(result.balance == 3000.0 + 1000.0 - 4000.0);
        CHECK(result.insufficent_funds == true);
      }
    }
  } // namespace _test_apply_contributions_and_withdrawals

  // ***************************************************************************
  // apply_allocation **********************************************************
  // ***************************************************************************

  namespace _test_apply_allocation {
    __global__ void
    _test_kernel(const FLOAT stock_allocation_percent,
                 const StocksAndBondsFLOAT return_rate,
                 const CURRENCY balance_after_contributions_and_withdrawals,
                 const CURRENCY npv_income_without_current_month,
                 End *result) {
      result[0] = apply_allocation(stock_allocation_percent,
                                   return_rate,
                                   balance_after_contributions_and_withdrawals,
                                   npv_income_without_current_month);
    }

    End _test_fn(FLOAT stock_allocation_percent,
                 const StocksAndBondsFLOAT &return_rate,
                 const CURRENCY balance_after_contributions_and_withdrawals,
                 const CURRENCY npv_income_without_current_month) {
      thrust::device_vector<End> d_result(1);

      _test_kernel<<<1, 1>>>(stock_allocation_percent,
                             return_rate,
                             balance_after_contributions_and_withdrawals,
                             npv_income_without_current_month,
                             d_result.data().get());

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      std::vector<End> h_result = device_vector_to_host(d_result);
      return h_result[0];
    }

    TEST_CASE("run_common::apply_allocation") {
      SUBCASE("basic") {
        const End result = _test_fn(0.5, {0.05, 0.03}, 1000.0, 100.0);
        const CURRENCY stocks_amount = 1000.0 * 0.5;
        CHECK(result.stock_allocation_percent_before_returns == 0.5);
        CHECK(result.stock_allocation_amount_before_returns ==
              doctest::Approx(stocks_amount));
        CHECK(result.balance ==
              doctest::Approx(stocks_amount * 1.05 +
                              (1000.0 - stocks_amount) * 1.03));
        CHECK(result.stock_allocation_on_total_portfolio_or_zero_if_no_wealth ==
              doctest::Approx(stocks_amount / 1100.0));
      }
    }
  } // namespace _test_apply_allocation

} // namespace run_common
