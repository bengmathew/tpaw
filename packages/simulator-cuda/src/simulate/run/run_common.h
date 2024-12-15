#ifndef RUN_COMMON_H
#define RUN_COMMON_H

#include "src/public_headers/opt_currency.h"
#include "src/public_headers/stocks_and_bonds_float.h"
#include "src/simulate/run/run_result_padded.h"
#include "src/utils/account_for_withdrawal.h"
#include "src/utils/cuda_utils.h"
#include <cmath>
#include <cstdint>

namespace run_common {

  // ***************************************************************************
  // apply_withdrawal_ceiling_and_floor ****************************************
  // ***************************************************************************

  struct TargetWithdrawals {
    CURRENCY essential;
    CURRENCY discretionary;
    CURRENCY general;

    __host__ __device__ void print(uint32_t num_tabs) const;
  };

  __device__ __forceinline__ TargetWithdrawals
  apply_withdrawal_ceiling_and_floor(
      const TargetWithdrawals &target,
      const CURRENCY curr_month_discretionary_expense,
      const OptCURRENCY spending_ceiling,
      const OptCURRENCY spending_floor,
      const bool withdrawal_started,
      const bool debug = false) {

    CURRENCY general = target.general;
    CURRENCY discretionary = target.discretionary;

    if (spending_ceiling.is_set) {
      discretionary = fmin(discretionary, curr_month_discretionary_expense);
      general = fmin(general, spending_ceiling.opt_value);
    }

    if (spending_floor.is_set) {
      discretionary = fmax(discretionary, curr_month_discretionary_expense);
      if (withdrawal_started) {
        general = fmax(general, spending_floor.opt_value);
      }
    }

    return TargetWithdrawals{.essential = target.essential,
                             .discretionary = discretionary,
                             .general = general};
  }

  // ***************************************************************************
  // apply_contributions_and_withdrawals ***************************************
  // ***************************************************************************

  struct AfterContributionsAndWithdrawals {
    struct Withdrawals {
      CURRENCY essential;
      CURRENCY discretionary;
      CURRENCY general;
      CURRENCY total;
      FLOAT from_savings_portfolio_rate_or_nan_or_inf;

      __host__ __device__ void print(uint32_t num_tabs) const;
    };

    CURRENCY contributions;
    Withdrawals withdrawals;
    CURRENCY balance;
    bool insufficent_funds;

    __host__ __device__ void print(uint32_t num_tabs) const;
  };

  __device__ __forceinline__ AfterContributionsAndWithdrawals
  apply_contributions_and_withdrawals(
      const CURRENCY balance_starting,
      const CURRENCY contributions,
      const TargetWithdrawals &target_withdrawals,
      const bool debug = false) {

    AccountForWithdrawal account(balance_starting + contributions);
    const CURRENCY withdrawal_essential =
        account.withdraw(target_withdrawals.essential);
    const CURRENCY withdrawal_discretionary =
        account.withdraw(target_withdrawals.discretionary);
    const CURRENCY withdrawal_general =
        account.withdraw(target_withdrawals.general);
    const CURRENCY withdraw_total =
        withdrawal_essential + withdrawal_discretionary + withdrawal_general;

    const CURRENCY from_contributions = fmin(withdraw_total, contributions);
    const CURRENCY from_savings_portfolio = withdraw_total - from_contributions;
    // 0/0 is nan, but !0/0 is Inf.
    const auto from_savings_portfolio_rate_or_nan_or_inf =
        static_cast<FLOAT>(from_savings_portfolio / balance_starting);

    return {
        .contributions = contributions,
        .withdrawals =
            AfterContributionsAndWithdrawals::Withdrawals{
                .essential = withdrawal_essential,
                .discretionary = withdrawal_discretionary,
                .general = withdrawal_general,
                .total = withdraw_total,
                .from_savings_portfolio_rate_or_nan_or_inf =
                    from_savings_portfolio_rate_or_nan_or_inf},
        .balance = account.balance,
        .insufficent_funds = account.insufficient_funds,
    };
  }

  // ***************************************************************************
  // apply_allocation **********************************************************
  // ***************************************************************************

  struct End {
    FLOAT stock_allocation_percent_before_returns;
    CURRENCY stock_allocation_amount_before_returns;
    CURRENCY balance;
    FLOAT stock_allocation_on_total_portfolio_or_zero_if_no_wealth;

    __host__ __device__ void print(uint32_t num_tabs) const;
  };

  __device__ __forceinline__ End
  apply_allocation(FLOAT stock_allocation_percent,
                   const StocksAndBondsFLOAT &return_rate,
                   const CURRENCY balance_after_contributions_and_withdrawals,
                   const CURRENCY npv_income_without_current_month) {
    const CURRENCY stock_allocation_amount{
        balance_after_contributions_and_withdrawals * stock_allocation_percent};
    const CURRENCY bonds_allocation_amount{
        balance_after_contributions_and_withdrawals - stock_allocation_amount};
    const CURRENCY stock_allocation_amount_after_return{
        stock_allocation_amount *
        static_cast<CURRENCY>(FLOAT_L(1.0) + return_rate.stocks)};

    const CURRENCY balance{
        fma(static_cast<CURRENCY>(FLOAT_L(1.0) + return_rate.bonds),
            bonds_allocation_amount,
            stock_allocation_amount_after_return)};

    const auto stock_allocation_on_total_portfolio = static_cast<FLOAT>(
        stock_allocation_amount / (balance_after_contributions_and_withdrawals +
                                   npv_income_without_current_month));
    return {.stock_allocation_percent_before_returns = stock_allocation_percent,
            .stock_allocation_amount_before_returns = stock_allocation_amount,
            .balance = balance,
            .stock_allocation_on_total_portfolio_or_zero_if_no_wealth =
                isfinite(stock_allocation_on_total_portfolio)
                    ? stock_allocation_on_total_portfolio
                    : FLOAT_L(0.0)};
  }

  // ***************************************************************************
  // write_result **************************************************************
  // ***************************************************************************

  __device__ __forceinline__ void
  write_result(const CURRENCY balance_starting,
               const run_common::AfterContributionsAndWithdrawals
                   &savings_portfolio_after_contributions_and_withdrawals,
               const FLOAT withdrawals_from_savings_portfolio_rate,
               const run_common::End &savings_portfolio_at_end,
               const FLOAT spending_tilt,
               RunResultPadded_GPU &normal_result,
               const uint32_t run_index,
               const uint32_t month_index) {
    const uint32_t by_run_by_mfn_simulated_month_major_index =
        get_run_by_mfn_month_major_index(
            normal_result.num_runs_padded, run_index, month_index);
    normal_result.by_run_by_mfn_simulated_month_major_balance_start
        [by_run_by_mfn_simulated_month_major_index] = balance_starting;
    normal_result.by_run_by_mfn_simulated_month_major_withdrawals_essential
        [by_run_by_mfn_simulated_month_major_index] =
        savings_portfolio_after_contributions_and_withdrawals.withdrawals
            .essential;
    normal_result.by_run_by_mfn_simulated_month_major_withdrawals_discretionary
        [by_run_by_mfn_simulated_month_major_index] =
        savings_portfolio_after_contributions_and_withdrawals.withdrawals
            .discretionary;
    normal_result.by_run_by_mfn_simulated_month_major_withdrawals_general
        [by_run_by_mfn_simulated_month_major_index] =
        savings_portfolio_after_contributions_and_withdrawals.withdrawals
            .general;
    normal_result.by_run_by_mfn_simulated_month_major_withdrawals_total
        [by_run_by_mfn_simulated_month_major_index] =
        savings_portfolio_after_contributions_and_withdrawals.withdrawals.total;
    normal_result
        .by_run_by_mfn_simulated_month_major_withdrawals_from_savings_portfolio_rate
            [by_run_by_mfn_simulated_month_major_index] =
        withdrawals_from_savings_portfolio_rate;
    normal_result
        .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_savings_portfolio
            [by_run_by_mfn_simulated_month_major_index] =
        savings_portfolio_at_end.stock_allocation_percent_before_returns;
    normal_result
        .by_run_by_mfn_simulated_month_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth
            [by_run_by_mfn_simulated_month_major_index] =
        savings_portfolio_at_end
            .stock_allocation_on_total_portfolio_or_zero_if_no_wealth;
    normal_result.tpaw_by_run_by_mfn_simulated_month_major_spending_tilt
        [by_run_by_mfn_simulated_month_major_index] = spending_tilt;

    if (savings_portfolio_after_contributions_and_withdrawals
            .insufficent_funds) {
      normal_result.by_run_num_insufficient_fund_months[run_index] += 1;
    }
    // We could save some writes by doing this only for the last month,
    // (everything lese is overwritten), but it's not very expensive, and
    // it helps with testing _single_month().
    normal_result.by_run_ending_balance[run_index] =
        savings_portfolio_at_end.balance;
  }

} // namespace run_common

#endif // RUN_COMMON_H
