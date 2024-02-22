#[allow(unused_imports)]
use std::cmp::Ordering;

use crate::params::*;
use crate::portfolio_over_month;
use crate::portfolio_over_month::SingleMonthContext;
use crate::portfolio_over_month::TargetWithdrawals;
use crate::portfolio_over_month::Withdrawals;
use crate::pre_calculations::*;
use crate::utils::*;
use crate::RunResult;
use serde::Deserialize;
use serde::Serialize;

use self::random::memoized_random;

pub fn run(params: &Params, result: &mut RunResult) {
    let pre_calculations = do_pre_calculations(&params);

    let pre_withdrawal_from_using_expected_returns =
        run_using_expected_returns(&params, &pre_calculations)
            .into_iter()
            .map(|(pre_withdrawal, _)| pre_withdrawal)
            .collect();

    let num_runs = params.end_run - params.start_run;
    for run_index in 0..num_runs {
        run_using_historical_returns(
            &params,
            &pre_calculations,
            &pre_withdrawal_from_using_expected_returns,
            result,
            run_index,
        )
    }
}

pub fn run_using_expected_returns(
    params: &Params,
    pre_calculations: &PreCalculations,
) -> Vec<(SingleMonthPreWithdrawal, Withdrawals)> {
    let mut balance_starting = params.current_savings;
    let mut pass_forward = initial_pass_forward();
    return (0..params.num_months_to_simulate)
        .map(|month_index| {
            let context = SingleMonthContext {
                params,
                pre_calculations,
                month_index,
                returns: &&params.expected_monthly_returns,
                balance_starting,
            };
            let (
                pre_withdrawal,
                _savings_portfolio_after_contributions,
                savings_portfolio_after_withdrawals,
                savings_portfolio_at_end,
                curr_pass_forward,
            ) = run_for_single_month_using_fixed_returns(&context, &pass_forward);

            // if let Some(x) = &params.test {
            //     let ours = &savings_portfolio_at_end.balance;
            //     let truth = x.truth[month_index];
            //     let diff = ours - truth;
            //     web_sys::console::log_1(
            //         &format!(
            //             "{:3} {:15.2} {:15.2} {:15.2}",
            //             month_index, diff, ours, truth
            //         )
            //         .into(),
            //     );
            //     if month_index == 30 {
            //         web_sys::console::log_1(
            //             &wasm_bindgen::JsValue::from_serde(&(
            //                 &savings_portfolio_after_contributions,
            //                 &savings_portfolio_after_withdrawals,
            //                 &savings_portfolio_at_end,
            //             ))
            //             .unwrap(),
            //         );
            //     }
            // }

            balance_starting = savings_portfolio_at_end.balance;
            pass_forward = curr_pass_forward;
            return (
                pre_withdrawal,
                savings_portfolio_after_withdrawals.withdrawals,
            );
        })
        .collect();
}

fn run_using_historical_returns(
    params: &Params,
    pre_calculations: &PreCalculations,
    pre_withdrawal_from_using_expected_returns: &Vec<SingleMonthPreWithdrawal>,
    result: &mut RunResult,
    run_index: usize,
) {
    let n = params.num_months_to_simulate;

    let historical_index = ((params.start_run + run_index)
        ..(params.start_run + run_index + params.num_months))
        .collect();
    let index_into_historical_returns = if let Some(x) = &params.test {
        &x.index_into_historical_returns
    } else {
        if params.monte_carlo_sampling {
            &memoized_random(
                params.rand_seed,
                params.start_run,
                params.end_run - params.start_run,
                params.max_num_months,
                params.monte_carlo_block_size,
                params.historical_returns.len(),
                params.monte_carlo_stagger_run_starts,
            )[run_index]
        } else {
            &historical_index
        }
    };

    // web_sys::console::log_1(
    //     &wasm_bindgen::JsValue::from_serde(&(
    //         &index_into_historical_returns,
    //         &params.historical_returns
    //     ))
    //     .unwrap(),
    // );

    let mut balance_starting = params.current_savings;
    let mut pass_forward = initial_pass_forward();
    for month_index in 0..n {
        let context = SingleMonthContext {
            params,
            pre_calculations,
            month_index,
            returns: &params.historical_returns[index_into_historical_returns[month_index]],
            balance_starting,
        };
        let (new_balance, curr_pass_forward) = run_for_single_month_using_historical_returns(
            &context,
            &pass_forward,
            &pre_withdrawal_from_using_expected_returns[month_index],
            result,
            run_index,
        );
        balance_starting = new_balance;
        pass_forward = curr_pass_forward
    }
    result.by_run_ending_balance[run_index] = balance_starting;
}

#[inline(always)]
fn run_for_single_month_using_fixed_returns(
    context: &SingleMonthContext,
    pass_forward: &SingleMonthPassForward,
) -> (
    SingleMonthPreWithdrawal,
    portfolio_over_month::AfterContributions,
    portfolio_over_month::AfterWithdrawals,
    portfolio_over_month::End,
    SingleMonthPassForward,
) {
    let SingleMonthContext {
        params,
        month_index,
        returns,
        balance_starting,
        ..
    } = *context;
    let savings_portfolio_after_contributions = portfolio_over_month::apply_contributions(
        params.by_month.savings[month_index],
        balance_starting,
    );

    let pre_withdrawal = calculate_pre_withdrawal(context, &None);

    let target_withdrawals =
        calculate_target_withdrawals(context, &None, &pre_withdrawal, pass_forward);

    let savings_portfolio_after_withdrawals = portfolio_over_month::apply_target_withdrawals(
        &target_withdrawals,
        context,
        &savings_portfolio_after_contributions,
    );

    let stock_allocation = calculate_stock_allocation(
        &context,
        &pre_withdrawal,
        &savings_portfolio_after_withdrawals,
    );

    let savings_portfolio_at_end = portfolio_over_month::apply_allocation(
        stock_allocation,
        &returns,
        &savings_portfolio_after_withdrawals,
    );

    let curr_pass_forward = get_pass_forward(target_withdrawals);

    return (
        pre_withdrawal,
        savings_portfolio_after_contributions,
        savings_portfolio_after_withdrawals,
        savings_portfolio_at_end,
        curr_pass_forward,
    );
}

#[inline(always)]
fn run_for_single_month_using_historical_returns(
    context: &SingleMonthContext,
    pass_forward: &SingleMonthPassForward,
    pre_withdrawal_from_using_expected_returns: &SingleMonthPreWithdrawal,
    result: &mut RunResult,
    run_index: usize,
) -> (f64, SingleMonthPassForward) {
    let SingleMonthContext {
        params,
        pre_calculations,
        month_index,
        returns,
        balance_starting,
        ..
    } = *context;
    let savings_portfolio_after_contributions = portfolio_over_month::apply_contributions(
        params.by_month.savings[month_index],
        balance_starting,
    );

    let pre_withdrawal =
        calculate_pre_withdrawal(context, &Some(pre_withdrawal_from_using_expected_returns));

    let target_withdrawals = calculate_target_withdrawals(
        context,
        &Some(pre_withdrawal_from_using_expected_returns),
        &pre_withdrawal,
        pass_forward,
    );

    let savings_portfolio_after_withdrawals = portfolio_over_month::apply_target_withdrawals(
        &target_withdrawals,
        context,
        &savings_portfolio_after_contributions,
    );

    let stock_allocation = calculate_stock_allocation(
        &context,
        &pre_withdrawal,
        &savings_portfolio_after_withdrawals,
    );

    let savings_portfolio_at_end = portfolio_over_month::apply_allocation(
        stock_allocation,
        &returns,
        &savings_portfolio_after_withdrawals,
    );

    let stock_allocation_on_total_portfolio = {
        let x = savings_portfolio_at_end.stock_allocation_amount
            / (savings_portfolio_after_withdrawals.balance
                + pre_calculations
                    .tpaw
                    .net_present_value
                    .savings
                    .without_current_month[month_index]);
        if f64::is_nan(x) || f64::is_infinite(x) {
            0.0
        } else {
            x
        }
    };

    let withdrawals_from_savings_portfolio_rate = {
        let x = savings_portfolio_after_withdrawals
            .withdrawals
            .from_savings_portfolio_rate_or_nan;
        if x.is_nan() {
            let context2 = SingleMonthContext {
                params: context.params,
                pre_calculations: context.pre_calculations,
                month_index: context.month_index,
                returns: context.returns,
                balance_starting: 0.00001,
            };
            let savings_portfolio_after_contributions = portfolio_over_month::apply_contributions(
                params.by_month.savings[month_index],
                balance_starting,
            );

            let pre_withdrawal = calculate_pre_withdrawal(
                &context2,
                &Some(pre_withdrawal_from_using_expected_returns),
            );

            let target_withdrawals = calculate_target_withdrawals(
                &context2,
                &Some(pre_withdrawal_from_using_expected_returns),
                &pre_withdrawal,
                pass_forward,
            );

            let savings_portfolio_after_withdrawals =
                portfolio_over_month::apply_target_withdrawals(
                    &target_withdrawals,
                    &context2,
                    &savings_portfolio_after_contributions,
                );
            savings_portfolio_after_withdrawals
                .withdrawals
                .from_savings_portfolio_rate_or_nan
        } else {
            x
        }
    };

    let by_mfn_by_run_index = (month_index * (params.end_run - params.start_run)) + run_index;
    result.by_mfn_by_run_balance_start[by_mfn_by_run_index] = balance_starting;
    result.by_mfn_by_run_withdrawals_essential[by_mfn_by_run_index] =
        savings_portfolio_after_withdrawals.withdrawals.essential;
    result.by_mfn_by_run_withdrawals_discretionary[by_mfn_by_run_index] =
        savings_portfolio_after_withdrawals
            .withdrawals
            .discretionary;
    result.by_mfn_by_run_withdrawals_regular[by_mfn_by_run_index] =
        savings_portfolio_after_withdrawals.withdrawals.regular;
    result.by_mfn_by_run_withdrawals_total[by_mfn_by_run_index] =
        savings_portfolio_after_withdrawals.withdrawals.total;
    result.by_mfn_by_run_withdrawals_from_savings_portfolio_rate[by_mfn_by_run_index] =
        withdrawals_from_savings_portfolio_rate;
    result.by_mfn_by_run_after_withdrawals_allocation_stocks_savings[by_mfn_by_run_index] =
        savings_portfolio_at_end.stock_allocation_percent;
    result.by_mfn_by_run_after_withdrawals_allocation_stocks_total[by_mfn_by_run_index] =
        stock_allocation_on_total_portfolio;
    let by_run_by_mfn_index = run_index * params.num_months_to_simulate + month_index;
    result.by_run_by_mfn_returns_stocks[by_run_by_mfn_index] = context.returns.stocks;
    result.by_run_by_mfn_returns_bonds[by_run_by_mfn_index] = context.returns.bonds;

    if savings_portfolio_after_withdrawals.insufficient_funds {
        result.by_run_num_insufficient_fund_months[run_index] =
            result.by_run_num_insufficient_fund_months[run_index] + 1;
    }

    let curr_pass_forward = get_pass_forward(target_withdrawals);
    return (savings_portfolio_at_end.balance, curr_pass_forward);
}

// -----------------------------------------------
// --------------------- ACTUAL ------------------
// -----------------------------------------------

struct SingleMonthPassForward {}

fn initial_pass_forward() -> SingleMonthPassForward {
    return SingleMonthPassForward {};
}

#[derive(Serialize, Deserialize)]
pub struct SingleMonthPreWithdrawal {
    wealth_less_essential_and_lmp_expenses: f64,
}

// ---- PRE WITHDRAWAL ----

#[inline(always)]
fn calculate_pre_withdrawal(
    context: &SingleMonthContext,
    _pre_withdrawal_from_using_expected_returns: &Option<&SingleMonthPreWithdrawal>,
) -> SingleMonthPreWithdrawal {
    let SingleMonthContext {
        pre_calculations,
        month_index,
        ..
    } = *context;

    let net_present_value = &pre_calculations.spaw.net_present_value;

    let wealth_less_essential_and_lmp_expenses = context.balance_starting
        + net_present_value.savings.with_current_month[month_index]
        - net_present_value.withdrawals.essential.with_current_month[month_index]
        - net_present_value.withdrawals.lmp.with_current_month[month_index];

    SingleMonthPreWithdrawal {
        wealth_less_essential_and_lmp_expenses,
    }
}

// ---- TARGET WITHDRAWAL ----
#[inline(always)]
fn calculate_target_withdrawals(
    context: &SingleMonthContext,
    pre_withdrawal_from_using_expected_returns: &Option<&SingleMonthPreWithdrawal>,
    pre_withdrawal: &SingleMonthPreWithdrawal,
    _pass_forward: &SingleMonthPassForward,
) -> TargetWithdrawals {
    let SingleMonthContext {
        params,
        pre_calculations,
        month_index,
        ..
    } = *context;

    let withdrawal_started = month_index >= params.withdrawal_start_month;
    let PreCalculationsForSPAW {
        net_present_value,
        cumulative_1_plus_g_over_1_plus_r,
    } = &pre_calculations.spaw;

    let SingleMonthPreWithdrawal {
        wealth_less_essential_and_lmp_expenses,
    } = pre_withdrawal;

    let scale = match pre_withdrawal_from_using_expected_returns {
        None => 1.0,
        Some(results_from_using_expected_returns) => {
            wealth_less_essential_and_lmp_expenses
                / results_from_using_expected_returns.wealth_less_essential_and_lmp_expenses
        }
    };

    let regular_without_lmp = if !withdrawal_started {
        0.0
    } else {
        f64::max(
            (context.balance_starting + net_present_value.savings.with_current_month[month_index]
                - net_present_value.withdrawals.lmp.with_current_month[month_index]
                - net_present_value.withdrawals.essential.with_current_month[month_index]
                - net_present_value
                    .withdrawals
                    .discretionary
                    .with_current_month[month_index]
                    * scale
                - net_present_value.legacy.with_current_month[month_index] * scale)
                / cumulative_1_plus_g_over_1_plus_r[month_index],
            0.0,
        )
    };

    // Should be 0 if withdrawal has not started.
    let lmp = params.lmp[month_index];
    let essential = params.by_month.withdrawals_essential[month_index];
    let discretionary = params.by_month.withdrawals_discretionary[month_index] * scale;
    TargetWithdrawals {
        lmp,
        essential,
        discretionary,
        regular_without_lmp,
    }
}

// ---- STOCK ALLOCATION ----
#[inline(always)]
fn calculate_stock_allocation(
    context: &SingleMonthContext,
    _pre_withdrawal: &SingleMonthPreWithdrawal,
    _savings_portfolio_after_withdrawals: &portfolio_over_month::AfterWithdrawals,
) -> f64 {
    context
        .params
        .target_allocation
        .regular_portfolio
        .spaw_and_swr[context.month_index]
}

fn get_pass_forward(_withdrawal_target: TargetWithdrawals) -> SingleMonthPassForward {
    return SingleMonthPassForward {};
}
