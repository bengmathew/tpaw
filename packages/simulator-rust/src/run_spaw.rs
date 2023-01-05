use std::cmp::Ordering;

use crate::params::*;
use crate::portfolio_over_year;
use crate::portfolio_over_year::SingleYearContext;
use crate::portfolio_over_year::TargetWithdrawals;
use crate::portfolio_over_year::Withdrawals;
use crate::pre_calculations::*;
use crate::utils::*;
use crate::RunResult;
use serde::Deserialize;
use serde::Serialize;

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
) -> Vec<(SingleYearPreWithdrawal, Withdrawals)> {
    let mut balance_starting = params.current_savings;
    let mut pass_forward = initial_pass_forward();
    return (0..params.num_years)
        .map(|year_index| {
            let context = SingleYearContext {
                params,
                pre_calculations,
                year_index,
                returns: &params.expected_returns,
                balance_starting,
            };
            let (
                pre_withdrawal,
                _savings_portfolio_after_contributions,
                savings_portfolio_after_withdrawals,
                savings_portfolio_at_end,
                curr_pass_forward,
            ) = run_for_single_year_using_fixed_returns(&context, &pass_forward);

            // if let Some(x) = &params.test {
            //     let ours = &savings_portfolio_at_end.balance;
            //     let truth = x.truth[year_index];
            //     let diff = ours - truth;
            //     web_sys::console::log_1(
            //         &format!(
            //             "{:3} {:15.2} {:15.2} {:15.2}",
            //             year_index, diff, ours, truth
            //         )
            //         .into(),
            //     );
            //     if year_index == 30 {
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
    pre_withdrawal_from_using_expected_returns: &Vec<SingleYearPreWithdrawal>,
    result: &mut RunResult,
    run_index: usize,
) {
    let n = params.num_years;

    let historical_index = ((params.start_run + run_index)
        ..(params.start_run + run_index + params.num_years))
        .collect();
    let index_into_historical_returns = if let Some(x) = &params.test {
        &x.index_into_historical_returns
    } else {
        if params.monte_carlo_sampling {
            memoized_random(
                params.end_run - params.start_run,
                params.num_years,
                params.historical_returns.len(),
                run_index,
            )
        } else {
            &historical_index
        }
    };
    // web_sys::console::log_1(&wasm_bindgen::JsValue::from_serde(index_into_historical_returns).unwrap());

    let mut balance_starting = params.current_savings;
    let mut pass_forward = initial_pass_forward();
    for year_index in 0..n {
        let context = SingleYearContext {
            params,
            pre_calculations,
            year_index,
            returns: &params.historical_returns[index_into_historical_returns[year_index]],
            balance_starting,
        };
        let (new_balance, curr_pass_forward) = run_for_single_year_using_historical_returns(
            &context,
            &pass_forward,
            &pre_withdrawal_from_using_expected_returns[year_index],
            result,
            run_index,
        );
        balance_starting = new_balance;
        pass_forward = curr_pass_forward
    }
    result.by_run_ending_balance[run_index] = balance_starting;
}

#[inline(always)]
fn run_for_single_year_using_fixed_returns(
    context: &SingleYearContext,
    pass_forward: &SingleYearPassForward,
) -> (
    SingleYearPreWithdrawal,
    portfolio_over_year::AfterContributions,
    portfolio_over_year::AfterWithdrawals,
    portfolio_over_year::End,
    SingleYearPassForward,
) {
    let SingleYearContext {
        params,
        year_index,
        returns,
        balance_starting,
        ..
    } = *context;
    let savings_portfolio_after_contributions = portfolio_over_year::apply_contributions(
        params.by_year.savings[year_index],
        balance_starting,
    );

    let pre_withdrawal = calculate_pre_withdrawal(context, &None);

    let target_withdrawals =
        calculate_target_withdrawals(context, &None, &pre_withdrawal, pass_forward);

    let savings_portfolio_after_withdrawals = portfolio_over_year::apply_target_withdrawals(
        &target_withdrawals,
        context,
        &savings_portfolio_after_contributions,
    );

    let stock_allocation = calculate_stock_allocation(
        &context,
        &pre_withdrawal,
        &savings_portfolio_after_withdrawals,
    );

    let savings_portfolio_at_end = portfolio_over_year::apply_allocation(
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
fn run_for_single_year_using_historical_returns(
    context: &SingleYearContext,
    pass_forward: &SingleYearPassForward,
    pre_withdrawal_from_using_expected_returns: &SingleYearPreWithdrawal,
    result: &mut RunResult,
    run_index: usize,
) -> (f64, SingleYearPassForward) {
    let SingleYearContext {
        params,
        pre_calculations,
        year_index,
        returns,
        balance_starting,
        ..
    } = *context;
    let savings_portfolio_after_contributions = portfolio_over_year::apply_contributions(
        params.by_year.savings[year_index],
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

    let savings_portfolio_after_withdrawals = portfolio_over_year::apply_target_withdrawals(
        &target_withdrawals,
        context,
        &savings_portfolio_after_contributions,
    );

    let stock_allocation = calculate_stock_allocation(
        &context,
        &pre_withdrawal,
        &savings_portfolio_after_withdrawals,
    );

    let savings_portfolio_at_end = portfolio_over_year::apply_allocation(
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
                    .without_current_year[year_index]);
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
            let context2 = SingleYearContext {
                params: context.params,
                pre_calculations: context.pre_calculations,
                year_index: context.year_index,
                returns: context.returns,
                balance_starting: 1.0,
            };
            let savings_portfolio_after_contributions = portfolio_over_year::apply_contributions(
                params.by_year.savings[year_index],
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

            let savings_portfolio_after_withdrawals = portfolio_over_year::apply_target_withdrawals(
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

    let year_run_index = (year_index * (params.end_run - params.start_run)) + run_index;
    result.by_yfn_by_run_balance_start[year_run_index] = balance_starting;
    result.by_yfn_by_run_withdrawals_essential[year_run_index] =
        savings_portfolio_after_withdrawals.withdrawals.essential;
    result.by_yfn_by_run_withdrawals_discretionary[year_run_index] =
        savings_portfolio_after_withdrawals
            .withdrawals
            .discretionary;
    result.by_yfn_by_run_withdrawals_regular[year_run_index] =
        savings_portfolio_after_withdrawals.withdrawals.regular;
    result.by_yfn_by_run_withdrawals_total[year_run_index] =
        savings_portfolio_after_withdrawals.withdrawals.total;
    result.by_yfn_by_run_withdrawals_from_savings_portfolio_rate[year_run_index] =
        withdrawals_from_savings_portfolio_rate;
    result.by_yfn_by_run_after_withdrawals_allocation_stocks_savings[year_run_index] =
        savings_portfolio_at_end.stock_allocation_percent;
    result.by_yfn_by_run_after_withdrawals_allocation_stocks_total[year_run_index] =
        stock_allocation_on_total_portfolio;

    if savings_portfolio_after_withdrawals.insufficient_funds {
        result.by_run_num_insufficient_fund_years[run_index] =
            result.by_run_num_insufficient_fund_years[run_index] + 1;
    }

    // Test
    // result.by_yfn_by_run_returns_stocks[year_run_index] = returns.stocks;
    // result.by_yfn_by_run_returns_bonds[year_run_index] = returns.bonds;

    // if let Some(x) = &params.test {
    //     let ours = &savings_portfolio_at_end.balance;
    //     let truth = x.truth[year_index];
    //     let diff = ours - truth;
    //     web_sys::console::log_1(
    //         &format!(
    //             "{:3} {:15.2} {:15.2} {:15.2}",
    //             year_index, diff, ours, truth
    //         )
    //         .into(),
    //     );
    //     if year_index == 0 {
    //         web_sys::console::log_1(
    //             &wasm_bindgen::JsValue::from_serde(&(
    //                 &savings_portfolio_after_contributions,
    //                 &savings_portfolio_after_withdrawals,
    //                 &savings_portfolio_at_end,
    //                 &returns
    //             ))
    //             .unwrap(),
    //         );
    //     }
    // }

    let curr_pass_forward = get_pass_forward(target_withdrawals);
    return (savings_portfolio_at_end.balance, curr_pass_forward);
}

// -----------------------------------------------
// --------------------- ACTUAL ------------------
// -----------------------------------------------

struct SingleYearPassForward {}

fn initial_pass_forward() -> SingleYearPassForward {
    return SingleYearPassForward {};
}

#[derive(Serialize, Deserialize)]
pub struct SingleYearPreWithdrawal {
    wealth_less_essential_and_lmp_expenses: f64,
}

// ---- PRE WITHDRAWAL ----

#[inline(always)]
fn calculate_pre_withdrawal(
    context: &SingleYearContext,
    _pre_withdrawal_from_using_expected_returns: &Option<&SingleYearPreWithdrawal>,
) -> SingleYearPreWithdrawal {
    let SingleYearContext {
        pre_calculations,
        year_index,
        ..
    } = *context;

    let net_present_value = &pre_calculations.spaw.net_present_value;

    let wealth_less_essential_and_lmp_expenses = context.balance_starting
        + net_present_value.savings.with_current_year[year_index]
        - net_present_value.withdrawals.essential.with_current_year[year_index]
        - net_present_value.withdrawals.lmp.with_current_year[year_index];

    SingleYearPreWithdrawal {
        wealth_less_essential_and_lmp_expenses,
    }
}

// ---- TARGET WITHDRAWAL ----
#[inline(always)]
fn calculate_target_withdrawals(
    context: &SingleYearContext,
    pre_withdrawal_from_using_expected_returns: &Option<&SingleYearPreWithdrawal>,
    pre_withdrawal: &SingleYearPreWithdrawal,
    _pass_forward: &SingleYearPassForward,
) -> TargetWithdrawals {
    let SingleYearContext {
        params,
        pre_calculations,
        year_index,
        ..
    } = *context;

    let withdrawal_started = year_index >= params.withdrawal_start_year;
    let PreCalculationsForSPAW {
        net_present_value,
        cumulative_1_plus_g_over_1_plus_r,
    } = &pre_calculations.spaw;

    let SingleYearPreWithdrawal {
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
            (context.balance_starting + net_present_value.savings.with_current_year[year_index]
                - net_present_value.withdrawals.lmp.with_current_year[year_index]
                - net_present_value.withdrawals.essential.with_current_year[year_index]
                - net_present_value
                    .withdrawals
                    .discretionary
                    .with_current_year[year_index]
                    * scale
                - net_present_value.legacy.with_current_year[year_index] * scale)
                / cumulative_1_plus_g_over_1_plus_r[year_index],
            0.0,
        )
    };

    // Should be 0 if withdrawal has not started.
    let lmp = params.lmp[year_index];
    let essential = params.by_year.withdrawals_essential[year_index];
    let discretionary = params.by_year.withdrawals_discretionary[year_index] * scale;
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
    context: &SingleYearContext,
    _pre_withdrawal: &SingleYearPreWithdrawal,
    _savings_portfolio_after_withdrawals: &portfolio_over_year::AfterWithdrawals,
) -> f64 {
    context
        .params
        .target_allocation
        .regular_portfolio
        .spaw_and_swr[context.year_index]
}

fn get_pass_forward(_withdrawal_target: TargetWithdrawals) -> SingleYearPassForward {
    return SingleYearPassForward {};
}
