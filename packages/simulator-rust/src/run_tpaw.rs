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

    // Test
    // result.by_run_by_mfn_returns_stocks[month_run_index] = returns.stocks;
    // result.by_run_by_mfn_returns_bonds[month_run_index] = returns.bonds;

    // if let Some(x) = &params.test {
    //     if month_index % 12 == 0 {
    //         // let ours = &savings_portfolio_at_end.balance;
    //         let ours = &context.balance_starting;
    //         let truth = x.truth[month_index];
    //         let diff = ours - truth;
    //         web_sys::console::log_1(
    //             &format!(
    //                 "{:3} {:15.15} {:15.5} {:15.5}",
    //                 month_index, diff, ours, truth
    //             )
    //             .into(),
    //         );
    //     }
    // }
    // if run_index == 0 {
    // web_sys::console::log_1(
    //     &wasm_bindgen::JsValue::from_serde(&(
    //         &savings_portfolio_after_contributions,
    //         &savings_portfolio_after_withdrawals,
    //         &savings_portfolio_at_end,
    //     ))
    //     .unwrap(),
    // );
    // }

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
struct SingleMonthPreWithdrawalScale {
    withdrawals_discretionary: f64,
    legacy: f64,
}

#[derive(Serialize, Deserialize)]
struct SingleMonthPreWithdrawalPresentValueOfSpending {
    withdrawals_regular: f64,
    withdrawals_discretionary: f64,
    legacy: f64,
}

#[derive(Serialize, Deserialize)]
pub struct SingleMonthPreWithdrawal {
    wealth: f64,
    scale: SingleMonthPreWithdrawalScale,
    present_value_of_spending: SingleMonthPreWithdrawalPresentValueOfSpending,
    expected_returns_legacy_portfolio: f64,
}

// ---- PRE WITHDRAWAL ----
#[inline(always)]
fn calculate_pre_withdrawal(
    context: &SingleMonthContext,
    pre_withdrawal_from_using_expected_returns: &Option<&SingleMonthPreWithdrawal>,
) -> SingleMonthPreWithdrawal {
    let SingleMonthContext {
        params,
        pre_calculations,
        month_index,
        balance_starting,
        ..
    } = *context;

    let net_present_value = &pre_calculations.tpaw.net_present_value;

    // ---- WEALTH ----
    let wealth = balance_starting + net_present_value.savings.with_current_month[month_index];

    // ---- SCALE ----
    let scale = match pre_withdrawal_from_using_expected_returns {
        None => SingleMonthPreWithdrawalScale {
            legacy: 0.0,
            withdrawals_discretionary: 0.0,
        },
        Some(pre_withdrawal_from_using_expected_returns) => {
            let elasticity_of_wealth_wrt_stocks =
                if pre_withdrawal_from_using_expected_returns.wealth == 0.0 {
                    (params.target_allocation.legacy_portfolio
                        + params.target_allocation.regular_portfolio.tpaw[month_index]
                        + params.target_allocation.regular_portfolio.tpaw[month_index])
                        / 3.0
                } else {
                    (pre_withdrawal_from_using_expected_returns
                        .present_value_of_spending
                        .legacy
                        / pre_withdrawal_from_using_expected_returns.wealth)
                        * params.target_allocation.legacy_portfolio
                        + (pre_withdrawal_from_using_expected_returns
                            .present_value_of_spending
                            .withdrawals_discretionary
                            / pre_withdrawal_from_using_expected_returns.wealth)
                            * params.target_allocation.regular_portfolio.tpaw[month_index]
                        + (pre_withdrawal_from_using_expected_returns
                            .present_value_of_spending
                            .withdrawals_regular
                            / pre_withdrawal_from_using_expected_returns.wealth)
                            * params.target_allocation.regular_portfolio.tpaw[month_index]
                };

            let elasticity_of_extra_withdrawal_goals_wrt_wealth =
                if elasticity_of_wealth_wrt_stocks == 0.0 {
                    0.0
                } else {
                    params.target_allocation.regular_portfolio.tpaw[month_index]
                        / elasticity_of_wealth_wrt_stocks
                };

            let elasticity_of_legacy_goals_wrt_wealth = if elasticity_of_wealth_wrt_stocks == 0.0 {
                0.0
            } else {
                params.target_allocation.legacy_portfolio / elasticity_of_wealth_wrt_stocks
            };

            let percent_increase_in_wealth_over_scheduled =
                if pre_withdrawal_from_using_expected_returns.wealth == 0.0 {
                    0.0
                } else {
                    wealth / pre_withdrawal_from_using_expected_returns.wealth - 1.0
                };

            let legacy = f64::max(
                percent_increase_in_wealth_over_scheduled * elasticity_of_legacy_goals_wrt_wealth,
                -1.0,
            );

            let withdrawals_discretionary = f64::max(
                percent_increase_in_wealth_over_scheduled
                    * elasticity_of_extra_withdrawal_goals_wrt_wealth,
                -1.0,
            );
            SingleMonthPreWithdrawalScale {
                legacy,
                withdrawals_discretionary,
            }
        }
    };

    // ------ RETURNS -----
    let expected_returns_legacy_portfolio = params.expected_monthly_returns.stocks
        * params.target_allocation.legacy_portfolio
        + params.expected_monthly_returns.bonds * (1.0 - params.target_allocation.legacy_portfolio);

    // ---- PRESENT VALUE OF SPENDING ----
    let present_value_of_spending = {
        let mut account = AccountForWithdrawal::new(wealth);

        account.withdraw(net_present_value.withdrawals.lmp.with_current_month[month_index]);
        account.withdraw(net_present_value.withdrawals.essential.with_current_month[month_index]);
        let discretionary = account.withdraw(
            net_present_value
                .withdrawals
                .discretionary
                .with_current_month[month_index]
                * (1.0 + scale.withdrawals_discretionary),
        );
        let legacy = account.withdraw(
            (params.legacy_target * (1.0 + scale.legacy))
                / f64::powi(
                    1.0 + expected_returns_legacy_portfolio,
                    (params.num_months - month_index) as i32,
                ),
        );
        let regular = account.balance;

        SingleMonthPreWithdrawalPresentValueOfSpending {
            legacy,
            withdrawals_discretionary: discretionary,
            withdrawals_regular: regular,
        }
    };

    SingleMonthPreWithdrawal {
        wealth,
        present_value_of_spending,
        scale,
        expected_returns_legacy_portfolio,
    }
}

// ---- TARGET WITHDRAWAL ----
#[inline(always)]
fn calculate_target_withdrawals(
    context: &SingleMonthContext,
    _pre_withdrawal_from_using_expected_returns: &Option<&SingleMonthPreWithdrawal>,
    pre_withdrawal: &SingleMonthPreWithdrawal,
    _pass_forward: &SingleMonthPassForward,
) -> TargetWithdrawals {
    let SingleMonthContext {
        params,
        month_index,
        pre_calculations,
        ..
    } = *context;
    let SingleMonthPreWithdrawal {
        present_value_of_spending,
        scale,
        ..
    } = pre_withdrawal;

    let withdrawal_started = month_index >= params.withdrawal_start_month;

    let regular_without_lmp = if !withdrawal_started {
        0.0
    } else {
        present_value_of_spending.withdrawals_regular
            / pre_calculations.tpaw.cumulative_1_plus_g_over_1_plus_r[month_index]
    };

    // Should be 0 if withdrawal has not started.
    let lmp = params.lmp[month_index];
    let essential = params.by_month.withdrawals_essential[month_index];
    let discretionary = params.by_month.withdrawals_discretionary[month_index]
        * (1.0 + scale.withdrawals_discretionary);

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
    pre_withdrawal: &SingleMonthPreWithdrawal,
    savings_portfolio_after_withdrawals: &portfolio_over_month::AfterWithdrawals,
) -> f64 {
    let SingleMonthContext {
        params,
        pre_calculations,
        month_index,
        ..
    } = *context;
    let net_present_value = &pre_calculations.tpaw.net_present_value;
    let SingleMonthPreWithdrawal {
        scale,
        expected_returns_legacy_portfolio,
        ..
    } = pre_withdrawal;

    // If savings portfolio balance is 0, we approximate the limit as it goes to 0.
    let savings_portfolio_balance = f64::max(savings_portfolio_after_withdrawals.balance, 0.00001);
    let mut account = AccountForWithdrawal::new(
        savings_portfolio_balance + net_present_value.savings.without_current_month[month_index],
    );

    account.withdraw(net_present_value.withdrawals.lmp.without_current_month[month_index]);
    account.withdraw(
        net_present_value
            .withdrawals
            .essential
            .without_current_month[month_index],
    );

    let present_value_of_discretionary_withdrawals = account.withdraw(
        net_present_value
            .withdrawals
            .discretionary
            .without_current_month[month_index]
            * (1.0 + pre_withdrawal.scale.withdrawals_discretionary),
    );
    let present_value_of_desired_legacy = account.withdraw(
        (params.legacy_target * (1.0 + scale.legacy))
            / f64::powi(
                1.0 + expected_returns_legacy_portfolio,
                (params.num_months - month_index) as i32,
            ),
    );
    let present_value_of_regular_withdrawals = account.balance;

    let stocks_target = present_value_of_desired_legacy * params.target_allocation.legacy_portfolio
        + present_value_of_discretionary_withdrawals
            * params.target_allocation.regular_portfolio.tpaw[month_index]
        + present_value_of_regular_withdrawals
            * params.target_allocation.regular_portfolio.tpaw[month_index];

    // if month_index == 0 {
    //     web_sys::console::log_1(
    //         &wasm_bindgen::JsValue::from_serde(&(
    //             &stocks_target,
    //             &present_value_of_desired_legacy,
    //             &params.target_allocation.legacy_portfolio,
    //             &present_value_of_discretionary_withdrawals,
    //             &present_value_of_regular_withdrawals,
    //             &params.target_allocation.regular_portfolio.tpaw[month_index]
    //         ))
    //         .unwrap(),
    //     );
    // }

    return (stocks_target / savings_portfolio_balance).clamp(0.0, 1.0);
}

fn get_pass_forward(_withdrawal_target: TargetWithdrawals) -> SingleMonthPassForward {
    return SingleMonthPassForward {};
}
