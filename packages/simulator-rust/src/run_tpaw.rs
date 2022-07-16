use std::cmp::Ordering;

use crate::params::*;
use crate::portfolio_over_year;
use crate::portfolio_over_year::SingleYearContext;
use crate::portfolio_over_year::TargetWithdrawals;
use crate::portfolio_over_year::Withdrawals;
use crate::pre_calculations::*;
use crate::run_tpaw;
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

    let mut params_for_bond_returns = params.clone();
    params_for_bond_returns
        .target_allocation
        .regular_portfolio
        .tpaw = 0.0;
    params_for_bond_returns
        .target_allocation
        .regular_portfolio
        .spaw_and_swr = vec![0.0; params.num_years].into_boxed_slice();
    params_for_bond_returns.expected_returns.stocks = 0.0;
    params_for_bond_returns.expected_returns.bonds = params.expected_returns.bonds;
    let withdrawals_from_using_bond_returns: Vec<Withdrawals> =
        run_tpaw::run_using_expected_returns(
            &params_for_bond_returns,
            &do_pre_calculations(&params_for_bond_returns),
        )
        .into_iter()
        .map(|(_, withdrawals)| withdrawals)
        .collect();

    let num_runs = params.end_run - params.start_run;
    for run_index in 0..num_runs {
        run_using_historical_returns(
            &params,
            &pre_calculations,
            &pre_withdrawal_from_using_expected_returns,
            &withdrawals_from_using_bond_returns,
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
                savings_portfolio_after_withdrawals,
                savings_portfolio_at_end,
                curr_pass_forward,
            ) = run_for_single_year_using_fixed_returns(&context, &pass_forward);
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
    withdrawal_from_using_bond_returns: &Vec<Withdrawals>,
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
            &withdrawal_from_using_bond_returns[year_index],
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

    let target_withdrawals = calculate_target_withdrawals(
        context,
        balance_starting,
        &None,
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

    let curr_pass_forward = get_pass_forward(target_withdrawals);

    (
        pre_withdrawal,
        savings_portfolio_after_withdrawals,
        savings_portfolio_at_end,
        curr_pass_forward,
    )
}

#[inline(always)]
fn run_for_single_year_using_historical_returns(
    context: &SingleYearContext,
    pass_forward: &SingleYearPassForward,
    pre_withdrawal_from_using_expected_returns: &SingleYearPreWithdrawal,
    withdrawal_from_using_bond_returns: &Withdrawals,
    result: &mut RunResult,
    run_index: usize,
) -> (f64, SingleYearPassForward) {
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

    let pre_withdrawal =
        calculate_pre_withdrawal(context, &Some(pre_withdrawal_from_using_expected_returns));

    let target_withdrawals = calculate_target_withdrawals(
        context,
        balance_starting,
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
        savings_portfolio_after_withdrawals
            .withdrawals
            .from_savings_portfolio_rate;
    result.by_yfn_by_run_excess_withdrawals_regular[year_run_index] =
        savings_portfolio_after_withdrawals.withdrawals.regular
            - withdrawal_from_using_bond_returns.regular;
    result.by_yfn_by_run_after_withdrawals_allocation_stocks[year_run_index] =
        savings_portfolio_at_end.stock_allocation;
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
    // web_sys::console::log_1(
    //     &wasm_bindgen::JsValue::from_serde(&(
    //         &savings_portfolio_after_contributions,
    //         &savings_portfolio_after_withdrawals,
    //         &savings_portfolio_at_end,
    //     ))
    //     .unwrap(),
    // );
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
struct SingleYearPreWithdrawalScale {
    withdrawals_discretionary: f64,
    legacy: f64,
}

#[derive(Serialize, Deserialize)]
struct SingleYearPreWithdrawalPresentValueOfSpending {
    withdrawals_regular: f64,
    withdrawals_discretionary: f64,
    legacy: f64,
}

#[derive(Serialize, Deserialize)]
pub struct SingleYearPreWithdrawal {
    wealth: f64,
    scale: SingleYearPreWithdrawalScale,
    present_value_of_spending: SingleYearPreWithdrawalPresentValueOfSpending,
    expected_returns_legacy_portfolio: f64,
}

// ---- PRE WITHDRAWAL ----
#[inline(always)]
fn calculate_pre_withdrawal(
    context: &SingleYearContext,
    pre_withdrawal_from_using_expected_returns: &Option<&SingleYearPreWithdrawal>,
) -> SingleYearPreWithdrawal {
    let SingleYearContext {
        params,
        pre_calculations,
        year_index,
        balance_starting,
        ..
    } = *context;

    let net_present_value = &pre_calculations.tpaw.net_present_value;

    // ---- WEALTH ----
    let wealth = balance_starting + net_present_value.savings.with_current_year[year_index];

    // ---- SCALE ----
    let scale = match pre_withdrawal_from_using_expected_returns {
        None => SingleYearPreWithdrawalScale {
            legacy: 0.0,
            withdrawals_discretionary: 0.0,
        },
        Some(pre_withdrawal_from_using_expected_returns) => {
            let elasticity_of_wealth_wrt_stocks =
                if pre_withdrawal_from_using_expected_returns.wealth == 0.0 {
                    (params.target_allocation.legacy_portfolio
                        + params.target_allocation.regular_portfolio.tpaw
                        + params.target_allocation.regular_portfolio.tpaw)
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
                            * params.target_allocation.regular_portfolio.tpaw
                        + (pre_withdrawal_from_using_expected_returns
                            .present_value_of_spending
                            .withdrawals_regular
                            / pre_withdrawal_from_using_expected_returns.wealth)
                            * params.target_allocation.regular_portfolio.tpaw
                };

            let elasticity_of_extra_withdrawal_goals_wrt_wealth = if elasticity_of_wealth_wrt_stocks
                == 0.0
            {
                0.0
            } else {
                params.target_allocation.regular_portfolio.tpaw / elasticity_of_wealth_wrt_stocks
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
            SingleYearPreWithdrawalScale {
                legacy,
                withdrawals_discretionary,
            }
        }
    };

    // ------ RETURNS -----
    let expected_returns_legacy_portfolio = params.expected_returns.stocks
        * params.target_allocation.legacy_portfolio
        + params.expected_returns.bonds * (1.0 - params.target_allocation.legacy_portfolio);

    // ---- PRESENT VALUE OF SPENDING ----
    let present_value_of_spending = {
        let mut account = AccountForWithdrawal::new(wealth);

        account.withdraw(net_present_value.withdrawals.lmp.with_current_year[year_index]);
        account.withdraw(net_present_value.withdrawals.essential.with_current_year[year_index]);
        let discretionary = account.withdraw(
            net_present_value
                .withdrawals
                .discretionary
                .with_current_year[year_index]
                * (1.0 + scale.withdrawals_discretionary),
        );
        let legacy = account.withdraw(
            (params.legacy_target * (1.0 + scale.legacy))
                / f64::powi(
                    1.0 + expected_returns_legacy_portfolio,
                    (params.num_years - year_index) as i32,
                ),
        );
        let regular = account.balance;

        SingleYearPreWithdrawalPresentValueOfSpending {
            legacy,
            withdrawals_discretionary: discretionary,
            withdrawals_regular: regular,
        }
    };

    SingleYearPreWithdrawal {
        wealth,
        present_value_of_spending,
        scale,
        expected_returns_legacy_portfolio,
    }
}

// ---- TARGET WITHDRAWAL ----
#[inline(always)]
fn calculate_target_withdrawals(
    context: &SingleYearContext,
    _savings_portfolio_starting_balance: f64,
    _pre_withdrawal_from_using_expected_returns: &Option<&SingleYearPreWithdrawal>,
    pre_withdrawal: &SingleYearPreWithdrawal,
    _pass_forward: &SingleYearPassForward,
) -> TargetWithdrawals {
    let SingleYearContext {
        params, year_index, ..
    } = *context;
    let SingleYearPreWithdrawal {
        present_value_of_spending,
        scale,
        ..
    } = pre_withdrawal;

    let withdrawal_started = year_index >= params.withdrawal_start_year;

    let regular_without_lmp = if !withdrawal_started {
        0.0
    } else {
        let p = present_value_of_spending.withdrawals_regular;
        let r = params.expected_returns.stocks * params.target_allocation.regular_portfolio.tpaw
            + params.expected_returns.bonds
                * (1.0 - params.target_allocation.regular_portfolio.tpaw);
        let g = params.spending_tilt;
        let n = params.num_years - year_index;
        if f64::abs(r - g) < 0.0000000001 {
            p / f64::from(n as i32)
        } else {
            (p * (r - g)) / ((1.0 - f64::powi((1.0 + g) / (1.0 + r), n as i32)) * (1.0 + r))
        }
    };

    // Should be 0 if withdrawal has not started.
    let lmp = params.lmp[year_index];
    let essential = params.by_year.withdrawals_essential[year_index];
    let discretionary = params.by_year.withdrawals_discretionary[year_index]
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
    context: &SingleYearContext,
    pre_withdrawal: &SingleYearPreWithdrawal,
    savings_portfolio_after_withdrawals: &portfolio_over_year::AfterWithdrawals,
) -> f64 {
    let SingleYearContext {
        params,
        pre_calculations,
        year_index,
        ..
    } = *context;
    let net_present_value = &pre_calculations.tpaw.net_present_value;
    let SingleYearPreWithdrawal {
        scale,
        expected_returns_legacy_portfolio,
        ..
    } = pre_withdrawal;

    let mut account = AccountForWithdrawal::new(
        savings_portfolio_after_withdrawals.balance
            + net_present_value.savings.without_current_year[year_index],
    );

    account.withdraw(net_present_value.withdrawals.lmp.without_current_year[year_index]);
    account.withdraw(net_present_value.withdrawals.essential.without_current_year[year_index]);

    let present_value_of_discretionary_withdrawals = account.withdraw(
        net_present_value
            .withdrawals
            .discretionary
            .without_current_year[year_index]
            * (1.0 + pre_withdrawal.scale.withdrawals_discretionary),
    );
    let present_value_of_desired_legacy = account.withdraw(
        (params.legacy_target * (1.0 + scale.legacy))
            / f64::powi(
                1.0 + expected_returns_legacy_portfolio,
                (params.num_years - year_index) as i32,
            ),
    );
    let present_value_of_regular_withdrawals = account.balance;

    let stocks_target = present_value_of_desired_legacy * params.target_allocation.legacy_portfolio
        + present_value_of_discretionary_withdrawals
            * params.target_allocation.regular_portfolio.tpaw
        + present_value_of_regular_withdrawals * params.target_allocation.regular_portfolio.tpaw;

    let stocks_achieved = f64::min(savings_portfolio_after_withdrawals.balance, stocks_target);
    let stock_allocation = if savings_portfolio_after_withdrawals.balance > 0.0 {
        stocks_achieved / savings_portfolio_after_withdrawals.balance
    } else {
        0.0
    };

    return stock_allocation;
}

fn get_pass_forward(_withdrawal_target: TargetWithdrawals) -> SingleYearPassForward {
    return SingleYearPassForward {};
}
