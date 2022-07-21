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
        .tpaw = vec![0.0; params.num_years].into_boxed_slice();
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
        savings_portfolio_after_contributions,
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

    let mut stock_allocation_on_total_portfolio = savings_portfolio_at_end.stock_allocation_amount
        / (savings_portfolio_after_withdrawals.balance
            + pre_calculations
                .tpaw
                .net_present_value
                .savings
                .without_current_year[year_index]);
    stock_allocation_on_total_portfolio = if f64::is_nan(stock_allocation_on_total_portfolio)
        || f64::is_infinite(stock_allocation_on_total_portfolio)
    {
        0.0
    } else {
        stock_allocation_on_total_portfolio
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
        savings_portfolio_after_withdrawals
            .withdrawals
            .from_savings_portfolio_rate;
    result.by_yfn_by_run_excess_withdrawals_regular[year_run_index] =
        savings_portfolio_after_withdrawals.withdrawals.regular
            - withdrawal_from_using_bond_returns.regular;
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

#[derive(Serialize, Deserialize)]
pub struct SingleYearPreWithdrawal {}

struct SingleYearPassForward {
    withdrawal: f64,
}

fn initial_pass_forward() -> SingleYearPassForward {
    return SingleYearPassForward { withdrawal: 0.0 };
}

// ---- PRE WITHDRAWAL ----

#[inline(always)]
fn calculate_pre_withdrawal(
    _context: &SingleYearContext,
    _pre_withdrawal_from_using_expected_returns: &Option<&SingleYearPreWithdrawal>,
) -> SingleYearPreWithdrawal {
    SingleYearPreWithdrawal {}
}

// ---- TARGET WITHDRAWAL ----
#[inline(always)]
fn calculate_target_withdrawals(
    context: &SingleYearContext,
    savings_portfolio_starting_balance: f64,
    _pre_withdrawal_from_using_expected_returns: &Option<&SingleYearPreWithdrawal>,
    _pre_withdrawal: &SingleYearPreWithdrawal,
    pass_forward: &SingleYearPassForward,
) -> TargetWithdrawals {
    let SingleYearContext {
        params, year_index, ..
    } = *context;

    let regular_without_lmp = match year_index.cmp(&params.withdrawal_start_year) {
        Ordering::Less => 0.0,
        Ordering::Equal => match params.swr_withdrawal {
            ParamsSWRWithdrawal::AsPercent { percent } => {
                savings_portfolio_starting_balance * percent
            }
            ParamsSWRWithdrawal::AsAmount { amount } => amount,
        },
        Ordering::Greater => pass_forward.withdrawal,
    };
    TargetWithdrawals {
        lmp: 0.0,
        essential: params.by_year.withdrawals_essential[year_index],
        discretionary: params.by_year.withdrawals_discretionary[year_index],
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

fn get_pass_forward(withdrawal_target: TargetWithdrawals) -> SingleYearPassForward {
    return SingleYearPassForward {
        withdrawal: withdrawal_target.regular_without_lmp,
    };
}
