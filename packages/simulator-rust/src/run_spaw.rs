use crate::params::*;
use crate::portfolio_over_year;
use crate::portfolio_over_year::SingleYearContext;
use crate::portfolio_over_year::TargetWithdrawals;
use crate::pre_calculations::*;
use crate::utils::*;
use crate::RunResult;
use serde::Deserialize;
use serde::Serialize;

pub fn run(params: &Params, pre_calculations: &PreCalculations, result: &mut RunResult) {
    let results_from_using_expected_returns =
        run_using_expected_returns(&params, &pre_calculations);
    for run_index in 0..params.num_runs {
        run_using_historical_returns(
            &params,
            &pre_calculations,
            &results_from_using_expected_returns,
            result,
            run_index as usize,
        )
    }
}

fn run_using_expected_returns(
    params: &Params,
    pre_calculations: &PreCalculations,
) -> Vec<SingleYearPreWithdrawal> {
    let mut balance_starting = params.current_savings;
    (0..params.num_years as usize)
        .map(|year_index| {
            let context = SingleYearContext {
                params,
                pre_calculations,
                year_index,
                returns: &params.expected_returns,
                balance_starting,
            };
            let r = run_for_single_year_using_expected_returns(&context, &None);
            balance_starting = r.0;
            r.1
        })
        .collect()
}

fn run_using_historical_returns(
    params: &Params,
    pre_calculations: &PreCalculations,
    pre_withdrawal_from_using_expected_returns: &Vec<SingleYearPreWithdrawal>,
    result: &mut RunResult,
    run_index: usize,
) {
    let n = params.num_years as usize;

    let index_into_historical_returns = if let Some(x) = &params.test {
        &x.index_into_historical_returns
    } else {
        memoized_random(
            params.num_runs,
            params.num_years as usize,
            params.historical_returns.len(),
            run_index as usize,
        )
    };

    let mut balance_starting = params.current_savings;
    for year_index in 0..n {
        let context = SingleYearContext {
            params,
            pre_calculations,
            year_index,
            returns: &params.historical_returns[index_into_historical_returns[year_index]],
            balance_starting,
        };
        balance_starting = run_for_single_year_using_historical_returns(
            &context,
            &Some(pre_withdrawal_from_using_expected_returns),
            result,
            run_index,
        );
    }
    result.by_run_ending_balance[run_index] = balance_starting;
}

#[inline(always)]
fn run_for_single_year_using_expected_returns(
    context: &SingleYearContext,
    pre_withdrawal_from_using_expected_returns: &Option<&Vec<SingleYearPreWithdrawal>>,
) -> (f64, SingleYearPreWithdrawal) {
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
    let curr_pre_withdrawal_from_using_expected_returns =
        match pre_withdrawal_from_using_expected_returns {
            None => None,
            Some(x) => Some(&x[year_index]),
        };

    let pre_withdrawal =
        calculate_pre_withdrawal(context, &curr_pre_withdrawal_from_using_expected_returns);

    let target_withdrawals = calculate_target_withdrawals(
        context,
        &curr_pre_withdrawal_from_using_expected_returns,
        &pre_withdrawal,
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

    (savings_portfolio_at_end.balance, pre_withdrawal)
}

#[inline(always)]
fn run_for_single_year_using_historical_returns(
    context: &SingleYearContext,
    pre_withdrawal_from_using_expected_returns: &Option<&Vec<SingleYearPreWithdrawal>>,
    result: &mut RunResult,
    run_index: usize,
) -> f64 {
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

    let curr_pre_withdrawal_from_using_expected_returns =
        match pre_withdrawal_from_using_expected_returns {
            None => None,
            Some(x) => Some(&x[year_index]),
        };

    let pre_withdrawal =
        calculate_pre_withdrawal(context, &curr_pre_withdrawal_from_using_expected_returns);

    let target_withdrawals = calculate_target_withdrawals(
        context,
        &curr_pre_withdrawal_from_using_expected_returns,
        &pre_withdrawal,
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

    let year_run_index = (year_index * params.num_runs as usize) + run_index;
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
    result.by_yfn_by_run_after_withdrawals_allocation_stocks[year_run_index] =
        savings_portfolio_at_end.stock_allocation;

    // if let Some(x) = &params.test {
    //     let ours = &savings_portfolio_at_end.balance;
    //     let truth = x.truth[year_index];
    //     let diff = ours - truth;
    //     console::log_1(
    //         &format!(
    //             "{:3} {:15.2} {:15.2} {:15.2}",
    //             year_index, diff, ours, truth
    //         )
    //         .into(),
    //     );
    //     if year_index == 0 {
    //         console::log_1(
    //             &JsValue::from_serde(&(
    //                 &savings_portfolio_after_contributions,
    //                 &savings_portfolio_after_withdrawals,
    //                 &savings_portfolio_at_end,
    //             ))
    //             .unwrap(),
    //         );
    //     }
    // }

    savings_portfolio_at_end.balance
}

// -----------------------------------------------
// --------------------- ACTUAL ------------------
// -----------------------------------------------

#[derive(Serialize, Deserialize)]
struct SingleYearPreWithdrawal {
    wealth_less_essential_and_lmp_expenses: f64,
}

// ---- PRE WITHDRAWAL ----

#[inline(always)]
fn calculate_pre_withdrawal(
    context: &SingleYearContext,
    pre_withdrawal_from_using_expected_returns: &Option<&SingleYearPreWithdrawal>,
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
) -> TargetWithdrawals {
    let SingleYearContext {
        params,
        pre_calculations,
        year_index,
        ..
    } = *context;

    let withdrawal_started = (year_index as i32) >= (params.withdrawal_start_year);
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

    let lmp = if withdrawal_started { params.lmp } else { 0.0 };
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
  pre_withdrawal: &SingleYearPreWithdrawal,
  savings_portfolio_after_withdrawals: &portfolio_over_year::AfterWithdrawals,
) -> f64 {
    context.params.target_allocation.regular_portfolio.spaw[context.year_index]
}
