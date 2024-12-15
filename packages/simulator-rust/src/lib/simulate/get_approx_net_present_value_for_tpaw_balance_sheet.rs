#![allow(non_camel_case_types)]
use std::collections::HashMap;

use crate::{
    cuda_bridge_utils::{f_cuda, f_cuda_currency, f_cuda_currency_npv},
    shared_types::StocksAndBonds,
    wire::{WireIdAndDouble, WireSimulationResultTpawNetPresentValueApproxForBalanceSheet},
};

use super::process_plan_params_server::process_amount_timed::{
    AmountTimedProcessed, AmountTimedProcessed_Group,
};

#[derive(Clone)]
pub struct TPAW_ApproxNetPresentValueForBalanceSheet {
    pub future_savings: HashMap<String, f_cuda_currency_npv>,
    pub income_during_retirement: HashMap<String, f_cuda_currency_npv>,
    pub essential_expenses: HashMap<String, f_cuda_currency_npv>,
    pub discretionary_expenses: HashMap<String, f_cuda_currency_npv>,
    pub legacy_target: f_cuda_currency,
}

fn _hashmap_to_wire(map: HashMap<String, f_cuda_currency_npv>) -> Vec<WireIdAndDouble> {
    map.into_iter()
        .map(|(k, v)| WireIdAndDouble {
            id: k,
            value: v as f64,
        })
        .collect()
}

impl From<TPAW_ApproxNetPresentValueForBalanceSheet>
    for WireSimulationResultTpawNetPresentValueApproxForBalanceSheet
{
    fn from(value: TPAW_ApproxNetPresentValueForBalanceSheet) -> Self {
        Self {
            future_savings: _hashmap_to_wire(value.future_savings),
            income_during_retirement: _hashmap_to_wire(value.income_during_retirement),
            essential_expenses: _hashmap_to_wire(value.essential_expenses),
            discretionary_expenses: _hashmap_to_wire(value.discretionary_expenses),
            legacy_target: value.legacy_target,
        }
    }
}

// Typically we don't work in  Work in f_cuda types but convert at boundary, but
// in this case we want to compute at the same precision as cuda because we are
// explicitly keeping this in line with the approx net present value
// calculations done in CUDA.
pub fn get_approx_net_present_value_for_tpaw_balance_sheet(
    amount_timed: &AmountTimedProcessed,
    empirical_monthly_non_log_expected_return: StocksAndBonds<f_cuda>,
    net_present_value_approx_legacy_target_expected_run: f_cuda_currency,
    stock_allocation_total_portfolio_expected_run_by_mfn: &[f_cuda],
) -> TPAW_ApproxNetPresentValueForBalanceSheet {
    let num_months = stock_allocation_total_portfolio_expected_run_by_mfn.len();

    let bond_rate_arr: Vec<f_cuda> = (0..num_months)
        .map(|_| empirical_monthly_non_log_expected_return.bonds)
        .collect();
    let regular_rate_arr: Vec<f_cuda> = stock_allocation_total_portfolio_expected_run_by_mfn
        .iter()
        .map(|stock_allocation| {
            empirical_monthly_non_log_expected_return.stocks * stock_allocation
                + empirical_monthly_non_log_expected_return.bonds * (1.0 - stock_allocation)
        })
        .collect();
    TPAW_ApproxNetPresentValueForBalanceSheet {
        future_savings: _for_amount_timed_group(
            &amount_timed.future_savings,
            &bond_rate_arr,
            false,
        ),
        income_during_retirement: _for_amount_timed_group(
            &amount_timed.income_during_retirement,
            &bond_rate_arr,
            false,
        ),
        essential_expenses: _for_amount_timed_group(
            &amount_timed.essential_expenses,
            &bond_rate_arr,
            false,
        ),
        discretionary_expenses: _for_amount_timed_group(
            &amount_timed.discretionary_expenses,
            &regular_rate_arr,
            false,
        ),
        legacy_target: net_present_value_approx_legacy_target_expected_run,
    }
}

fn _for_amount_timed_group(
    group: &AmountTimedProcessed_Group,
    rate_arr: &[f_cuda],
    debug: bool,
) -> HashMap<String, f_cuda_currency_npv> {
    group
        .by_id
        .iter()
        .map(|(k, v)| (k.clone(), _get_net_present_value(v, rate_arr, debug)))
        .collect()
}

fn _get_net_present_value(amounts: &[f64], rates: &[f_cuda], debug: bool) -> f_cuda_currency_npv {
    amounts
        .iter()
        .zip(rates.iter())
        .rev()
        .fold(0.0, |acc, (amount, rate)| {
            let result = (*amount as f_cuda_currency_npv) + acc / (1.0 + rate);
            if debug {
                println!("{}", result);
            }
            result
        })
}
