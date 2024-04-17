pub mod constants;
pub mod data_for_market_based_plan_param_values;
pub mod historical_monthly_returns;
pub mod params;
pub mod plan_params;
mod portfolio_over_month;
mod pre_calculations;
mod run_spaw;
mod run_swr;
mod run_tpaw;
pub mod utils;

use crate::{plan_params::PlanParams, return_series::periodize_log_returns};
use data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues;
use params::*;
use plan_params::process_plan_params::{
    process_market_data::MarketDataProcessed, process_plan_params,
};
use return_series::Stats;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use tsify::Tsify;
use utils::*;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct BaseAndLogStats {
    pub base: Stats,
    pub log: Stats,
}

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct RunResultInfo {
    market_data: MarketDataProcessed,
}

#[wasm_bindgen]
pub struct RunResult {
    by_mfn_by_run_balance_start: Vec<f64>,
    by_mfn_by_run_withdrawals_essential: Vec<f64>,
    by_mfn_by_run_withdrawals_discretionary: Vec<f64>,
    by_mfn_by_run_withdrawals_regular: Vec<f64>,
    by_mfn_by_run_withdrawals_total: Vec<f64>,
    by_mfn_by_run_withdrawals_from_savings_portfolio_rate: Vec<f64>,
    by_mfn_by_run_after_withdrawals_allocation_stocks_savings: Vec<f64>,
    by_mfn_by_run_after_withdrawals_allocation_stocks_total: Vec<f64>,
    by_run_ending_balance: Vec<f64>,
    by_run_num_insufficient_fund_months: Vec<i32>,
    // Needed only for testing, but does not impact performance by much,
    // so ok to leave it in.
    by_run_by_mfn_returns_stocks: Vec<f64>,
    by_run_by_mfn_returns_bonds: Vec<f64>,
    annual_stats_for_sampled_stock_returns: Option<BaseAndLogStats>,
    annual_stats_for_sampled_bond_returns: Option<BaseAndLogStats>,
    pub info: RunResultInfo,
}

#[wasm_bindgen]
impl RunResult {
    pub fn by_mfn_by_run_balance_start(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_mfn_by_run_balance_start)
    }
    pub fn by_mfn_by_run_withdrawals_essential(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_mfn_by_run_withdrawals_essential)
    }
    pub fn by_mfn_by_run_withdrawals_discretionary(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_mfn_by_run_withdrawals_discretionary)
    }
    pub fn by_mfn_by_run_withdrawals_regular(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_mfn_by_run_withdrawals_regular)
    }
    pub fn by_mfn_by_run_withdrawals_total(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_mfn_by_run_withdrawals_total)
    }
    pub fn by_mfn_by_run_withdrawals_from_savings_portfolio_rate(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_mfn_by_run_withdrawals_from_savings_portfolio_rate)
    }
    pub fn by_mfn_by_run_after_withdrawals_allocation_stocks_savings(
        &self,
    ) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_mfn_by_run_after_withdrawals_allocation_stocks_savings)
    }
    pub fn by_mfn_by_run_after_withdrawals_allocation_stocks_total(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_mfn_by_run_after_withdrawals_allocation_stocks_total)
    }
    pub fn by_run_num_insufficient_fund_months(&self) -> js_sys::Int32Array {
        vec_i32_js_view(&self.by_run_num_insufficient_fund_months)
    }
    pub fn test(&self) -> f64 {
        return 3.5;
    }
    pub fn by_run_ending_balance(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_run_ending_balance)
    }
    pub fn by_run_by_mfn_returns_stocks(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_run_by_mfn_returns_stocks)
    }
    pub fn by_run_by_mfn_returns_bonds(&self) -> js_sys::Float64Array {
        vec_f64_js_view(&self.by_run_by_mfn_returns_bonds)
    }
    pub fn annual_stats_for_sampled_stock_returns(&self) -> Option<BaseAndLogStats> {
        self.annual_stats_for_sampled_stock_returns
    }
    pub fn annual_stats_for_sampled_bond_returns(&self) -> Option<BaseAndLogStats> {
        self.annual_stats_for_sampled_bond_returns
    }
}

#[wasm_bindgen]
#[derive(Copy, Clone, Debug)]
pub enum ParamsSWRWithdrawalType {
    AsPercent = "asPercent",
    AsAmount = "asAmount",
}

#[wasm_bindgen]
pub fn run(
    plan_params: PlanParams,
    market_data: DataForMarketBasedPlanParamValues,
    start_run: usize,
    end_run: usize,
    num_months_to_simulate: usize,
    current_savings: f64,
    target_allocation_regular_portfolio_tpaw: Box<[f64]>,
    target_allocation_regular_portfolio_spaw: Box<[f64]>,
    target_allocation_legacy_portfolio: f64,
    swr_withdrawal_type: ParamsSWRWithdrawalType,
    swr_withdrawal_value: f64,
    legacy_target: f64,
    legacy_external: f64,
    spending_tilt: Box<[f64]>,
    spending_ceiling: Option<f64>,
    spending_floor: Option<f64>,
    max_num_months: usize,
    rand_seed: u64,
    test_truth: Option<Box<[f64]>>,
    test_index_into_historical_returns: Option<Box<[usize]>>,
) -> RunResult {
    let plan_params_processed = process_plan_params(&plan_params, &market_data);

    let params = Params {
        start_run,
        end_run,
        num_months_to_simulate,
        current_savings,
        target_allocation: ParamsTargetAllocation {
            regular_portfolio: ParamsTargetAllocationRegularPortfolio {
                tpaw: target_allocation_regular_portfolio_tpaw,
                spaw_and_swr: target_allocation_regular_portfolio_spaw,
            },
            legacy_portfolio: target_allocation_legacy_portfolio,
        },
        swr_withdrawal: match swr_withdrawal_type {
            ParamsSWRWithdrawalType::AsPercent => ParamsSWRWithdrawal::AsPercent {
                percent: swr_withdrawal_value,
            },
            ParamsSWRWithdrawalType::AsAmount => ParamsSWRWithdrawal::AsAmount {
                amount: swr_withdrawal_value,
            },
            _ => panic!(),
        },
        legacy_target,
        legacy_external,
        spending_tilt,
        spending_ceiling,
        spending_floor,
        max_num_months,
        rand_seed,
        test: if let Some(truth) = test_truth {
            Some(ParamsTest {
                truth,
                index_into_historical_returns: test_index_into_historical_returns.unwrap().to_vec(),
            })
        } else {
            None
        },
    };
    let num_runs = end_run - start_run;

    let create_vec = || vec![0.0; (num_runs * num_months_to_simulate) as usize];
    let mut result = RunResult {
        by_mfn_by_run_balance_start: create_vec(),
        by_mfn_by_run_withdrawals_essential: create_vec(),
        by_mfn_by_run_withdrawals_discretionary: create_vec(),
        by_mfn_by_run_withdrawals_regular: create_vec(),
        by_mfn_by_run_withdrawals_total: create_vec(),
        by_mfn_by_run_withdrawals_from_savings_portfolio_rate: create_vec(),
        by_mfn_by_run_after_withdrawals_allocation_stocks_savings: create_vec(),
        by_mfn_by_run_after_withdrawals_allocation_stocks_total: create_vec(),
        by_run_num_insufficient_fund_months: vec![0; (num_runs) as usize],
        by_run_ending_balance: vec![0.0; (num_runs) as usize],
        // Test
        by_run_by_mfn_returns_stocks: create_vec(),
        by_run_by_mfn_returns_bonds: create_vec(),
        annual_stats_for_sampled_stock_returns: None,
        annual_stats_for_sampled_bond_returns: None,
        info: RunResultInfo {
            market_data: plan_params_processed.market_data,
        },
    };

    match plan_params.advanced.strategy {
        plan_params::Strategy::TPAW => {
            run_tpaw::run(&plan_params, &plan_params_processed, &params, &mut result)
        }
        plan_params::Strategy::SPAW => {
            run_spaw::run(&plan_params, &plan_params_processed, &params, &mut result)
        }
        plan_params::Strategy::SWR => {
            run_swr::run(&plan_params, &plan_params_processed, &params, &mut result)
        }
    };

    if result.by_run_by_mfn_returns_stocks.len() >= 12 {
        fn get(monthly_non_log_returns: &[f64]) -> BaseAndLogStats {
            let monthly_log_returns = monthly_non_log_returns
                .iter()
                .map(|x| x.ln_1p())
                .collect::<Vec<f64>>();
            let annualized_log_returns = periodize_log_returns(&monthly_log_returns, 12);
            let annualized_non_log_returns = annualized_log_returns
                .iter()
                .map(|x| x.exp_m1())
                .collect::<Vec<f64>>();
            BaseAndLogStats {
                base: Stats::from_series(&annualized_non_log_returns),
                log: Stats::from_series(&annualized_log_returns),
            }
        }
        result.annual_stats_for_sampled_stock_returns =
            Some(get(&result.by_run_by_mfn_returns_stocks));
        result.annual_stats_for_sampled_bond_returns =
            Some(get(&result.by_run_by_mfn_returns_bonds));
    }
    result
}

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

#[wasm_bindgen]
pub fn sort(data: Box<[f64]>) -> Box<[f64]> {
    let mut result: Vec<f64> = Vec::from(data);
    result.sort_unstable_by(cmp_f64);
    return result.into_boxed_slice();
}

#[wasm_bindgen]
pub fn process_plan_params_lib(
    plan_params: PlanParams,
    market_data: DataForMarketBasedPlanParamValues,
) -> Vec<u8> {
    let plan_params_processed =
        plan_params::process_plan_params::process_plan_params(&plan_params, &market_data);
    rmp_serde::encode::to_vec_named(&plan_params_processed).unwrap()
}

#[wasm_bindgen]
pub fn test() {}
