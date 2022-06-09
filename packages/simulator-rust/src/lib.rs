mod params;
mod portfolio_over_year;
mod pre_calculations;
mod run_spaw;
mod run_tpaw;
mod utils;

use std::cmp::Ordering;

use params::*;
use pre_calculations::*;
use utils::*;
use wasm_bindgen::prelude::*;
use web_sys::console;

fn to_js_arr(x: &Vec<f64>) -> js_sys::Float64Array {
    unsafe { js_sys::Float64Array::view(&x[..]) }
}

#[wasm_bindgen]
pub struct RunResult {
    by_yfn_by_run_balance_start: Vec<f64>,
    by_yfn_by_run_withdrawals_essential: Vec<f64>,
    by_yfn_by_run_withdrawals_discretionary: Vec<f64>,
    by_yfn_by_run_withdrawals_regular: Vec<f64>,
    by_yfn_by_run_withdrawals_total: Vec<f64>,
    by_yfn_by_run_withdrawals_from_savings_portfolio_rate: Vec<f64>,
    by_yfn_by_run_after_withdrawals_allocation_stocks: Vec<f64>,
    by_run_ending_balance: Vec<f64>,
}

#[wasm_bindgen]
impl RunResult {
    pub fn by_yfn_by_run_balance_start(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_yfn_by_run_balance_start)
    }
    pub fn by_yfn_by_run_withdrawals_essential(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_yfn_by_run_withdrawals_essential)
    }
    pub fn by_yfn_by_run_withdrawals_discretionary(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_yfn_by_run_withdrawals_discretionary)
    }
    pub fn by_yfn_by_run_withdrawals_regular(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_yfn_by_run_withdrawals_regular)
    }
    pub fn by_yfn_by_run_withdrawals_total(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_yfn_by_run_withdrawals_total)
    }
    pub fn by_yfn_by_run_withdrawals_from_savings_portfolio_rate(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_yfn_by_run_withdrawals_from_savings_portfolio_rate)
    }
    pub fn by_yfn_by_run_after_withdrawals_allocation_stocks(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_yfn_by_run_after_withdrawals_allocation_stocks)
    }
    pub fn by_run_ending_balance(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_run_ending_balance)
    }
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
pub fn run(
    strategy: ParamsStrategy,
    num_runs: i32,
    num_years: i32,
    withdrawal_start_year: i32,
    expected_returns_stocks: f64,
    expected_returns_bonds: f64,
    historical_returns_stocks: Box<[f64]>,
    historical_returns_bonds: Box<[f64]>,
    current_savings: f64,
    target_allocation_regular_portfolio_tpaw: f64,
    target_allocation_regular_portfolio_spaw: Box<[f64]>,
    target_allocation_legacy_portfolio: f64,
    lmp: f64,
    by_year_savings: Box<[f64]>,
    by_year_withdrawals_essential: Box<[f64]>,
    by_year_withdrawals_discretionary: Box<[f64]>,
    legacy_target: f64,
    legacy_external: f64,
    spending_tilt: f64,
    spending_ceiling: Option<f64>,
    spending_floor: Option<f64>,
    test_truth: Option<Box<[f64]>>,
    test_index_into_historical_returns: Option<Box<[usize]>>,
) -> RunResult {
    let expected_returns = ReturnsAtPointInTime {
        stocks: expected_returns_stocks,
        bonds: expected_returns_bonds,
    };
    let params = Params {
        strategy,
        num_runs,
        num_years,
        withdrawal_start_year,
        current_savings,
        expected_returns,
        historical_returns: (0..historical_returns_stocks.len())
            .map(|i| ReturnsAtPointInTime {
                stocks: historical_returns_stocks[i],
                bonds: historical_returns_bonds[i],
            })
            .collect(),
        target_allocation: ParamsTargetAllocation {
            regular_portfolio: ParamsTargetAllocationRegularPortfolio {
                tpaw: target_allocation_regular_portfolio_tpaw,
                spaw: target_allocation_regular_portfolio_spaw,
            },
            legacy_portfolio: target_allocation_legacy_portfolio,
        },
        lmp,
        by_year: ParamsByYear {
            savings: by_year_savings,
            withdrawals_essential: by_year_withdrawals_essential,
            withdrawals_discretionary: by_year_withdrawals_discretionary,
        },
        legacy_target,
        legacy_external,
        spending_tilt,
        spending_ceiling,
        spending_floor,
        test: if let Some(truth) = test_truth {
            Some(ParamsTest {
                truth,
                index_into_historical_returns: test_index_into_historical_returns.unwrap().to_vec(),
            })
        } else {
            None
        },
    };

    let pre_calculations = pre_calculations(&params);

    let create_vec = || vec![0.0; (num_runs * num_years) as usize];
    let mut result = RunResult {
        by_yfn_by_run_balance_start: create_vec(),
        by_yfn_by_run_withdrawals_essential: create_vec(),
        by_yfn_by_run_withdrawals_discretionary: create_vec(),
        by_yfn_by_run_withdrawals_regular: create_vec(),
        by_yfn_by_run_withdrawals_total: create_vec(),
        by_yfn_by_run_withdrawals_from_savings_portfolio_rate: create_vec(),
        by_yfn_by_run_after_withdrawals_allocation_stocks: create_vec(),
        by_run_ending_balance: vec![0.0; (num_runs) as usize],
    };

    match strategy {
        params::ParamsStrategy::TPAW => run_tpaw::run(&params, &pre_calculations, &mut result),
        params::ParamsStrategy::SPAW => run_spaw::run(&params, &pre_calculations, &mut result),
        _=> panic!()
    };
    return result;
}
