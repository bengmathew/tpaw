mod params;
mod portfolio_over_month;
mod pre_calculations;
mod run_spaw;
mod run_swr;
mod run_tpaw;
mod utils;

use serde::{Deserialize, Serialize};

use std::cmp::Ordering;

use params::*;
use utils::*;
use wasm_bindgen::prelude::*;
use web_sys::console;

fn to_js_arr(x: &Vec<f64>) -> js_sys::Float64Array {
    unsafe { js_sys::Float64Array::view(&x[..]) }
}
fn to_js_arr_i32(x: &Vec<i32>) -> js_sys::Int32Array {
    unsafe { js_sys::Int32Array::view(&x[..]) }
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
    annual_stats_for_sampled_stock_returns: StatsForWindowSize,
    annual_stats_for_sampled_bond_returns: StatsForWindowSize,
}

#[wasm_bindgen]
impl RunResult {
    pub fn by_mfn_by_run_balance_start(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_mfn_by_run_balance_start)
    }
    pub fn by_mfn_by_run_withdrawals_essential(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_mfn_by_run_withdrawals_essential)
    }
    pub fn by_mfn_by_run_withdrawals_discretionary(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_mfn_by_run_withdrawals_discretionary)
    }
    pub fn by_mfn_by_run_withdrawals_regular(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_mfn_by_run_withdrawals_regular)
    }
    pub fn by_mfn_by_run_withdrawals_total(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_mfn_by_run_withdrawals_total)
    }
    pub fn by_mfn_by_run_withdrawals_from_savings_portfolio_rate(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_mfn_by_run_withdrawals_from_savings_portfolio_rate)
    }
    pub fn by_mfn_by_run_after_withdrawals_allocation_stocks_savings(
        &self,
    ) -> js_sys::Float64Array {
        to_js_arr(&self.by_mfn_by_run_after_withdrawals_allocation_stocks_savings)
    }
    pub fn by_mfn_by_run_after_withdrawals_allocation_stocks_total(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_mfn_by_run_after_withdrawals_allocation_stocks_total)
    }
    pub fn by_run_num_insufficient_fund_months(&self) -> js_sys::Int32Array {
        to_js_arr_i32(&self.by_run_num_insufficient_fund_months)
    }
    pub fn test(&self) -> f64 {
        return 3.5;
    }
    pub fn by_run_ending_balance(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_run_ending_balance)
    }
    pub fn by_run_by_mfn_returns_stocks(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_run_by_mfn_returns_stocks)
    }
    pub fn by_run_by_mfn_returns_bonds(&self) -> js_sys::Float64Array {
        to_js_arr(&self.by_run_by_mfn_returns_bonds)
    }
    pub fn annual_stats_for_sampled_stock_returns(&self) -> StatsForWindowSize {
        self.annual_stats_for_sampled_stock_returns
    }
    pub fn annual_stats_for_sampled_bond_returns(&self) -> StatsForWindowSize {
        self.annual_stats_for_sampled_bond_returns
    }
    // pub fn average_annual_returns_stocks(&self) -> f64 {
    // get_stats_for_window_size_from_log_returns(
    //     &get_log_returns(&self.by_run_by_mfn_returns_stocks),
    //     12,
    // )
    //     .mean
    // }
    // pub fn average_annual_returns_bonds(&self) -> f64 {
    //     get_stats_for_window_size_from_log_returns(
    //         &get_log_returns(&self.by_run_by_mfn_returns_bonds),
    //         12,
    //     )
    //     .mean
    // }
}

#[wasm_bindgen]
#[derive(Copy, Clone, Debug)]
pub enum ParamsSWRWithdrawalType {
    AsPercent = "asPercent",
    AsAmount = "asAmount",
}

#[wasm_bindgen]
pub fn run(
    strategy: ParamsStrategy,
    start_run: usize,
    end_run: usize,
    num_months: usize,
    num_months_to_simulate: usize,
    withdrawal_start_month: usize,
    expected_returns_stocks: f64,
    expected_returns_bonds: f64,
    historical_returns_stocks: Box<[f64]>,
    historical_returns_bonds: Box<[f64]>,
    current_savings: f64,
    target_allocation_regular_portfolio_tpaw: Box<[f64]>,
    target_allocation_regular_portfolio_spaw: Box<[f64]>,
    target_allocation_legacy_portfolio: f64,
    swr_withdrawal_type: ParamsSWRWithdrawalType,
    swr_withdrawal_value: f64,
    lmp: Box<[f64]>,
    by_month_savings: Box<[f64]>,
    by_month_withdrawals_essential: Box<[f64]>,
    by_month_withdrawals_discretionary: Box<[f64]>,
    legacy_target: f64,
    legacy_external: f64,
    spending_tilt: Box<[f64]>,
    spending_ceiling: Option<f64>,
    spending_floor: Option<f64>,
    monte_carlo_sampling: bool,
    monte_carlo_block_size: usize,
    max_num_months: usize,
    test_truth: Option<Box<[f64]>>,
    test_index_into_historical_returns: Option<Box<[usize]>>,
) -> RunResult {
    let expected_monthly_returns = ReturnsAtPointInTime {
        stocks: expected_returns_stocks,
        bonds: expected_returns_bonds,
    };
    let params = Params {
        strategy,
        start_run,
        end_run,
        num_months,
        num_months_to_simulate,
        withdrawal_start_month,
        current_savings,
        expected_monthly_returns,
        historical_returns: (0..historical_returns_stocks.len())
            .map(|i| ReturnsAtPointInTime {
                stocks: historical_returns_stocks[i],
                bonds: historical_returns_bonds[i],
            })
            .collect(),
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
        lmp,
        by_month: ParamsByMonth {
            savings: by_month_savings,
            withdrawals_essential: by_month_withdrawals_essential,
            withdrawals_discretionary: by_month_withdrawals_discretionary,
        },
        legacy_target,
        legacy_external,
        spending_tilt,
        spending_ceiling,
        spending_floor,
        monte_carlo_sampling,
        monte_carlo_block_size,
        max_num_months,
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
        annual_stats_for_sampled_stock_returns: StatsForWindowSize {
            n: 0,
            of_base: Stats {
                mean: 0.0,
                variance: 0.0,
                standard_deviation: 0.0,
                n: 0,
            },
            of_log: Stats {
                mean: 0.0,
                variance: 0.0,
                standard_deviation: 0.0,
                n: 0,
            },
        },
        annual_stats_for_sampled_bond_returns: StatsForWindowSize {
            n: 0,
            of_base: Stats {
                mean: 0.0,
                variance: 0.0,
                standard_deviation: 0.0,
                n: 0,
            },
            of_log: Stats {
                mean: 0.0,
                variance: 0.0,
                standard_deviation: 0.0,
                n: 0,
            },
        },
    };

    match strategy {
        params::ParamsStrategy::TPAW => run_tpaw::run(&params, &mut result),
        params::ParamsStrategy::SPAW => run_spaw::run(&params, &mut result),
        params::ParamsStrategy::SWR => run_swr::run(&params, &mut result),
        _ => panic!(),
    };

    if (result.by_run_by_mfn_returns_stocks.len() >= 12) {
        result.annual_stats_for_sampled_stock_returns = get_stats_for_window_size_from_log_returns(
            &get_log_returns(&result.by_run_by_mfn_returns_stocks),
            12,
        );
        result.annual_stats_for_sampled_bond_returns = get_stats_for_window_size_from_log_returns(
            &get_log_returns(&result.by_run_by_mfn_returns_bonds),
            12,
        );
    }
    return result;
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
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct SampledReturnStats {
    pub one_year: StatsForWindowSize,
    pub five_year: StatsForWindowSize,
    pub ten_year: StatsForWindowSize,
    pub thirty_year: StatsForWindowSize,
    pub n: usize,
}

#[wasm_bindgen]
pub fn get_sampled_returns_stats(
    monthly_returns: Box<[f64]>,
    block_size: usize,
    num_months: usize,
) -> SampledReturnStats {
    // let indexes = &memoized_random(1, num_months, block_size, monthly_returns.len())[0];
    let indexes =
        &generate_random_index_sequences(1, num_months, block_size, monthly_returns.len())[0];
    let ln_one_plus_monthly: Vec<f64> = indexes
        .iter()
        .map(|i| (monthly_returns[*i] + 1.0).ln())
        .collect();

    return SampledReturnStats {
        one_year: get_stats_for_window_size_from_log_returns(&ln_one_plus_monthly, 12 * 1),
        five_year: get_stats_for_window_size_from_log_returns(&ln_one_plus_monthly, 12 * 5),
        ten_year: get_stats_for_window_size_from_log_returns(&ln_one_plus_monthly, 12 * 10),
        thirty_year: get_stats_for_window_size_from_log_returns(&ln_one_plus_monthly, 12 * 30),
        n: 3,
    };
}

pub fn test_log() {
    console::log_1(&"hello".to_string().into());
}

#[wasm_bindgen]
pub fn clear_memoized_random() {
    utils::clear_memoized_random_store();
}
