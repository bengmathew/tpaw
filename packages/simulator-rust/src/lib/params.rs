use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::utils::*;

use self::shared_types::StocksAndBonds;

#[derive(Clone)]
pub struct ParamsByMonth {
    pub savings: Box<[f64]>,
    pub withdrawals_essential: Box<[f64]>,
    pub withdrawals_discretionary: Box<[f64]>,
}

#[derive(Clone)]
pub struct ParamsTargetAllocationRegularPortfolio {
    pub tpaw: Box<[f64]>,
    pub spaw_and_swr: Box<[f64]>,
}

#[derive(Clone)]
pub struct ParamsTargetAllocation {
    pub regular_portfolio: ParamsTargetAllocationRegularPortfolio,
    pub legacy_portfolio: f64,
}

#[derive(Clone)]
pub struct ParamsTest {
    pub truth: Box<[f64]>,
    pub index_into_historical_returns: Vec<usize>,
}

#[derive(Clone)]
pub enum ParamsSWRWithdrawal {
    AsPercent { percent: f64 },
    AsAmount { amount: f64 },
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Tsify)]
#[tsify( from_wasm_abi)]
pub enum ParamsStrategy {
    SPAW,
    TPAW,
    SWR,
}

#[derive(Clone)]
pub struct Params {
    pub strategy: ParamsStrategy,
    pub start_run: usize,
    pub end_run: usize,
    pub num_months: usize,
    pub num_months_to_simulate: usize,
    pub withdrawal_start_month: usize,
    pub current_savings: f64,
    pub expected_monthly_returns: StocksAndBonds<f64>,
    pub historical_returns: Vec<StocksAndBonds<f64>>,
    pub target_allocation: ParamsTargetAllocation,
    pub swr_withdrawal: ParamsSWRWithdrawal,
    pub lmp: Box<[f64]>,
    pub by_month: ParamsByMonth,
    pub legacy_target: f64,
    pub legacy_external: f64,
    pub spending_tilt: Box<[f64]>,
    pub spending_ceiling: Option<f64>,
    pub spending_floor: Option<f64>,
    pub monte_carlo_sampling: bool,
    pub monte_carlo_block_size: usize,
    pub monte_carlo_stagger_run_starts: bool,
    pub max_num_months: usize,
    pub rand_seed: u64,
    pub test: Option<ParamsTest>,
}
