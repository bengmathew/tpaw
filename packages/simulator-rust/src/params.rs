use wasm_bindgen::prelude::wasm_bindgen;

use crate::utils::*;

pub struct ParamsByYear {
    pub savings: Box<[f64]>,
    pub withdrawals_essential: Box<[f64]>,
    pub withdrawals_discretionary: Box<[f64]>,
}

pub struct ParamsTargetAllocationRegularPortfolio {
    pub tpaw: f64,
    pub spaw: Box<[f64]>,
}

pub struct ParamsTargetAllocation {
    pub regular_portfolio: ParamsTargetAllocationRegularPortfolio,
    pub legacy_portfolio: f64,
}

pub struct ParamsTest {
    pub truth: Box<[f64]>,
    pub index_into_historical_returns: Vec<usize>,
}

#[wasm_bindgen]
#[derive(Copy, Clone, Debug)]
pub enum ParamsStrategy {
    SPAW = "SPAW",
    TPAW = "TPAW",
}

pub struct Params {
    pub strategy: ParamsStrategy,
    pub num_runs: i32,
    pub num_years: i32,
    pub withdrawal_start_year: i32,
    pub current_savings: f64,
    pub expected_returns: ReturnsAtPointInTime,
    pub historical_returns: Vec<ReturnsAtPointInTime>,
    pub target_allocation: ParamsTargetAllocation,
    pub lmp: f64,
    pub by_year: ParamsByYear,
    pub legacy_target: f64,
    pub legacy_external: f64,
    pub spending_tilt: f64,
    pub spending_ceiling: Option<f64>,
    pub spending_floor: Option<f64>,
    pub test: Option<ParamsTest>,
}
