use wasm_bindgen::prelude::wasm_bindgen;

use crate::utils::*;

#[derive(Clone)]
pub struct ParamsByYear {
    pub savings: Box<[f64]>,
    pub withdrawals_essential: Box<[f64]>,
    pub withdrawals_discretionary: Box<[f64]>,
}

#[derive(Clone)]
pub struct ParamsTargetAllocationRegularPortfolio {
    pub tpaw: f64,
    pub spaw: Box<[f64]>,
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

#[wasm_bindgen]
#[derive(Copy, Clone, Debug)]
pub enum ParamsStrategy {
    SPAW = "SPAW",
    TPAW = "TPAW",
}

#[derive(Clone)]
pub struct Params {
    pub strategy: ParamsStrategy,
    pub start_run: usize,
    pub end_run: usize,
    pub num_years: usize,
    pub withdrawal_start_year: usize,
    pub current_savings: f64,
    pub expected_returns: ReturnsAtPointInTime,
    pub historical_returns: Vec<ReturnsAtPointInTime>,
    pub target_allocation: ParamsTargetAllocation,
    pub lmp: Box<[f64]>,
    pub by_year: ParamsByYear,
    pub legacy_target: f64,
    pub legacy_external: f64,
    pub spending_tilt: f64,
    pub spending_ceiling: Option<f64>,
    pub spending_floor: Option<f64>,
    pub monte_carlo_sampling: bool,
    pub test: Option<ParamsTest>,
}
