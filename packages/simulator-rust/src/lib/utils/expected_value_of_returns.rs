use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EmpiricalAnnualNonLogExpectedReturnInfo {
    pub value: f64,
    pub block_size: Option<usize>,
    pub log_volatility_scale: f64,
}

#[inline(always)]
pub fn annual_non_log_to_monthly_non_log_return_rate(annual_non_log: f64) -> f64 {
    (1.0 + annual_non_log).powf(1.0 / 12.0) - 1.0
}

#[inline(always)]
pub fn monthly_non_log_to_annual_non_log_return_rate(monthly_non_log: f64) -> f64 {
    (1.0 + monthly_non_log).powi(12) - 1.0
}
