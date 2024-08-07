use crate::{
    expected_value_of_returns::annual_non_log_to_monthly_non_log_return_rate,
    plan_params::{self},
};
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use super::process_market_data::MarketDataProcessed;

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct InflationProcessed {
    pub annual: f64,
    pub monthly: f64,
}

pub fn process_annual_inflation(
    inflation: &plan_params::AnnualInfaltion,
    market_data: &MarketDataProcessed,
) -> InflationProcessed {
    let annual = match inflation {
        plan_params::AnnualInfaltion::Suggested => market_data.inflation.suggested_annual,
        plan_params::AnnualInfaltion::Manual { value } => *value,
    };
    let monthly = annual_non_log_to_monthly_non_log_return_rate(annual);
    return InflationProcessed { annual, monthly };
}
