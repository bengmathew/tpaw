use crate::{
    data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues,
    expected_value_of_returns::annual_non_log_to_monthly_non_log_return_rate, plan_params::{self, normalize_plan_params::plan_params_normalized},
};
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use super::get_suggested_annual_inflation::get_suggested_annual_inflation;

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct InflationProcessed {
    pub annual: f64,
    pub monthly: f64,
}

pub fn process_annual_inflation(
    inflation: &plan_params_normalized::AnnualInfaltion,
    market_data: &DataForMarketBasedPlanParamValues,
) -> InflationProcessed {
    let annual = match inflation {
        plan_params_normalized::AnnualInfaltion::Suggested => get_suggested_annual_inflation(market_data),
        plan_params_normalized::AnnualInfaltion::Manual { value } => *value,
    };
    let monthly = annual_non_log_to_monthly_non_log_return_rate(annual);
    return InflationProcessed { annual, monthly };
}
