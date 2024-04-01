pub mod get_suggested_annual_inflation;
pub mod plan_params_processed;
pub mod process_annual_inflation;
pub mod process_by_month_params;
pub mod process_expected_returns_for_planning;
pub mod process_historical_monthly_log_returns_adjustment;
pub mod process_risk;

use self::{
    plan_params_processed::PlanParamsProcessed, process_annual_inflation::process_annual_inflation,
    process_by_month_params::process_by_month_params,
    process_expected_returns_for_planning::process_expected_returns_for_planning,
    process_historical_monthly_log_returns_adjustment::process_historical_monthly_log_returns_adjustment,
};
use crate::{
    data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues, vec_f64_js_view,
};
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use super::plan_params_rust::PlanParamsRust;

#[wasm_bindgen]
pub struct PlanParamsProcessedJS {
    #[wasm_bindgen(skip)]
    pub without_arrays: PlanParamsProcessed,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(tag = "type", rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum PlanParamsProcessedJSArrayName {
    #[serde(rename = "historicalMonthlyReturnsAdjusted.stocks.logSeries")]
    #[allow(non_camel_case_types)]
    HistoricalMonthlyReturnsAdjusted_Stocks_LogSeries,
    #[serde(rename = "historicalMonthlyReturnsAdjusted.bonds.logSeries")]
    #[allow(non_camel_case_types)]
    HistoricalMonthlyReturnsAdjusted_Bonds_LogSeries,
    #[allow(non_camel_case_types)]
    #[serde(rename = "wealth.total")]
    Wealth_Total,
    #[allow(non_camel_case_types)]
    #[serde(rename = "wealth.futureSavings.byId")]
    Wealth_FutureSavings_ById { id: String },
    #[allow(non_camel_case_types)]
    #[serde(rename = "wealth.futureSavings.total")]
    Wealth_FutureSavings_Total,
    #[allow(non_camel_case_types)]
    #[serde(rename = "wealth.incomeDuringRetirement.byId")]
    Wealth_IncomeDuringRetirement_ById { id: String },
    #[allow(non_camel_case_types)]
    #[serde(rename = "wealth.incomeDuringRetirement.total")]
    Wealth_IncomeDuringRetirement_Total,
    #[allow(non_camel_case_types)]
    #[serde(rename = "adjustmentsToSpending.extraSpending.essential.byId")]
    AdjustmentsToSpending_ExtraSpending_Essential_ById { id: String },
    #[allow(non_camel_case_types)]
    #[serde(rename = "adjustmentsToSpending.extraSpending.essential.total")]
    AdjustmentsToSpending_ExtraSpending_Essential_Total,
    #[allow(non_camel_case_types)]
    #[serde(rename = "adjustmentsToSpending.extraSpending.discretionary.byId")]
    AdjustmentsToSpending_ExtraSpending_Discretionary_ById { id: String },
    #[allow(non_camel_case_types)]
    #[serde(rename = "adjustmentsToSpending.extraSpending.discretionary.total")]
    AdjustmentsToSpending_ExtraSpending_Discretionary_Total,
    #[allow(non_camel_case_types)]
    #[serde(rename = "risk.tpawAndSPAW.lmp")]
    Risk_TPAWAndSPAW_LMP,
}

#[wasm_bindgen]
impl PlanParamsProcessedJS {
    pub fn without_arrays(&self) -> String {
        serde_json::to_string(&self.without_arrays).unwrap()
    }
    pub fn array(&self, array_name: PlanParamsProcessedJSArrayName) -> js_sys::Float64Array {
        match array_name {
            PlanParamsProcessedJSArrayName::HistoricalMonthlyReturnsAdjusted_Stocks_LogSeries => {
                vec_f64_js_view(
                    &self
                        .without_arrays
                        .historical_monthly_returns_adjusted
                        .stocks
                        .log_series,
                )
            }
            PlanParamsProcessedJSArrayName::HistoricalMonthlyReturnsAdjusted_Bonds_LogSeries => {
                vec_f64_js_view(
                    &self
                        .without_arrays
                        .historical_monthly_returns_adjusted
                        .bonds
                        .log_series,
                )
            }
            PlanParamsProcessedJSArrayName::Wealth_Total=> vec_f64_js_view(
                &&self
                    .without_arrays
                    .by_month
                    .wealth
                    .total
            ),
            PlanParamsProcessedJSArrayName::Wealth_FutureSavings_ById { id } => vec_f64_js_view(
                &self
                    .without_arrays
                    .by_month
                    .wealth
                    .future_savings
                    .by_id
                    .iter()
                    .find(|&x| x.id == id)
                    .unwrap()
                    .values,
            ),
            PlanParamsProcessedJSArrayName::Wealth_FutureSavings_Total => vec_f64_js_view(
                &self
                    .without_arrays
                    .by_month
                    .wealth
                    .future_savings
                    .total
            ),
            PlanParamsProcessedJSArrayName::Wealth_IncomeDuringRetirement_ById { id } => {
                vec_f64_js_view(
                    &self
                        .without_arrays
                        .by_month
                        .wealth
                        .income_during_retirement
                        .by_id
                        .iter()
                        .find(|&x| x.id == id)
                        .unwrap()
                        .values,
                )
            }
            PlanParamsProcessedJSArrayName::Wealth_IncomeDuringRetirement_Total  => {
                vec_f64_js_view(
                    &self
                        .without_arrays
                        .by_month
                        .wealth
                        .income_during_retirement
                        .total
                )
            }
            PlanParamsProcessedJSArrayName::AdjustmentsToSpending_ExtraSpending_Essential_ById {
                id,
            } => vec_f64_js_view(
                &self
                    .without_arrays
                    .by_month
                    .adjustments_to_spending
                    .extra_spending
                    .essential
                    .by_id
                    .iter()
                    .find(|&x| x.id == id)
                    .unwrap()
                    .values,
            ),
            PlanParamsProcessedJSArrayName::AdjustmentsToSpending_ExtraSpending_Essential_Total => vec_f64_js_view(
                &self
                    .without_arrays
                    .by_month
                    .adjustments_to_spending
                    .extra_spending
                    .essential
                    .total
            ),
            PlanParamsProcessedJSArrayName::AdjustmentsToSpending_ExtraSpending_Discretionary_ById {
                id,
            } => vec_f64_js_view(
                &self
                    .without_arrays
                    .by_month
                    .adjustments_to_spending
                    .extra_spending
                    .discretionary
                    .by_id
                    .iter()
                    .find(|&x| x.id == id)
                    .unwrap()
                    .values,
            ),
            PlanParamsProcessedJSArrayName::AdjustmentsToSpending_ExtraSpending_Discretionary_Total=> vec_f64_js_view(
                &&self
                    .without_arrays
                    .by_month
                    .adjustments_to_spending
                    .extra_spending
                    .discretionary
                    .total
            ),
            PlanParamsProcessedJSArrayName::Risk_TPAWAndSPAW_LMP => {
                vec_f64_js_view(&&self.without_arrays.by_month.risk.tpaw_and_spaw.lmp)
            }
        }
    }
}

pub fn process_plan_params(
    plan_params_norm: &PlanParamsRust,
    market_data: &DataForMarketBasedPlanParamValues,
) -> PlanParamsProcessedJS {
    let expected_returns_for_planning = process_expected_returns_for_planning(
        &plan_params_norm.advanced.expected_returns_for_planning,
        &plan_params_norm.advanced.sampling,
        &plan_params_norm
            .advanced
            .historical_monthly_log_returns_adjustment
            .standard_deviation,
        market_data,
    );

    let historical_monthly_returns_adjusted = process_historical_monthly_log_returns_adjustment(
        &expected_returns_for_planning,
        market_data,
        plan_params_norm
            .advanced
            .historical_monthly_log_returns_adjustment
            .override_to_fixed_for_testing,
    );

    let inflation =
        process_annual_inflation(&plan_params_norm.advanced.annual_inflation, market_data);

    let by_month = process_by_month_params(
        &plan_params_norm,
        inflation.monthly,
        // &expected_returns_for_planning.monthly_non_log_for_simulation,
    );
    PlanParamsProcessedJS {
        without_arrays: PlanParamsProcessed {
            expected_returns_for_planning,
            historical_monthly_returns_adjusted,
            inflation,
            by_month,
        },
    }
}
