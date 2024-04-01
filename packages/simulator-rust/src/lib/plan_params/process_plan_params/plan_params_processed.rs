use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::{shared_types::StocksAndBonds, vec_f64_js_view};

use super::process_expected_returns_for_planning::ExpectedReturnsForPlanningProcessed;
use super::{
    process_annual_inflation::InflationProcessed, process_by_month_params::ProcessedByMonthParams,
};
use crate::historical_monthly_returns::{
    HistoricalMonthlyLogReturnsAdjustedInfo, HistoricalMonthlyLogReturnsAdjustedStats,
};

#[wasm_bindgen]
pub struct StocksAndBondsHistoricalMonthlyLogReturnsAdjustedInfo {
    #[wasm_bindgen(skip)]
    pub stocks: HistoricalMonthlyLogReturnsAdjustedInfo,
    #[wasm_bindgen(skip)]
    pub bonds: HistoricalMonthlyLogReturnsAdjustedInfo,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct PlanParamsProcessed {
    pub expected_returns_for_planning: ExpectedReturnsForPlanningProcessed,
    pub historical_monthly_returns_adjusted:
        StocksAndBonds<HistoricalMonthlyLogReturnsAdjustedInfo>,
    pub inflation: InflationProcessed,
    pub by_month: ProcessedByMonthParams,
}
