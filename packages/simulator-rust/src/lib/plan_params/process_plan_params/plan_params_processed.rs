use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::shared_types::StocksAndBonds;

use super::{
    process_annual_inflation::InflationProcessed, process_by_month_params::ProcessedByMonthParams,
    process_market_data::MarketDataProcessed,
    process_returns_stats_for_planning::ReturnsStatsForPlanningProcessed,
    process_risk::RiskProcessed,
};
use crate::historical_monthly_returns::HistoricalMonthlyLogReturnsAdjustedInfo;

#[wasm_bindgen]
pub struct StocksAndBondsHistoricalMonthlyLogReturnsAdjustedInfo {
    #[wasm_bindgen(skip)]
    pub stocks: HistoricalMonthlyLogReturnsAdjustedInfo,
    #[wasm_bindgen(skip)]
    pub bonds: HistoricalMonthlyLogReturnsAdjustedInfo,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct PlanParamsProcessed {
    pub market_data: MarketDataProcessed,
    pub returns_stats_for_planning: ReturnsStatsForPlanningProcessed,
    pub historical_returns_adjusted: StocksAndBonds<HistoricalMonthlyLogReturnsAdjustedInfo>,
    pub risk: RiskProcessed,
    pub inflation: InflationProcessed,
    pub by_month: ProcessedByMonthParams,
}
