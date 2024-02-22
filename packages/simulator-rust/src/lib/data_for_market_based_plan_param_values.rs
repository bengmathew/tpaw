use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;


#[derive(Serialize, Deserialize, Tsify, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct SP500 {
    #[serde(rename = "closingTime")]
    pub closing_time_ms: i64,
    pub value: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct BondRates {
    #[serde(rename = "closingTime")]
    pub closing_time_ms: i64,
    pub five_year: f64,
    pub seven_year: f64,
    pub ten_year: f64,
    pub twenty_year: f64,
    pub thirty_year: f64,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct Inflation {
    pub value: f64,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct DataForMarketBasedPlanParamValues {
    pub sp500: SP500,
    pub bond_rates: BondRates,
    pub inflation: Inflation,
    #[serde(rename = "timestampMSForHistoricalReturns")]
    pub timestamp_ms_for_historical_returns: i64,
}
