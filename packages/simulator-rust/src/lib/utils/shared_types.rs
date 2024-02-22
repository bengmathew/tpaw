use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LogAndNonLog<T> {
    pub log: T,
    pub non_log: T,
}

#[derive(Serialize, Deserialize, Copy, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct StocksAndBonds<T> {
    pub stocks: T,
    pub bonds: T,
}


#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct MonthAndStocks {
    pub month: i64,
    pub stocks: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Stocks {
    pub stocks: f64,
}


#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Monthly<T> {
    pub monthly: T,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Log<T> {
    pub log: T,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseAndLog<T> {
    pub base: T,
    pub log: T,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct SlopeAndIntercept {
    pub slope: f64,
    pub intercept: f64,
}

impl SlopeAndIntercept {
    pub fn y(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

pub enum StocksOrBonds {
    Stocks,
    Bonds,
}

#[derive(Serialize, Deserialize, Clone, Copy, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct YearAndMonth {
    pub year: u16,
    pub month: u8,
}

#[derive(Serialize, Deserialize, Clone, Copy, Tsify, PartialEq, Debug, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SimpleRange<T> {
    pub start: T,
    pub end: T,
}
