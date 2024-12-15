#![allow(non_camel_case_types)]

use crate::{
    fget_item_at_or_before_key::FGetItemAtOrBeforeKey,
    historical_monthly_returns::{
        data::average_annual_real_earnings_for_sp500_for_10_years::AverageAnnualRealEarningsForSP500For10Years,
        HistoricalReturnsInfo,
    },
    wire::{
        WireDailyMarketDataForPresets, WireDailyMarketDataForPresetsBondRates,
        WireDailyMarketDataForPresetsInflation, WireDailyMarketDataForPresetsSp500,
    },
};
use serde::{Deserialize, Serialize};

pub const MARKET_DATA_FOR_PRESETS_DIRECTORY: &str = "v2/for_presets";
pub const MARKET_DATA_VT_AND_BND_DIRECTORY: &str = "v2/vt_and_bnd";

// -------------------------
// DailyMarketDataForPresets
// -------------------------
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct DailyMarketDataForPresets_Inflation {
    pub closing_time_ms: i64,
    pub value: f64,
}

impl From<DailyMarketDataForPresets_Inflation> for WireDailyMarketDataForPresetsInflation {
    fn from(value: DailyMarketDataForPresets_Inflation) -> Self {
        Self {
            closing_timestamp_ms: value.closing_time_ms,
            value: value.value,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct DailyMarketDataForPresets_SP500 {
    pub closing_time_ms: i64,
    pub value: f64,
}

impl From<DailyMarketDataForPresets_SP500> for WireDailyMarketDataForPresetsSp500 {
    fn from(value: DailyMarketDataForPresets_SP500) -> Self {
        Self {
            closing_timestamp_ms: value.closing_time_ms,
            value: value.value,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct DailyMarketDataForPresets_BondRates {
    pub closing_time_ms: i64,
    pub five_year: f64,
    pub seven_year: f64,
    pub ten_year: f64,
    pub twenty_year: f64,
    pub thirty_year: f64,
}

impl From<DailyMarketDataForPresets_BondRates> for WireDailyMarketDataForPresetsBondRates {
    fn from(value: DailyMarketDataForPresets_BondRates) -> Self {
        Self {
            closing_timestamp_ms: value.closing_time_ms,
            five_year: value.five_year,
            seven_year: value.seven_year,
            ten_year: value.ten_year,
            twenty_year: value.twenty_year,
            thirty_year: value.thirty_year,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct DailyMarketDataForPresets {
    pub closing_time_ms: i64,
    pub inflation: DailyMarketDataForPresets_Inflation,
    pub sp500: DailyMarketDataForPresets_SP500,
    pub bond_rates: DailyMarketDataForPresets_BondRates,
}

impl DailyMarketDataForPresets {
    pub fn new_for_testing(closing_time_ms: i64) -> Self {
        Self {
            closing_time_ms,
            inflation: DailyMarketDataForPresets_Inflation {
                closing_time_ms,
                value: 0.02,
            },
            sp500: DailyMarketDataForPresets_SP500 {
                closing_time_ms,
                value: 0.01,
            },
            bond_rates: DailyMarketDataForPresets_BondRates {
                closing_time_ms,
                five_year: 0.01,
                seven_year: 0.01,
                ten_year: 0.01,
                twenty_year: 0.01,
                thirty_year: 0.01,
            },
        }
    }
}

impl From<DailyMarketDataForPresets> for WireDailyMarketDataForPresets {
    fn from(value: DailyMarketDataForPresets) -> Self {
        Self {
            closing_timestamp_ms: value.closing_time_ms,
            inflation: value.inflation.into(),
            sp500: value.sp500.into(),
            bond_rates: value.bond_rates.into(),
        }
    }
}

// -----------------------
// MarketDataForSimulation
// -----------------------
pub struct MarketDataSeriesForSimulation<'a> {
    pub daily_market_data_for_presets_series: &'a Vec<DailyMarketDataForPresets>,
    pub historical_monthly_returns_info_series: &'a Vec<HistoricalReturnsInfo>,
    pub average_annual_real_earnings_for_sp500_for_10_years_series:
        &'a [AverageAnnualRealEarningsForSP500For10Years],
}
impl<'a> MarketDataSeriesForSimulation<'a> {
    pub fn for_timestamp(
        &self,
        timestamp_ms_for_market_data: i64,
    ) -> MarketDataAtTimestampForSimulation<'a> {
        MarketDataAtTimestampForSimulation {
            daily_market_data_for_presets: &self
                .daily_market_data_for_presets_series
                .fget_item_at_or_before_key(timestamp_ms_for_market_data, |x| x.closing_time_ms),
            historical_monthly_returns_info: &self
                .historical_monthly_returns_info_series
                .fget_item_at_or_before_key(timestamp_ms_for_market_data, |x| x.timestamp_ms),
            average_annual_real_earnings_for_sp500_for_10_years: &self
                .average_annual_real_earnings_for_sp500_for_10_years_series
                .fget_item_at_or_before_key(timestamp_ms_for_market_data, |x| x.added_date_ms),
        }
    }
}

pub struct MarketDataAtTimestampForSimulation<'a> {
    pub daily_market_data_for_presets: &'a DailyMarketDataForPresets,
    pub historical_monthly_returns_info: &'a HistoricalReturnsInfo,
    pub average_annual_real_earnings_for_sp500_for_10_years:
        &'a AverageAnnualRealEarningsForSP500For10Years,
}

// ------------
// VTAndBNDData
// ------------
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct VTAndBNDData_PercentageChangeFromLastClose {
    pub vt: f64,
    pub bnd: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct VTAndBNDData {
    pub closing_time_ms: i64,
    pub percentage_change_from_last_close: VTAndBNDData_PercentageChangeFromLastClose,
}

impl PartialEq for VTAndBNDData_PercentageChangeFromLastClose {
    fn eq(&self, other: &Self) -> bool {
        self.vt - other.vt < 0.000000000000001 && self.bnd - other.bnd < 0.000000000000001
    }
}

// ----------------------------------------------
// MarketDataForCurrentPortfolioBalanceEstimation
// ----------------------------------------------
pub struct MarketDataSeriesForPortfolioBalanceEstimation<'a> {
    pub vt_and_bnd_series: &'a Vec<VTAndBNDData>,
}
