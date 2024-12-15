use crate::{
    historical_monthly_returns::{
        data::average_annual_real_earnings_for_sp500_for_10_years::AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS,
        HISTORICAL_MONTHLY_RETURNS,
    },
    ny_tz::get_conservative_market_closes_between_timestamps_ms,
    wire::{wire_simulation_args, NoMessage},
};
use chrono::Datelike;

use crate::market_data::{
    download_daily_market_data_series::download_daily_market_data_series,
    market_data_defs::{DailyMarketDataForPresets, VTAndBNDData},
};

use super::market_data_defs::{
    DailyMarketDataForPresets_BondRates, DailyMarketDataForPresets_Inflation,
    DailyMarketDataForPresets_SP500, MarketDataSeriesForPortfolioBalanceEstimation,
    MarketDataSeriesForSimulation, VTAndBNDData_PercentageChangeFromLastClose,
};

pub struct DownloadedData {
    pub daily_market_data_for_presets_series: Vec<DailyMarketDataForPresets>,
    pub vt_and_bnd_series: Vec<VTAndBNDData>,
}


pub async fn dowload_data() ->  DownloadedData {
    let (daily_market_data_for_presets_series, vt_and_bnd_series) =
        download_daily_market_data_series().await;
    DownloadedData {
        daily_market_data_for_presets_series,
        vt_and_bnd_series,
    }
}

pub enum DailyMarketSeriesSrc {
    Live,
    SyntheticLiveRepeated,
    SyntheticConstant {
        annual_percentage_change_vt: f64,
        annual_percentage_change_bnd: f64,
    },
}

impl From<wire_simulation_args::MarketDailySeriesSrc> for DailyMarketSeriesSrc {
    fn from(value: wire_simulation_args::MarketDailySeriesSrc) -> Self {
        match value {
            wire_simulation_args::MarketDailySeriesSrc::Live(NoMessage {}) => Self::Live,
            wire_simulation_args::MarketDailySeriesSrc::SyntheticLiveRepeated(NoMessage {}) => {
                Self::SyntheticLiveRepeated
            }
            wire_simulation_args::MarketDailySeriesSrc::SyntheticConstant(constant) => {
                Self::SyntheticConstant {
                    annual_percentage_change_vt: constant.annual_percentage_change_vt,
                    annual_percentage_change_bnd: constant.annual_percentage_change_bnd,
                }
            }
        }
    }
}

pub async fn get_market_data_series_for_simulation_synthetic_override(
    src: &DailyMarketSeriesSrc,
    simulation_timestamp_ms: i64,
    live: &Vec<DailyMarketDataForPresets>,
) -> Option<Vec<DailyMarketDataForPresets>> {
    
    match src {
        DailyMarketSeriesSrc::Live => None,
        DailyMarketSeriesSrc::SyntheticLiveRepeated => Some(
            live.iter()
                .map(|x| x.clone())
                .chain(
                    get_conservative_market_closes_between_timestamps_ms(
                        live.last().unwrap().closing_time_ms..=simulation_timestamp_ms,
                    )
                    .iter()
                    .map(|x| x.timestamp_millis())
                    .enumerate()
                    .map(|(i, x)| {
                        let live_src = &live[i];
                        DailyMarketDataForPresets {
                            closing_time_ms: x,
                            inflation: DailyMarketDataForPresets_Inflation {
                                closing_time_ms: x,
                                value: live_src.inflation.value,
                            },
                            sp500: DailyMarketDataForPresets_SP500 {
                                closing_time_ms: x,
                                value: live_src.sp500.value,
                            },
                            bond_rates: DailyMarketDataForPresets_BondRates {
                                closing_time_ms: x,
                                five_year: live_src.bond_rates.five_year,
                                seven_year: live_src.bond_rates.seven_year,
                                ten_year: live_src.bond_rates.ten_year,
                                twenty_year: live_src.bond_rates.twenty_year,
                                thirty_year: live_src.bond_rates.thirty_year,
                            },
                        }
                    }),
                )
                .collect(),
        ),

        DailyMarketSeriesSrc::SyntheticConstant { .. } => Some(
            get_conservative_market_closes_between_timestamps_ms(
                live[0].closing_time_ms..=simulation_timestamp_ms,
            )
            .iter()
            .map(|x| x.timestamp_millis())
            .map(|x| DailyMarketDataForPresets {
                closing_time_ms: x,
                inflation: DailyMarketDataForPresets_Inflation {
                    closing_time_ms: x,
                    value: 0.02,
                },
                sp500: DailyMarketDataForPresets_SP500 {
                    closing_time_ms: x,
                    value: 4000.0,
                },
                bond_rates: DailyMarketDataForPresets_BondRates {
                    closing_time_ms: x,
                    five_year: 0.017,
                    seven_year: 0.016,
                    ten_year: 0.015,
                    twenty_year: 0.014,
                    thirty_year: 0.013,
                },
            })
            .collect(),
        ),
    }
}

pub async fn get_market_data_series_for_simulation<'a>(
    synthetic_override: Option<&'a Vec<DailyMarketDataForPresets>>,
    downloaded_data: &'a DownloadedData,
) -> MarketDataSeriesForSimulation<'a> {
    // let downloaded_data = get_dowloaded_data().await;
    MarketDataSeriesForSimulation {
        daily_market_data_for_presets_series: synthetic_override
            .unwrap_or(&downloaded_data.daily_market_data_for_presets_series),
        historical_monthly_returns_info_series: &HISTORICAL_MONTHLY_RETURNS,
        average_annual_real_earnings_for_sp500_for_10_years_series:
            &AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS,
    }
}

pub async fn get_daily_market_data_for_portfolio_balance_estimation_synthetic_override(
    src: &DailyMarketSeriesSrc,
    simulation_timestamp_ms: i64,
    live: &Vec<VTAndBNDData>,
) -> Option<Vec<VTAndBNDData>> {
    match src {
        DailyMarketSeriesSrc::Live => None,
        DailyMarketSeriesSrc::SyntheticLiveRepeated => Some(
            live.iter()
                .map(|x| x.clone())
                .chain(
                    get_conservative_market_closes_between_timestamps_ms(
                        live.last().unwrap().closing_time_ms..=simulation_timestamp_ms,
                    )
                    .iter()
                    .map(|x| x.timestamp_millis())
                    .enumerate()
                    .map(|(i, x)| {
                        let live_src = &live[i];
                        VTAndBNDData {
                            closing_time_ms: x,
                            percentage_change_from_last_close: live_src
                                .percentage_change_from_last_close
                                .clone(),
                        }
                    }),
                )
                .collect(),
        ),
        DailyMarketSeriesSrc::SyntheticConstant {
            annual_percentage_change_vt,
            annual_percentage_change_bnd,
        } => Some(
            get_conservative_market_closes_between_timestamps_ms(
                live[0].closing_time_ms..=simulation_timestamp_ms,
            )
            .iter()
            .map(|x| {
                let days_since_last_market_close = if x.weekday() == chrono::Weekday::Mon {
                    3
                } else {
                    1
                };
                let from_annual = |annual_rate: f64| {
                    (1.0 + annual_rate).powf(days_since_last_market_close as f64 / 365.0) - 1.0
                };
                VTAndBNDData {
                    closing_time_ms: x.timestamp_millis(),
                    percentage_change_from_last_close: VTAndBNDData_PercentageChangeFromLastClose {
                        vt: from_annual(*annual_percentage_change_vt),
                        bnd: from_annual(*annual_percentage_change_bnd),
                    },
                }
            })
            .collect(),
        ),
    }
}

pub async fn get_market_data_series_for_portfolio_balance_estimation<'a>(
    synthetic_override: Option<&'a Vec<VTAndBNDData>>,
    downloaded_data: &'a DownloadedData,
) -> MarketDataSeriesForPortfolioBalanceEstimation<'a> {
    MarketDataSeriesForPortfolioBalanceEstimation {
        vt_and_bnd_series: synthetic_override.unwrap_or(&downloaded_data.vt_and_bnd_series),
    }
}

