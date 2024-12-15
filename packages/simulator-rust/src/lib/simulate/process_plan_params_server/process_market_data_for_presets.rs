#![allow(non_camel_case_types)]

use crate::{
    historical_monthly_returns::data::{
        average_annual_real_earnings_for_sp500_for_10_years::AverageAnnualRealEarningsForSP500For10Years,
        CAPEBasedRegressionResults,
    },
    market_data::market_data_defs::MarketDataAtTimestampForSimulation,
    utils::{return_series::Mean, round::RoundP, shared_types::StocksAndBonds},
    wire::{
        WireMarketDataForPresetsProcessed, WireMarketDataForPresetsProcessedBonds,
        WireMarketDataForPresetsProcessedExpectedReturns,
        WireMarketDataForPresetsProcessedInflation, WireMarketDataForPresetsProcessedSource,
        WireMarketDataForPresetsProcessedStocks,
    },
};

use crate::market_data::market_data_defs::{
    DailyMarketDataForPresets, DailyMarketDataForPresets_BondRates,
    DailyMarketDataForPresets_Inflation, DailyMarketDataForPresets_SP500,
};

#[derive(Clone)]
pub struct MarketDataForPresetsProcessed_Source {
    pub daily_market_data: DailyMarketDataForPresets,
    pub average_annual_real_earnings_for_sp500_for_10_years:
        AverageAnnualRealEarningsForSP500For10Years,
}

impl From<MarketDataForPresetsProcessed_Source> for WireMarketDataForPresetsProcessedSource {
    fn from(value: MarketDataForPresetsProcessed_Source) -> Self {
        Self {
            daily_market_data: value.daily_market_data.into(),
            average_annual_real_earnings_for_sp500_for_10_years: value
                .average_annual_real_earnings_for_sp500_for_10_years
                .into(),
        }
    }
}

#[derive(Clone)]
pub struct MarketDataForPresetsProcessed_Stocks {
    pub cape_not_rounded: f64,
    pub one_over_cape_not_rounded: f64,
    pub one_over_cape_rounded: f64,
    pub empirical_annual_non_log_regressions_stocks: CAPEBasedRegressionResults,
    pub regression_prediction: f64,
    pub conservative_estimate: f64,
    pub historical: f64,
}

impl From<MarketDataForPresetsProcessed_Stocks> for WireMarketDataForPresetsProcessedStocks {
    fn from(value: MarketDataForPresetsProcessed_Stocks) -> Self {
        Self {
            cape_not_rounded: value.cape_not_rounded,
            one_over_cape_not_rounded: value.one_over_cape_not_rounded,
            one_over_cape_rounded: value.one_over_cape_rounded,
            empirical_annual_non_log_regressions_stocks: value
                .empirical_annual_non_log_regressions_stocks
                .into(),
            regression_prediction: value.regression_prediction,
            conservative_estimate: value.conservative_estimate,
            historical: value.historical,
        }
    }
}

#[derive(Clone)]
pub struct MarketDataForPresetsProcessed_Bonds {
    pub tips_yield_20_year: f64,
    pub historical: f64,
}

impl From<MarketDataForPresetsProcessed_Bonds> for WireMarketDataForPresetsProcessedBonds {
    fn from(value: MarketDataForPresetsProcessed_Bonds) -> Self {
        Self {
            tips_yield_20_year: value.tips_yield_20_year,
            historical: value.historical,
        }
    }
}

#[derive(Clone)]
pub struct MarketDataForPresetsProcessed_ExpectedReturns {
    pub stocks: MarketDataForPresetsProcessed_Stocks,
    pub bonds: MarketDataForPresetsProcessed_Bonds,
}

impl From<MarketDataForPresetsProcessed_ExpectedReturns>
    for WireMarketDataForPresetsProcessedExpectedReturns
{
    fn from(value: MarketDataForPresetsProcessed_ExpectedReturns) -> Self {
        Self {
            stocks: value.stocks.into(),
            bonds: value.bonds.into(),
        }
    }
}

#[derive(Clone)]
pub struct MarketDataForPresetsProcessed_Inflation {
    pub suggested_annual: f64,
}

impl From<MarketDataForPresetsProcessed_Inflation> for WireMarketDataForPresetsProcessedInflation {
    fn from(value: MarketDataForPresetsProcessed_Inflation) -> Self {
        Self {
            suggested_annual: value.suggested_annual,
        }
    }
}

#[derive(Clone)]
pub struct MarketDataForPresetsProcessed {
    pub source_rounded: MarketDataForPresetsProcessed_Source,
    pub expected_returns: MarketDataForPresetsProcessed_ExpectedReturns,
    pub inflation: MarketDataForPresetsProcessed_Inflation,
}

impl From<MarketDataForPresetsProcessed> for WireMarketDataForPresetsProcessed {
    fn from(value: MarketDataForPresetsProcessed) -> Self {
        Self {
            source_rounded: value.source_rounded.into(),
            expected_returns: value.expected_returns.into(),
            inflation: value.inflation.into(),
        }
    }
}

pub fn process_market_data_for_presets(
    market_data_at_timestamp_for_simulation: &MarketDataAtTimestampForSimulation,
) -> MarketDataForPresetsProcessed {
    let MarketDataAtTimestampForSimulation {
        daily_market_data_for_presets,
        historical_monthly_returns_info,
        average_annual_real_earnings_for_sp500_for_10_years,
    } = market_data_at_timestamp_for_simulation;

    let block_size_effective = None;
    let log_volatility_scale_effective = StocksAndBonds {
        stocks: 1.0,
        bonds: 1.0,
    };

    let source_rounded = {
        let daily_data_rounded = DailyMarketDataForPresets {
            closing_time_ms: daily_market_data_for_presets.closing_time_ms,
            sp500: DailyMarketDataForPresets_SP500 {
                closing_time_ms: daily_market_data_for_presets.sp500.closing_time_ms,
                // No rounding needed for SP500
                value: daily_market_data_for_presets.sp500.value,
            },
            bond_rates: DailyMarketDataForPresets_BondRates {
                closing_time_ms: daily_market_data_for_presets.bond_rates.closing_time_ms,
                five_year: daily_market_data_for_presets
                    .bond_rates
                    .five_year
                    .round_p(3),
                seven_year: daily_market_data_for_presets
                    .bond_rates
                    .seven_year
                    .round_p(3),
                ten_year: daily_market_data_for_presets.bond_rates.ten_year.round_p(3),
                twenty_year: daily_market_data_for_presets
                    .bond_rates
                    .twenty_year
                    .round_p(3),
                thirty_year: daily_market_data_for_presets
                    .bond_rates
                    .thirty_year
                    .round_p(3),
            },
            inflation: DailyMarketDataForPresets_Inflation {
                closing_time_ms: daily_market_data_for_presets.inflation.closing_time_ms,
                value: daily_market_data_for_presets.inflation.value.round_p(3),
            },
        };

        MarketDataForPresetsProcessed_Source {
            daily_market_data: daily_data_rounded,
            // No rounding needed for this
            average_annual_real_earnings_for_sp500_for_10_years:
                **average_annual_real_earnings_for_sp500_for_10_years,
        }
    };

    let historical_monthly_returns = &historical_monthly_returns_info.returns;
    let expected_returns = {
        let stocks = {
            let sp500 = &source_rounded.daily_market_data.sp500;
            let one_over_cape = source_rounded
                .average_annual_real_earnings_for_sp500_for_10_years
                .value
                / sp500.value;
            let empirical_annual_non_log_regressions_stocks = historical_monthly_returns
                .stocks
                .annual_log_mean_from_one_over_cape_regression_info
                .y(one_over_cape.ln_1p())
                .map(|annual_log_mean| {
                    historical_monthly_returns
                        .stocks
                        .get_empirical_annual_non_log_from_log_monthly_expected_value(
                            annual_log_mean / 12.0,
                            block_size_effective,
                            log_volatility_scale_effective.stocks,
                        )
                        .value
                });
            let regression_prediction = empirical_annual_non_log_regressions_stocks
                .iter()
                .mean()
                .round_p(3);
            let conservative_estimate = {
                let mut with_one_over_cape: Vec<f64> = std::iter::once(one_over_cape)
                    .chain(empirical_annual_non_log_regressions_stocks.iter())
                    .collect();
                with_one_over_cape.sort_by(|a, b| a.partial_cmp(b).unwrap());
                with_one_over_cape.iter().take(4).mean().round_p(3)
            };
            MarketDataForPresetsProcessed_Stocks {
                cape_not_rounded: 1.0 / one_over_cape,
                one_over_cape_not_rounded: one_over_cape,
                one_over_cape_rounded: one_over_cape.round_p(3),
                empirical_annual_non_log_regressions_stocks,
                regression_prediction,
                conservative_estimate,
                historical: historical_monthly_returns
                    .stocks
                    .get_empirical_annual_non_log_from_log_monthly_expected_value(
                        historical_monthly_returns.stocks.log.stats.mean,
                        block_size_effective,
                        log_volatility_scale_effective.stocks,
                    )
                    .value,
            }
        };
        let bonds = {
            MarketDataForPresetsProcessed_Bonds {
                historical: historical_monthly_returns
                    .bonds
                    .get_empirical_annual_non_log_from_log_monthly_expected_value(
                        historical_monthly_returns.bonds.log.stats.mean,
                        block_size_effective,
                        log_volatility_scale_effective.bonds,
                    )
                    .value,
                tips_yield_20_year: source_rounded.daily_market_data.bond_rates.twenty_year,
            }
        };
        MarketDataForPresetsProcessed_ExpectedReturns { stocks, bonds }
    };

    let inflation = MarketDataForPresetsProcessed_Inflation {
        suggested_annual: source_rounded.daily_market_data.inflation.value,
    };
    MarketDataForPresetsProcessed {
        source_rounded,
        expected_returns,
        inflation,
    }
}
