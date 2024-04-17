use crate::data_for_market_based_plan_param_values::BondRates;
use crate::data_for_market_based_plan_param_values::SP500;
use crate::historical_monthly_returns::data::average_annual_real_earnings_for_sp500_for_10_years::AverageAnnualRealEarningsForSP500For10Years;
use crate::historical_monthly_returns::get_historical_monthly_returns_info;
use crate::shared_types::SimpleRange;
use crate::shared_types::YearAndMonth;
use crate::{
    data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues,
    historical_monthly_returns::data::{
        average_annual_real_earnings_for_sp500_for_10_years::AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS,
        CAPEBasedRegressionResults,
    },
    return_series::Mean,
    round::RoundP,
    shared_types::StocksAndBonds,
};
use serde::Deserialize;
use serde::Serialize;
use tsify::Tsify;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Copy, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct MarketDataProcessedStocks {
    pub sp500: SP500,
    #[serde(rename = "averageRealEarningsForSP500For10Years")]
    pub average_real_earnings_for_sp500_10_years: AverageAnnualRealEarningsForSP500For10Years,
    pub cape_not_rounded: f64,
    #[serde(rename = "oneOverCAPENotRounded")]
    pub one_over_cape_not_rounded: f64,
    #[serde(rename = "oneOverCAPERounded")]
    pub one_over_cape_rounded: f64,
    pub empirical_annual_non_log_regressions_stocks: CAPEBasedRegressionResults,
    pub regression_prediction: f64,
    pub conservative_estimate: f64,
    pub historical: f64,
}

#[derive(Serialize, Deserialize, Copy, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct MarketDataProcessedBonds {
    pub bond_rates: BondRates,
    pub tips_yield_20_year: f64,
    pub historical: f64,
}

#[derive(Serialize, Deserialize, Copy, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct MarketDataProcessedExpectedReturns {
    pub stocks: MarketDataProcessedStocks,
    pub bonds: MarketDataProcessedBonds,
}
#[derive(Serialize, Deserialize, Copy, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct MarketDataProcessedInflation {
    pub suggested_annual: f64,
    #[serde(rename = "closingTime")]
    pub closing_time_ms: i64,
}

#[derive(Serialize, Deserialize, Copy, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct MarketDataProcessed {
    #[serde(rename = "lastUpdatedAtTimestamp")]
    pub last_updated_at_timestamp_ms: i64,
    #[serde(rename = "timestampForMarketData")]
    pub timestamp_ms_for_market_data: i64,
    pub historical_returns_month_range: SimpleRange<YearAndMonth>,
    pub expected_returns: MarketDataProcessedExpectedReturns,
    pub inflation: MarketDataProcessedInflation,
}

pub fn process_market_data(market_data: &DataForMarketBasedPlanParamValues) -> MarketDataProcessed {
    let block_size_effective = None;
    let log_volatility_scale_effective = StocksAndBonds {
        stocks: 1.0,
        bonds: 1.0,
    };

    let historical_monthly_returns_info =
        get_historical_monthly_returns_info(market_data.timestamp_ms_for_market_data);
    let historical_monthly_returns = &historical_monthly_returns_info.returns;
    let expected_returns = {
        let stocks = {
            let sp500 = &market_data.sp500;
            let average_earning = AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS
                .iter()
                .rev()
                .find(|x| market_data.timestamp_ms_for_market_data >= x.added_date_ms)
                .unwrap();
            let one_over_cape = average_earning.value / sp500.value;
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
            MarketDataProcessedStocks {
                sp500: *sp500,
                average_real_earnings_for_sp500_10_years: *average_earning,
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
            let bond_rates = BondRates {
                closing_time_ms: market_data.bond_rates.closing_time_ms,
                five_year: market_data.bond_rates.five_year.round_p(3),
                seven_year: market_data.bond_rates.seven_year.round_p(3),
                ten_year: market_data.bond_rates.ten_year.round_p(3),
                twenty_year: market_data.bond_rates.twenty_year.round_p(3),
                thirty_year: market_data.bond_rates.thirty_year.round_p(3),
            };
            MarketDataProcessedBonds {
                bond_rates,
                historical: historical_monthly_returns
                    .bonds
                    .get_empirical_annual_non_log_from_log_monthly_expected_value(
                        historical_monthly_returns.bonds.log.stats.mean,
                        block_size_effective,
                        log_volatility_scale_effective.bonds,
                    )
                    .value,
                tips_yield_20_year: bond_rates.twenty_year,
            }
        };
        MarketDataProcessedExpectedReturns { stocks, bonds }
    };
    let inflation = MarketDataProcessedInflation {
        suggested_annual: market_data.inflation.value.round_p(3),
        closing_time_ms: market_data.inflation.closing_time_ms,
    };
    MarketDataProcessed {
        last_updated_at_timestamp_ms: market_data.closing_time_ms,
        timestamp_ms_for_market_data: market_data.timestamp_ms_for_market_data,
        historical_returns_month_range: historical_monthly_returns_info.month_range.clone(),
        expected_returns,
        inflation,
    }
}
