use crate::expected_value_of_returns::annual_non_log_to_monthly_non_log_return_rate;
use crate::expected_value_of_returns::EmpiricalAnnualNonLogExpectedReturnInfo;
use crate::historical_monthly_returns::get_historical_monthly_returns_info;
use crate::plan_params;
use crate::plan_params::ExpectedReturnsForPlanningCustomStocksBase;
use crate::{
    shared_types::StocksAndBonds,
};
use serde::Deserialize;
use serde::Serialize;
use tsify::declare;
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use super::process_market_data::MarketDataProcessed;

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ReturnsStatsForPlanningProcessedPart {
    pub empirical_annual_non_log_expected_return_info: EmpiricalAnnualNonLogExpectedReturnInfo,
    pub empirical_annual_log_variance: f64,
    pub empirical_monthly_non_log_expected_return: f64,
}

#[declare]
pub type ReturnsStatsForPlanningProcessed = StocksAndBonds<ReturnsStatsForPlanningProcessedPart>;

pub fn process_returns_stats_for_planning(
    sampling: &plan_params::Sampling,
    returns_stats_for_planning: &plan_params::ReturnsStatsForPlanning,
    market_data: &MarketDataProcessed,
) -> ReturnsStatsForPlanningProcessed {
    let historical_returns_info =
        get_historical_monthly_returns_info(market_data.timestamp_ms_for_market_data);
    let block_size = match sampling {
        plan_params::Sampling::MonteCarlo { block_size, .. } => Some(*block_size as usize),
        plan_params::Sampling::Historical => None,
    };

    let result = |stocks: f64, bonds: f64| StocksAndBonds {
        stocks: {
            let empirical_annual_non_log_expected_return_info =
                EmpiricalAnnualNonLogExpectedReturnInfo {
                    value: stocks,
                    block_size,
                    log_volatility_scale: returns_stats_for_planning
                        .standard_deviation
                        .stocks
                        .scale
                        .log,
                };
            let empirical_annual_log_variance = historical_returns_info
                .returns
                .stocks
                .get_target_empirical_annual_log_variance(
                    &empirical_annual_non_log_expected_return_info,
                );
            ReturnsStatsForPlanningProcessedPart {
                empirical_annual_non_log_expected_return_info,
                empirical_annual_log_variance,
                empirical_monthly_non_log_expected_return:
                    annual_non_log_to_monthly_non_log_return_rate(stocks),
            }
        },
        bonds: {
            let empirical_annual_non_log_expected_return_info =
                EmpiricalAnnualNonLogExpectedReturnInfo {
                    value: bonds,
                    block_size,
                    log_volatility_scale: 0.0,
                };
            let empirical_annual_log_variance = historical_returns_info
                .returns
                .bonds
                .get_target_empirical_annual_log_variance(
                    &empirical_annual_non_log_expected_return_info,
                );
            ReturnsStatsForPlanningProcessedPart {
                empirical_annual_non_log_expected_return_info,
                empirical_annual_log_variance,
                empirical_monthly_non_log_expected_return:
                    annual_non_log_to_monthly_non_log_return_rate(bonds),
            }
        },
    };

    match &&returns_stats_for_planning
        .expected_value
        .empirical_annual_non_log
    {
        &plan_params::ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::RegressionPrediction_20YearTIPSYield => {
            result(
                market_data.expected_returns.stocks.regression_prediction,
                market_data.expected_returns.bonds.tips_yield_20_year,
            )
        }
        &plan_params::ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::ConservativeEstimate_20YearTIPSYield => {
            result(
                market_data.expected_returns.stocks.conservative_estimate,
                market_data.expected_returns.bonds.tips_yield_20_year,
            )
        }
        &plan_params::ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::OneOverCAPE_20YearTIPSYield => result(
            market_data.expected_returns.stocks.one_over_cape_rounded,
            market_data.expected_returns.bonds.tips_yield_20_year,
        ),
        &plan_params::ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::Historical => {
            result(market_data.expected_returns.stocks.historical, market_data.expected_returns.bonds.historical)
        }
        &plan_params::ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::FixedEquityPremium { equity_premium } => {
            let bonds = market_data.expected_returns.bonds.tips_yield_20_year;
            result(equity_premium + bonds, bonds)
        }
        &plan_params::ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::Custom { stocks, bonds } => {
            let stocks_base = match stocks.base {
                ExpectedReturnsForPlanningCustomStocksBase::RegressionPrediction => {
                    market_data.expected_returns.stocks.regression_prediction
                }
                ExpectedReturnsForPlanningCustomStocksBase::ConservativeEstimate => {
                    market_data.expected_returns.stocks.conservative_estimate
                }
                ExpectedReturnsForPlanningCustomStocksBase::OneOverCAPE => {
                    market_data.expected_returns.stocks.one_over_cape_rounded
                }
                ExpectedReturnsForPlanningCustomStocksBase::Historical => {
                    market_data.expected_returns.stocks.historical
                }
            };
            let bonds_base = match bonds.base {
                plan_params::ExpectedReturnsForPlanningCustomBondsBase::TwentyYearTIPSYield => {
                    market_data.expected_returns.bonds.tips_yield_20_year
                }
                plan_params::ExpectedReturnsForPlanningCustomBondsBase::Historical => {
                    market_data.expected_returns.bonds.historical
                }
            };
            result(stocks_base + stocks.delta, bonds_base + bonds.delta)
        }
        &plan_params::ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::Fixed { stocks, bonds } => {
            result(*stocks, *bonds)
        }
    }
}
