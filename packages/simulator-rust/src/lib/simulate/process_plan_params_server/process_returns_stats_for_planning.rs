use crate::{
    historical_monthly_returns::HistoricalReturnsInfo,
    utils::{
        expected_value_of_returns::{
            annual_non_log_to_monthly_non_log_return_rate, EmpiricalAnnualNonLogExpectedReturnInfo,
        },
        shared_types::StocksAndBonds,
    }, wire::{WireReturnsStatsForPlanningProcessed, WireReturnsStatsForPlanningProcessedPart},
};

use crate::simulate::plan_params_server::{
    PlanParamsServer_ExpectedReturnsForPlanning_Custom_Bonds_Base,
    PlanParamsServer_ExpectedReturnsForPlanning_Custom_Stocks_Base,
    PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog,
    PlanParamsServer_ReturnStatsForPlanning,
    PlanParamsServer_Sampling,
};

use super::process_market_data_for_presets::MarketDataForPresetsProcessed;

#[derive(Clone, Debug)]
pub struct ReturnsStatsForPlanningProcessedPart {
    pub empirical_annual_non_log_expected_return_info: EmpiricalAnnualNonLogExpectedReturnInfo,
    pub empirical_annual_log_variance: f64,
    // More accurately this should be called empirical_annual_non_log_expected_return_expressed_as_monthly.
    pub empirical_monthly_non_log_expected_return: f64,
}

impl From<ReturnsStatsForPlanningProcessedPart> for WireReturnsStatsForPlanningProcessedPart {
    fn from(other: ReturnsStatsForPlanningProcessedPart) -> Self {
        Self {
            empirical_annual_non_log_expected_return: other.empirical_annual_non_log_expected_return_info.value,
            empirical_annual_log_variance: other.empirical_annual_log_variance,
        }
    }
}

pub type ReturnsStatsForPlanningProcessed = StocksAndBonds<ReturnsStatsForPlanningProcessedPart>;

impl From<ReturnsStatsForPlanningProcessed> for WireReturnsStatsForPlanningProcessed {
    fn from(other: ReturnsStatsForPlanningProcessed) -> Self {
        Self {
            stocks: other.stocks.into(),
            bonds: other.bonds.into(),
        }
    }
}

pub fn process_returns_stats_for_planning(
    returns_stats_for_planning: &PlanParamsServer_ReturnStatsForPlanning,
    sampling: &PlanParamsServer_Sampling,
    market_data_for_presets_processed: &MarketDataForPresetsProcessed,
    historical_monthly_returns_info: &HistoricalReturnsInfo,
) -> ReturnsStatsForPlanningProcessed {
    let historical_monthly_returns = &historical_monthly_returns_info.returns;

    let block_size = match sampling {
        PlanParamsServer_Sampling::MonteCarlo(sampling) => Some(sampling.block_size as usize),
        PlanParamsServer_Sampling::Historical(_) => None,
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
            let empirical_annual_log_variance = historical_monthly_returns
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
            let empirical_annual_log_variance = historical_monthly_returns
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

    let expected_returns_preset_values = &market_data_for_presets_processed.expected_returns;
    match &&returns_stats_for_planning
        .expected_value
        .empirical_annual_non_log
    {
        &PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::RegressionPrediction_20YearTIPSYield => {
            
            result(
                expected_returns_preset_values.stocks.regression_prediction,
                expected_returns_preset_values.bonds.tips_yield_20_year,
            )
        }
        &PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::ConservativeEstimate_20YearTIPSYield => {
            result(
                expected_returns_preset_values.stocks.conservative_estimate,
                expected_returns_preset_values.bonds.tips_yield_20_year,
            )
        }
        &PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::OneOverCAPE_20YearTIPSYield => result(
            expected_returns_preset_values.stocks.one_over_cape_rounded,
            expected_returns_preset_values.bonds.tips_yield_20_year,
        ),
        &PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::Historical => {
            result(
                expected_returns_preset_values.stocks.historical,
                expected_returns_preset_values.bonds.historical,
            )
        }
        &PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::FixedEquityPremium { equity_premium } => {
            let bonds = expected_returns_preset_values.bonds.tips_yield_20_year;
            result(equity_premium + bonds, bonds)
        }
        &PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::Custom { stocks, bonds } => {
            let stocks_base = match stocks.base {
                PlanParamsServer_ExpectedReturnsForPlanning_Custom_Stocks_Base::RegressionPrediction => {
                    expected_returns_preset_values.stocks.regression_prediction
                }
                PlanParamsServer_ExpectedReturnsForPlanning_Custom_Stocks_Base::ConservativeEstimate => {
                    expected_returns_preset_values.stocks.conservative_estimate
                }
                PlanParamsServer_ExpectedReturnsForPlanning_Custom_Stocks_Base::OneOverCape => {
                    expected_returns_preset_values.stocks.one_over_cape_rounded
                }
                PlanParamsServer_ExpectedReturnsForPlanning_Custom_Stocks_Base::HistoricalStocks => {
                    expected_returns_preset_values.stocks.historical
                }
            };
            let bonds_base = match bonds.base {
                PlanParamsServer_ExpectedReturnsForPlanning_Custom_Bonds_Base::TwentyYearTipsYield => {
                    expected_returns_preset_values.bonds.tips_yield_20_year
                }
                PlanParamsServer_ExpectedReturnsForPlanning_Custom_Bonds_Base::HistoricalBonds => {
                    expected_returns_preset_values.bonds.historical
                }
            };
            result(stocks_base + stocks.delta, bonds_base + bonds.delta)
        }
        &PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::Fixed { stocks, bonds } => {
            result(*stocks, *bonds)
        }
    }
}
