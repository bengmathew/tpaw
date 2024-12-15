use std::{iter::zip, ops::RangeInclusive};

use crate::{
    cuda_bridge,
    cuda_bridge_utils::f_cuda,
    historical_monthly_returns::{
        HistoricalMonthlyLogReturnsAdjustedInfoArgs, HistoricalMonthlyLogReturnsAdjustedStats,
        HistoricalReturnsInfo,
    },
    return_series::Stats,
    shared_types::{LogAndNonLog, YearAndMonth},
    wire::{
        wire_plan_params_server_historical_returns_adjustment,
        WireHistoricalMonthlyLogReturnsAdjustedInfo, WireHistoricalReturnsProcessed,
    },
};

use crate::simulate::plan_params_server::PlanParamsServer_HistoricalReturnsAdjustment;

use super::process_returns_stats_for_planning::ReturnsStatsForPlanningProcessed;

#[derive(Clone)]
pub struct HistoricalReturnsProcessedPart {
    pub stats: HistoricalMonthlyLogReturnsAdjustedStats,
    pub args: HistoricalMonthlyLogReturnsAdjustedInfoArgs,
    pub src_annualized_stats: LogAndNonLog<Stats>,
}

impl From<HistoricalReturnsProcessedPart> for WireHistoricalMonthlyLogReturnsAdjustedInfo {
    fn from(value: HistoricalReturnsProcessedPart) -> Self {
        Self {
            stats: value.stats.into(),
            args: value.args.into(),
            src_annualized_stats: value.src_annualized_stats.into(),
        }
    }
}

#[derive(Clone)]
pub struct HistoricalReturnsProcessed {
    pub month_range: RangeInclusive<YearAndMonth>,
    pub stocks: HistoricalReturnsProcessedPart,
    pub bonds: HistoricalReturnsProcessedPart,
    pub cuda: Vec<cuda_bridge::HistoricalReturnsCuda>,
}

impl From<HistoricalReturnsProcessed> for WireHistoricalReturnsProcessed {
    fn from(value: HistoricalReturnsProcessed) -> Self {
        WireHistoricalReturnsProcessed {
            month_range: value.month_range.into(),
            stocks: value.stocks.into(),
            bonds: value.bonds.into(),
        }
    }
}
pub fn process_historical_returns(
    returns_stats_for_planning: &ReturnsStatsForPlanningProcessed,
    historical_returns_adjustment: &PlanParamsServer_HistoricalReturnsAdjustment,
    historical_monthly_returns_info: &HistoricalReturnsInfo,
) -> HistoricalReturnsProcessed {
    let historical_monthly_returns = &historical_monthly_returns_info.returns;

    let stocks_adjusted = {
        let mut empirical_annual_non_log_expected_return_info = returns_stats_for_planning
            .stocks
            .empirical_annual_non_log_expected_return_info
            .clone();

        match historical_returns_adjustment
            .override_to_fixed_for_testing
        {
            wire_plan_params_server_historical_returns_adjustment::OverrideToFixedForTesting::None(_) => {},
            wire_plan_params_server_historical_returns_adjustment::OverrideToFixedForTesting::ToExpectedReturnsForPlanning(_) => {
                empirical_annual_non_log_expected_return_info.log_volatility_scale = 0.0;
            },
            wire_plan_params_server_historical_returns_adjustment::OverrideToFixedForTesting::Manual(manual) => {
                empirical_annual_non_log_expected_return_info.log_volatility_scale = 0.0;
                empirical_annual_non_log_expected_return_info.value = manual.stocks;
            }
        }
        historical_monthly_returns
            .stocks
            .adjust_log_returns_detailed(&empirical_annual_non_log_expected_return_info)
    };

    let bonds_adjusted = {
        let mut empirical_annual_non_log_expected_return_info = returns_stats_for_planning
            .bonds
            .empirical_annual_non_log_expected_return_info
            .clone();
        empirical_annual_non_log_expected_return_info.log_volatility_scale =
            historical_returns_adjustment
                .standard_deviation
                .bonds
                .scale
                .log;
        match historical_returns_adjustment
            .override_to_fixed_for_testing
        {
            wire_plan_params_server_historical_returns_adjustment::OverrideToFixedForTesting::None(_) => {},
            wire_plan_params_server_historical_returns_adjustment::OverrideToFixedForTesting::ToExpectedReturnsForPlanning(_) => {
                empirical_annual_non_log_expected_return_info.log_volatility_scale = 0.0;
            },
            wire_plan_params_server_historical_returns_adjustment::OverrideToFixedForTesting::Manual(manual) => {
                empirical_annual_non_log_expected_return_info.log_volatility_scale = 0.0;
                empirical_annual_non_log_expected_return_info.value = manual.bonds;
            }
        }
        historical_monthly_returns
            .bonds
            .adjust_log_returns_detailed(&empirical_annual_non_log_expected_return_info)
    };

    let cuda: Vec<cuda_bridge::HistoricalReturnsCuda> = {
        let stock_series = stocks_adjusted.non_log_series;
        let bond_series = bonds_adjusted.non_log_series;
        assert_eq!(stock_series.len(), bond_series.len());

        zip(stock_series, bond_series)
            .map(|(s, b)| cuda_bridge::HistoricalReturnsCuda {
                stocks: cuda_bridge::HistoricalReturnsCuda_Part {
                    returns: s as f_cuda,
                    expected_return_change: 0.0 as f_cuda,
                },
                bonds: cuda_bridge::HistoricalReturnsCuda_Part {
                    returns: b as f_cuda,
                    expected_return_change: 0.0 as f_cuda,
                },
            })
            .collect()
    };

    let result = HistoricalReturnsProcessed {
        month_range: historical_monthly_returns_info.month_range.clone(),
        stocks: HistoricalReturnsProcessedPart {
            stats: stocks_adjusted.stats,
            args: stocks_adjusted.args,
            src_annualized_stats: stocks_adjusted.src_annualized_stats,
        },
        bonds: HistoricalReturnsProcessedPart {
            stats: bonds_adjusted.stats,
            args: bonds_adjusted.args,
            src_annualized_stats: bonds_adjusted.src_annualized_stats,
        },
        cuda,
    };
    result
}
