use crate::{
    data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues,
    expected_value_of_returns::EmpiricalAnnualNonLogExpectedReturnInfo,
    historical_monthly_returns::{
        get_historical_monthly_returns_info, HistoricalMonthlyLogReturnsAdjustedInfo,
    },
    plan_params::{
        HistoricalReturnsAdjustment, ReturnsStatsForPlanning, Sampling,
    },
    shared_types::StocksAndBonds,
};

use super::process_returns_stats_for_planning::ReturnsStatsForPlanningProcessed;

pub fn process_historical_returns_adjustment(
    returns_stats_for_planning: &ReturnsStatsForPlanningProcessed,
    historical_returns_adjustment: &HistoricalReturnsAdjustment,
    timestamp_ms_for_historical_returns: i64,
) -> StocksAndBonds<HistoricalMonthlyLogReturnsAdjustedInfo> {
    let historical_monthly_returns =
        &get_historical_monthly_returns_info(timestamp_ms_for_historical_returns).returns;

    let mut result = StocksAndBonds {
        stocks: {
            let mut empirical_annual_non_log_expected_return_info = returns_stats_for_planning
                .stocks
                .empirical_annual_non_log_expected_return_info
                .clone();
            if historical_returns_adjustment
                .standard_deviation
                .override_to_fixed_for_testing
            {
                empirical_annual_non_log_expected_return_info.log_volatility_scale = 0.0;
            }
            historical_monthly_returns
                .stocks
                .adjust_log_returns_detailed(&empirical_annual_non_log_expected_return_info)
        },
        bonds: {
            let mut empirical_annual_non_log_expected_return_info = returns_stats_for_planning
                .bonds
                .empirical_annual_non_log_expected_return_info
                .clone();
            empirical_annual_non_log_expected_return_info.log_volatility_scale =
                if historical_returns_adjustment
                    .standard_deviation
                    .override_to_fixed_for_testing
                {
                    0.0
                } else {
                    historical_returns_adjustment
                        .standard_deviation
                        .bonds
                        .scale
                        .log
                };
            historical_monthly_returns
                .bonds
                .adjust_log_returns_detailed(&empirical_annual_non_log_expected_return_info)
        },
    };
    result
}
