use super::process_expected_returns_for_planning::ExpectedReturnsForPlanningProcessed;
use crate::{
    data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues,
    historical_monthly_returns::{
        get_historical_monthly_returns, HistoricalMonthlyLogReturnsAdjustedInfo,
    },
    shared_types::StocksAndBonds,
};

pub fn process_historical_monthly_log_returns_adjustment(
    expected_returns_for_planning: &ExpectedReturnsForPlanningProcessed,
    market_data: &DataForMarketBasedPlanParamValues,
    override_to_fixed_for_testing: bool,
) -> StocksAndBonds<HistoricalMonthlyLogReturnsAdjustedInfo> {
    let historical_monthly_returns =
        get_historical_monthly_returns(market_data.timestamp_ms_for_historical_returns);

    let mut result = StocksAndBonds {
        stocks: historical_monthly_returns
            .stocks
            .adjust_log_returns_detailed(
                &expected_returns_for_planning
                    .empirical_annual_non_log_return_info
                    .stocks,
            ),
        bonds: historical_monthly_returns
            .bonds
            .adjust_log_returns_detailed(
                &expected_returns_for_planning
                    .empirical_annual_non_log_return_info
                    .bonds,
            ),
    };
    if override_to_fixed_for_testing {
        result.stocks.log_series = vec![
            expected_returns_for_planning
                .empirical_annual_non_log_return_info
                .stocks
                .value
                .ln_1p()
                / 12.0;
            result.stocks.log_series.len()
        ];
        result.bonds.log_series = vec![
            expected_returns_for_planning
                .empirical_annual_non_log_return_info
                .bonds
                .value
                .ln_1p()
                / 12.0;
            result.bonds.log_series.len()
        ];
    }
    result
}
