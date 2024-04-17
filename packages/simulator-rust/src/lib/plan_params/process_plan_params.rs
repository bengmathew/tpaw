pub mod plan_params_processed;
pub mod process_annual_inflation;
pub mod process_by_month_params;
pub mod process_historical_returns_adjustment;
pub mod process_market_data;
pub mod process_returns_stats_for_planning;
pub mod process_risk;

use self::{
    plan_params_processed::PlanParamsProcessed, process_annual_inflation::process_annual_inflation,
    process_by_month_params::process_by_month_params,
    process_historical_returns_adjustment::process_historical_returns_adjustment,
    process_market_data::process_market_data,
    process_returns_stats_for_planning::process_returns_stats_for_planning,
    process_risk::process_risk,
};
use crate::data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues;

use super::PlanParams;

pub fn process_plan_params(
    plan_params: &PlanParams,
    market_data: &DataForMarketBasedPlanParamValues,
) -> PlanParamsProcessed {
    let market_data_processed = process_market_data(market_data);

    let returns_stats_for_planning = process_returns_stats_for_planning(
        &plan_params.advanced.sampling,
        &plan_params.advanced.returns_stats_for_planning,
        &market_data_processed,
    );

    let historical_monthly_returns_adjusted = process_historical_returns_adjustment(
        &returns_stats_for_planning,
        &plan_params.advanced.historical_returns_adjustment,
        market_data.timestamp_ms_for_market_data,
    );

    let inflation = process_annual_inflation(
        &plan_params.advanced.annual_inflation,
        &market_data_processed,
    );

    let by_month = process_by_month_params(&plan_params, inflation.monthly);

    let risk = process_risk(&plan_params, &returns_stats_for_planning);

    PlanParamsProcessed {
        market_data: market_data_processed,
        returns_stats_for_planning,
        historical_returns_adjusted: historical_monthly_returns_adjusted,
        risk,
        inflation,
        by_month,
    }
}
