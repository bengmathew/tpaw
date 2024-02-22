use crate::{
    data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues, round::RoundP,
};

pub fn get_suggested_annual_inflation(market_data: &DataForMarketBasedPlanParamValues) -> f64 {
    market_data.inflation.value.round_p(3)
}
