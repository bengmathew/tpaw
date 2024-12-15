use crate::utils::expected_value_of_returns::annual_non_log_to_monthly_non_log_return_rate;

use crate::simulate::plan_params_server::PlanParamsServer_AnnualInflation;

use super::process_market_data_for_presets::MarketDataForPresetsProcessed;

#[derive(Debug, Clone)]
pub struct InflationProcessed {
    pub annual: f64,
    pub monthly: f64,
}

impl InflationProcessed {
    pub fn print(&self, indent: usize) {
        println!("{:indent$}{:.65}", "", self.annual, indent = indent);
        println!("{:indent$}{:.65}", "", self.monthly, indent = indent);
    }
}

pub fn process_annual_inflation(
    inflation: &PlanParamsServer_AnnualInflation,
    market_data_for_presets_processed: &MarketDataForPresetsProcessed,
) -> InflationProcessed {
    let annual = match inflation {
        PlanParamsServer_AnnualInflation::Suggested(_) => {
            market_data_for_presets_processed
                .inflation
                .suggested_annual
        }
        PlanParamsServer_AnnualInflation::Manual(value) => *value,
    };
    let monthly = annual_non_log_to_monthly_non_log_return_rate(annual);
    return InflationProcessed { annual, monthly };
}
