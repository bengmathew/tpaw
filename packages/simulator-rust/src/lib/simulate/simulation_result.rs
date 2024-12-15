use crate::cuda_bridge_utils::{f_cuda, f_cuda_currency};
use crate::to_float_wire::ToFloatWireVec;
use crate::wire::WirePortfolioBalanceEstimationResult;
use crate::wire::*;

use super::get_approx_net_present_value_for_tpaw_balance_sheet::TPAW_ApproxNetPresentValueForBalanceSheet;
use super::process_plan_params_server::PlanParamsProcessed;

#[derive(Clone)]
pub struct SimulationResultArrays {
    pub by_percentile_by_mfn_simulated_percentile_major_balance_start: Vec<f_cuda_currency>,
    pub by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential: Vec<f_cuda_currency>,
    pub by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary:
        Vec<f_cuda_currency>,
    pub by_percentile_by_mfn_simulated_percentile_major_withdrawals_general: Vec<f_cuda_currency>,
    pub by_percentile_by_mfn_simulated_percentile_major_withdrawals_total: Vec<f_cuda_currency>,
    pub by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate:
        Vec<f_cuda>,
    pub by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio:
        Vec<f_cuda>,
    pub by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth:
        Vec<f_cuda>,
    pub tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt: Vec<f_cuda>,
    pub by_percentile_ending_balance: Vec<f_cuda_currency>,
    pub tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn: Vec<f_cuda>,
}

impl From<SimulationResultArrays> for WireSimulationResultArrays {
    fn from(other: SimulationResultArrays) -> Self {
        Self {
            by_percentile_by_mfn_simulated_percentile_major_balance_start_x100: other.by_percentile_by_mfn_simulated_percentile_major_balance_start.to_float_wire(100),
            by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential_x100: other.by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential.to_float_wire(100)       ,
            by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary_x100: other.by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary.to_float_wire(100),
            by_percentile_by_mfn_simulated_percentile_major_withdrawals_general_x100: other.by_percentile_by_mfn_simulated_percentile_major_withdrawals_general.to_float_wire(100),
            by_percentile_by_mfn_simulated_percentile_major_withdrawals_total_x100: other.by_percentile_by_mfn_simulated_percentile_major_withdrawals_total.to_float_wire(100),
            by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate_x10000: other.by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate.to_float_wire(10000),
            by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio_x100: other.by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio.to_float_wire(100),
            by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth_x100: other.by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth.to_float_wire(100),
            tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt_x10000: other.tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt.to_float_wire(10000),
            by_percentile_ending_balance: other.by_percentile_ending_balance,
        }
    }
}

#[derive(Clone)]
pub struct SimulationResult {
    pub arrays: SimulationResultArrays,
    pub num_runs: u32,
    pub num_runs_with_insufficient_funds: u32,
    pub tpaw_approx_net_present_value_for_balance_sheet:
        Option<TPAW_ApproxNetPresentValueForBalanceSheet>,
}

impl SimulationResult {
    pub fn into_wire(
        self,
        portfolio_balance_estimation_result: Option<WirePortfolioBalanceEstimationResult>,
        plan_params_processed: PlanParamsProcessed,
        portfolio_balance_estimation_in_ms: i64,
        simulation_in_ms: i64,
        total_in_ms: i64,
    ) -> WireSimulationResult {
        WireSimulationResult {
            portfolio_balance_estimation_result_opt: portfolio_balance_estimation_result,
            plan_params_processed: plan_params_processed.into(),
            arrays: self.arrays.into(),
            num_runs: self.num_runs,
            num_runs_with_insufficient_funds: self.num_runs_with_insufficient_funds,
            tpaw_net_present_value_approx_for_balance_sheet_opt: self
                .tpaw_approx_net_present_value_for_balance_sheet
                .map(|v| v.into()),
            performance: WireSimulationPerformance {
                portfolio_balance_estimation_in_ms,
                simulation_in_ms,
                total_in_ms,
            },
        }
    }
}
