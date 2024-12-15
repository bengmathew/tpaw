pub mod get_approx_net_present_value_for_tpaw_balance_sheet;
pub mod plan_params_server;
pub mod process_plan_params_server;
pub mod simulation_result;

use self::process_plan_params_server::process_plan_params_server;
use crate::cuda_bridge::{cuda_simulate, PlanParamsCuda_C_Arrays, ResultCudaArrays};
use crate::cuda_bridge_utils::f_cuda;
use crate::market_data::market_data_defs::MarketDataSeriesForSimulation;
use crate::shared_types::StocksAndBonds;
use plan_params_server::PlanParamsServer;
use process_plan_params_server::PlanParamsProcessed;
use simulation_result::{SimulationResult, SimulationResultArrays};

pub fn simulate(
    current_portfolio_balance: f64,
    percentiles: &Vec<u32>,
    num_months_to_simulate: u32,
    plan_params_server: &PlanParamsServer,
    timestamp_for_market_data_ms: i64,
    market_data_series_for_simulation: &MarketDataSeriesForSimulation,
) -> (SimulationResult, PlanParamsProcessed) {
    let market_data_at_timestamp_for_simulation =
        market_data_series_for_simulation.for_timestamp(timestamp_for_market_data_ms);


    let mut plan_params_processed = process_plan_params_server(
        percentiles,
        &plan_params_server,
        &market_data_at_timestamp_for_simulation,
    );

    let n1 = percentiles.len() * num_months_to_simulate as usize;
    let mut result_arrays = SimulationResultArrays {
        by_percentile_by_mfn_simulated_percentile_major_balance_start: vec![0.0; n1],
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential: vec![0.0; n1],
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary: vec![0.0; n1],
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_general: vec![0.0; n1],
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_total: vec![0.0; n1],
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate: vec![0.0; n1],
        by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio: vec![0.0; n1],
        by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth: vec![0.0; n1],
        tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt: vec![0.0; n1],
        by_percentile_ending_balance: vec![0.0; percentiles.len()],
        tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn: vec![0.0; plan_params_server.ages.simulation_months.num_months as usize],
    };

    let mut result_arrays_cuda = ResultCudaArrays{
        by_percentile_by_mfn_simulated_percentile_major_balance_start: result_arrays.by_percentile_by_mfn_simulated_percentile_major_balance_start.as_mut_ptr(),
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential: result_arrays.by_percentile_by_mfn_simulated_percentile_major_withdrawals_essential.as_mut_ptr(),
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary: result_arrays.by_percentile_by_mfn_simulated_percentile_major_withdrawals_discretionary.as_mut_ptr(),
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_general: result_arrays.by_percentile_by_mfn_simulated_percentile_major_withdrawals_general.as_mut_ptr(),
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_total: result_arrays.by_percentile_by_mfn_simulated_percentile_major_withdrawals_total.as_mut_ptr(),
        by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate: result_arrays.by_percentile_by_mfn_simulated_percentile_major_withdrawals_from_savings_portfolio_rate.as_mut_ptr(),
        by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio: result_arrays.by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio.as_mut_ptr(),
        by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth: result_arrays.by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_total_portfolio_or_zero_if_no_wealth.as_mut_ptr(),
        tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt: result_arrays.tpaw_by_percentile_by_mfn_simulated_percentile_major_spending_tilt.as_mut_ptr(),
        by_percentile_ending_balance: result_arrays.by_percentile_ending_balance.as_mut_ptr(),
        tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn: result_arrays.tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn.as_mut_ptr(),
    };

    let result_not_arrays = unsafe {
        let plan_params_cuda_c_arrays: PlanParamsCuda_C_Arrays =
            (&mut plan_params_processed).into();
        cuda_simulate(
            num_months_to_simulate as u32,
            current_portfolio_balance,
            &plan_params_processed.into_cuda(&plan_params_server),
            &plan_params_cuda_c_arrays,
            &mut result_arrays_cuda,
        )
    };

    let tpaw_approx_net_present_value_for_balance_sheet =
        get_approx_net_present_value_for_tpaw_balance_sheet::get_approx_net_present_value_for_tpaw_balance_sheet(
            &plan_params_processed.amount_timed,
            {
                let src = &plan_params_processed.returns_stats_for_planning;
                StocksAndBonds::<f_cuda> {
                    stocks: src.stocks.empirical_monthly_non_log_expected_return as f_cuda,
                    bonds: src.bonds.empirical_monthly_non_log_expected_return as f_cuda,
                }
            },
            result_not_arrays.tpaw_net_present_value_exact_month_0_legacy ,
            &result_arrays.tpaw_stock_allocation_total_portfolio_for_month_0_by_mfn,
        );
    let result = SimulationResult {
        arrays: result_arrays,
        num_runs: result_not_arrays.num_runs,
        num_runs_with_insufficient_funds: result_not_arrays.num_runs_with_insufficient_funds,
        tpaw_approx_net_present_value_for_balance_sheet: Some(
            tpaw_approx_net_present_value_for_balance_sheet,
        ),
    };
    (result, plan_params_processed)
}
