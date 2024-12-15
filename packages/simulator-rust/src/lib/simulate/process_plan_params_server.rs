pub mod process_adjustments_to_spending;
pub mod process_amount_timed;
pub mod process_annual_inflation;
pub mod process_historical_returns;
pub mod process_market_data_for_presets;
pub mod process_returns_stats_for_planning;
pub mod process_risk;

use super::plan_params_server::{
    PlanParamsServer, PlanParamsServer_Sampling, PlanParamsServer_Strategy,
};
use crate::cuda_bridge::PlanParamsCuda_C_Arrays;
use crate::cuda_bridge_utils::f_cuda;
use crate::wire::WirePlanParamsProcessed;
use crate::{
    cuda_bridge::{
        PlanParamsCuda, PlanParamsCuda_Advanced, PlanParamsCuda_Advanced_ReturnStatsForPlanning,
        PlanParamsCuda_Advanced_Sampling, PlanParamsCuda_Advanced_Sampling_Historical,
        PlanParamsCuda_Advanced_Sampling_MonteCarlo,
        PlanParamsCuda_Advanced_Sampling_MonteCarloOrHistorical,
        PlanParamsCuda_Advanced_Sampling_Type_HistoricalSampling,
        PlanParamsCuda_Advanced_Sampling_Type_MonteCarloSampling,
        PlanParamsCuda_Advanced_Strategy_Strategy_SPAW,
        PlanParamsCuda_Advanced_Strategy_Strategy_SWR,
        PlanParamsCuda_Advanced_Strategy_Strategy_TPAW, PlanParamsCuda_Ages,
        PlanParamsCuda_Ages_SimulationMonths, StocksAndBondsFLOAT,
    },
    market_data::market_data_defs::MarketDataAtTimestampForSimulation,
};
use process_adjustments_to_spending::{
    process_adjustments_to_spending, AdjustmentsToSpendingProcessed,
};
use process_amount_timed::{process_amount_timed, AmountTimedProcessed};
use process_annual_inflation::{process_annual_inflation, InflationProcessed};
use process_historical_returns::{process_historical_returns, HistoricalReturnsProcessed};
use process_market_data_for_presets::{
    process_market_data_for_presets, MarketDataForPresetsProcessed,
};
use process_returns_stats_for_planning::{
    process_returns_stats_for_planning, ReturnsStatsForPlanningProcessed,
};
use process_risk::{process_risk, RiskProcessed};

#[derive(Clone)]
pub struct PlanParamsProcessed {
    pub percentiles: Vec<u32>,
    pub amount_timed: AmountTimedProcessed,
    pub market_data_for_presets: MarketDataForPresetsProcessed,
    pub returns_stats_for_planning: ReturnsStatsForPlanningProcessed,
    pub historical_returns: HistoricalReturnsProcessed,
    pub adjustments_to_spending: AdjustmentsToSpendingProcessed,
    pub inflation: InflationProcessed,
    pub risk: RiskProcessed,
}

impl PlanParamsProcessed {
    pub fn into_cuda(&self, plan_params_server: &PlanParamsServer) -> PlanParamsCuda {
        PlanParamsCuda {
            ages: PlanParamsCuda_Ages {
                simulation_months: {
                    let src = &plan_params_server.ages.simulation_months;
                    PlanParamsCuda_Ages_SimulationMonths {
                        num_months: src.num_months,
                        withdrawal_start_month: src.withdrawal_start_month,
                    }
                },
            },
            adjustments_to_spending: (&self.adjustments_to_spending).into(),
            risk: self.risk.into_cuda(&plan_params_server.risk),
            advanced: PlanParamsCuda_Advanced {
                return_stats_for_planning: PlanParamsCuda_Advanced_ReturnStatsForPlanning {
                    expected_returns_at_month_0: StocksAndBondsFLOAT {
                        stocks: self
                            .returns_stats_for_planning
                            .stocks
                            .empirical_monthly_non_log_expected_return
                            as f_cuda,
                        bonds: self
                            .returns_stats_for_planning
                            .bonds
                            .empirical_monthly_non_log_expected_return
                            as f_cuda,
                    },
                    annual_empirical_log_variance_stocks: self
                        .returns_stats_for_planning
                        .stocks
                        .empirical_annual_log_variance
                        as f_cuda,
                },
                sampling: match plan_params_server.advanced.sampling {
                    PlanParamsServer_Sampling::MonteCarlo(s) => PlanParamsCuda_Advanced_Sampling {
                        type_: PlanParamsCuda_Advanced_Sampling_Type_MonteCarloSampling,
                        monte_carlo_or_historical:
                            PlanParamsCuda_Advanced_Sampling_MonteCarloOrHistorical {
                                monte_carlo: PlanParamsCuda_Advanced_Sampling_MonteCarlo {
                                    seed: s.seed,
                                    num_runs: s.num_runs,
                                    block_size: s.block_size,
                                    stagger_run_starts: if s.stagger_run_starts { 1 } else { 0 },
                                },
                            },
                    },
                    PlanParamsServer_Sampling::Historical(_) => PlanParamsCuda_Advanced_Sampling {
                        type_: PlanParamsCuda_Advanced_Sampling_Type_HistoricalSampling,
                        monte_carlo_or_historical:
                            PlanParamsCuda_Advanced_Sampling_MonteCarloOrHistorical {
                                historical: PlanParamsCuda_Advanced_Sampling_Historical {
                                    ignore: 0,
                                },
                            },
                    },
                },
                strategy: match plan_params_server.advanced.strategy {
                    PlanParamsServer_Strategy::TPAW => {
                        PlanParamsCuda_Advanced_Strategy_Strategy_TPAW
                    }
                    PlanParamsServer_Strategy::SPAW => {
                        PlanParamsCuda_Advanced_Strategy_Strategy_SPAW
                    }
                    PlanParamsServer_Strategy::SWR => PlanParamsCuda_Advanced_Strategy_Strategy_SWR,
                },
            },
        }
    }
}

impl From<PlanParamsProcessed> for WirePlanParamsProcessed {
    fn from(other: PlanParamsProcessed) -> Self {
        Self {
            amount_timed: other.amount_timed.into(),
            market_data_for_presets: other.market_data_for_presets.into(),
            returns_stats_for_planning: other.returns_stats_for_planning.into(),
            historical_returns: other.historical_returns.into(),
            adjustments_to_spending: other.adjustments_to_spending.into(),
            annual_inflation: other.inflation.annual,
            risk: other.risk.into(),
        }
    }
}

impl From<&mut PlanParamsProcessed> for PlanParamsCuda_C_Arrays {
    fn from(other: &mut PlanParamsProcessed) -> Self {
        Self {
            future_savings_by_mfn: other.amount_timed.future_savings.total_by_mfn.as_mut_ptr(),
            income_during_retirement_by_mfn: other
                .amount_timed
                .income_during_retirement
                .total_by_mfn
                .as_mut_ptr(),
            essential_expenses_by_mfn: other
                .amount_timed
                .essential_expenses
                .total_by_mfn
                .as_mut_ptr(),
            discretionary_expenses_by_mfn: other
                .amount_timed
                .discretionary_expenses
                .total_by_mfn
                .as_mut_ptr(),
            tpaw_rra_including_pos_infinity_by_mfn: other
                .risk
                .tpaw
                .rra_unclamped_including_pos_infinity_by_mfn_fcuda
                .as_mut_ptr(),
            spaw_spending_tilt_by_mfn: other
                .risk
                .spaw
                .monthly_spending_tilt_by_mfn_fcuda
                .as_mut_ptr(),
            spaw_and_swr_stock_allocation_savings_portfolio_by_mfn: other
                .risk
                .spaw_and_swr
                .stock_allocation_savings_portfolio_by_mfn_fcuda
                .as_mut_ptr(),
            num_percentiles: other.percentiles.len() as u32,
            percentiles: other.percentiles.as_mut_ptr(),
            historical_returns_series: other.historical_returns.cuda.as_mut_ptr(),
            historical_returns_series_len: other.historical_returns.cuda.len() as u32,
        }
    }
}

pub fn process_plan_params_server(
    percentiles: &[u32],
    plan_params_server: &PlanParamsServer,
    market_data_at_timestamp_for_simulation: &MarketDataAtTimestampForSimulation,
) -> PlanParamsProcessed {
    let market_data_for_presets =
        process_market_data_for_presets(market_data_at_timestamp_for_simulation);

    let returns_stats_for_planning = process_returns_stats_for_planning(
        &plan_params_server.advanced.return_stats_for_planning,
        &plan_params_server.advanced.sampling,
        &market_data_for_presets,
        &market_data_at_timestamp_for_simulation.historical_monthly_returns_info,
    );

    let inflation = process_annual_inflation(
        &plan_params_server.advanced.annual_inflation,
        &market_data_for_presets,
    );

    PlanParamsProcessed {
        percentiles: percentiles.to_vec(),
        market_data_for_presets,
        amount_timed: process_amount_timed(
            plan_params_server.ages.simulation_months.num_months,
            &plan_params_server.wealth.future_savings,
            &plan_params_server.wealth.income_during_retirement,
            &plan_params_server
                .adjustments_to_spending
                .extra_spending
                .essential,
            &plan_params_server
                .adjustments_to_spending
                .extra_spending
                .discretionary,
            inflation.monthly,
        ),
        historical_returns: process_historical_returns(
            &returns_stats_for_planning,
            &plan_params_server.advanced.historical_returns_adjustment,
            &market_data_at_timestamp_for_simulation.historical_monthly_returns_info,
        ),
        adjustments_to_spending: process_adjustments_to_spending(plan_params_server, &inflation),
        risk: process_risk(
            &plan_params_server.constants,
            &plan_params_server.ages,
            &plan_params_server.risk,
        ),
        returns_stats_for_planning,
        inflation,
    }
}
