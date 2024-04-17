use crate::plan_params::PlanParams;
use crate::utils::*;
use crate::{
    params::*, plan_params::process_plan_params::plan_params_processed::PlanParamsProcessed,
};

use self::get_net_present_value_by_mfn::{
    cumulative_1_plus_g_over_1_plus_r, get_net_present_value_by_mfn, NetPresentValueByMFN,
};
use self::shared_types::StocksAndBonds;

pub struct NetPresentValueForWithdrawals {
    pub lmp: NetPresentValueByMFN,
    pub essential: NetPresentValueByMFN,
    pub discretionary: NetPresentValueByMFN,
}

pub struct SPAWNetPresentValue {
    pub savings: NetPresentValueByMFN,
    pub withdrawals: NetPresentValueForWithdrawals,
    pub legacy: NetPresentValueByMFN,
}

pub struct PreCalculationsForSPAW {
    pub net_present_value: SPAWNetPresentValue,
    pub cumulative_1_plus_g_over_1_plus_r: Vec<f64>,
}

pub struct TPAWNetPresentValue {
    pub savings: NetPresentValueByMFN,
    pub withdrawals: NetPresentValueForWithdrawals,
}

pub struct PreCalculationsForTPAW {
    pub net_present_value: TPAWNetPresentValue,
    pub cumulative_1_plus_g_over_1_plus_r: Vec<f64>,
}

pub struct PreCalculations {
    pub spaw: PreCalculationsForSPAW,
    pub tpaw: PreCalculationsForTPAW,
}

pub fn do_pre_calculations(
    plan_params: &PlanParams,
    plan_params_processed: &PlanParamsProcessed,
    params: &Params,
) -> PreCalculations {
    PreCalculations {
        spaw: pre_calculations_for_spaw(plan_params, plan_params_processed, params),
        tpaw: pre_calculations_for_tpaw(plan_params, plan_params_processed, params),
    }
}
fn pre_calculations_for_tpaw(
    plan_params: &PlanParams,
    plan_params_processed: &PlanParamsProcessed,
    params: &Params,
) -> PreCalculationsForTPAW {
    let num_months = plan_params.ages.simulation_months.num_months;
    let expected_returns = blend_returns(&StocksAndBonds {
        stocks: plan_params_processed
            .returns_stats_for_planning
            .stocks
            .empirical_monthly_non_log_expected_return,
        bonds: plan_params_processed
            .returns_stats_for_planning
            .bonds
            .empirical_monthly_non_log_expected_return,
    });

    let bonds_rate = vec![expected_returns(0.0); num_months as usize];

    let regular_rate: Vec<f64> = params
        .target_allocation
        .regular_portfolio
        .tpaw
        .iter()
        .map(|x| expected_returns(*x))
        .collect();

    let savings =
        get_net_present_value_by_mfn(&bonds_rate, &plan_params_processed.by_month.wealth.total);
    let lmp = get_net_present_value_by_mfn(
        &bonds_rate,
        &plan_params_processed.by_month.risk.tpaw_and_spaw.lmp,
    );
    let essential = get_net_present_value_by_mfn(
        &bonds_rate,
        &plan_params_processed
            .by_month
            .adjustments_to_spending
            .extra_spending
            .essential
            .total,
    );
    let discretionary = get_net_present_value_by_mfn(
        &regular_rate,
        &plan_params_processed
            .by_month
            .adjustments_to_spending
            .extra_spending
            .discretionary
            .total,
    );

    let result = PreCalculationsForTPAW {
        net_present_value: TPAWNetPresentValue {
            savings,
            withdrawals: NetPresentValueForWithdrawals {
                lmp,
                essential,
                discretionary,
            },
        },
        cumulative_1_plus_g_over_1_plus_r: cumulative_1_plus_g_over_1_plus_r(
            &regular_rate,
            &params.spending_tilt,
        ),
    };
    return result;
}

fn pre_calculations_for_spaw(
    plan_params: &PlanParams,
    plan_params_processed: &PlanParamsProcessed,
    params: &Params,
) -> PreCalculationsForSPAW {
    let expected_returns = blend_returns(&StocksAndBonds {
        stocks: plan_params_processed
            .returns_stats_for_planning
            .stocks
            .empirical_monthly_non_log_expected_return,
        bonds: plan_params_processed
            .returns_stats_for_planning
            .bonds
            .empirical_monthly_non_log_expected_return,
    });

    let rate: Vec<f64> = params
        .target_allocation
        .regular_portfolio
        .spaw_and_swr
        .iter()
        .map(|x| expected_returns(*x))
        .collect();
    let n = rate.len();
    let savings = get_net_present_value_by_mfn(&rate, &plan_params_processed.by_month.wealth.total);
    let lmp = get_net_present_value_by_mfn(
        &rate,
        &plan_params_processed.by_month.risk.tpaw_and_spaw.lmp,
    );
    let essential = get_net_present_value_by_mfn(
        &rate,
        &plan_params_processed
            .by_month
            .adjustments_to_spending
            .extra_spending
            .essential
            .total,
    );
    let discretionary = get_net_present_value_by_mfn(
        &rate,
        &plan_params_processed
            .by_month
            .adjustments_to_spending
            .extra_spending
            .discretionary
            .total,
    );
    let mut legacy_amount_by_month = vec![0.0; n];
    legacy_amount_by_month[n - 1] = params.legacy_target / (1.0 + rate[n - 1]);
    let legacy = get_net_present_value_by_mfn(&rate, &legacy_amount_by_month);

    PreCalculationsForSPAW {
        net_present_value: SPAWNetPresentValue {
            savings,
            withdrawals: NetPresentValueForWithdrawals {
                lmp,
                essential,
                discretionary,
            },
            legacy,
        },
        cumulative_1_plus_g_over_1_plus_r: cumulative_1_plus_g_over_1_plus_r(
            &rate,
            &params.spending_tilt,
        ),
    }
}
