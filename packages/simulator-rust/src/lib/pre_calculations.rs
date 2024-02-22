use crate::params::*;
use crate::utils::*;

use self::get_net_present_value_by_mfn::{
    cumulative_1_plus_g_over_1_plus_r, get_net_present_value_by_mfn, NetPresentValueByMFN,
};

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

pub fn do_pre_calculations(params: &Params) -> PreCalculations {
    PreCalculations {
        spaw: pre_calculations_for_spaw(params),
        tpaw: pre_calculations_for_tpaw(params),
    }
}
fn pre_calculations_for_tpaw(params: &Params) -> PreCalculationsForTPAW {
    let num_months = params.num_months;
    let expected_returns = blend_returns(&params.expected_monthly_returns);

    let bonds_rate = vec![expected_returns(0.0); num_months];

    let regular_rate: Vec<f64> = params
        .target_allocation
        .regular_portfolio
        .tpaw
        .iter()
        .map(|x| expected_returns(*x))
        .collect();

    let savings = get_net_present_value_by_mfn(&bonds_rate, &params.by_month.savings);
    let lmp = get_net_present_value_by_mfn(&bonds_rate, &params.lmp);
    let essential =
        get_net_present_value_by_mfn(&bonds_rate, &params.by_month.withdrawals_essential);
    let discretionary = get_net_present_value_by_mfn(
        &regular_rate,
        &params.by_month.withdrawals_discretionary,
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

fn pre_calculations_for_spaw(params: &Params) -> PreCalculationsForSPAW {
    let expected_returns = blend_returns(&params.expected_monthly_returns);
    let rate: Vec<f64> = params
        .target_allocation
        .regular_portfolio
        .spaw_and_swr
        .iter()
        .map(|x| expected_returns(*x))
        .collect();
    let n = rate.len();
    let savings = get_net_present_value_by_mfn(&rate, &params.by_month.savings);
    let lmp = get_net_present_value_by_mfn(&rate, &params.lmp);
    let essential =
        get_net_present_value_by_mfn(&rate, &params.by_month.withdrawals_essential);
    let discretionary =
        get_net_present_value_by_mfn(&rate, &params.by_month.withdrawals_discretionary);
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
