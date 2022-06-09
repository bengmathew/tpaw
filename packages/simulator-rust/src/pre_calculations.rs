use crate::params::*;
use crate::utils::*;

pub struct NetPresentValueOfSequence {
    pub with_current_year: Vec<f64>,
    pub without_current_year: Vec<f64>,
}

pub struct NetPresentValueForWithdrawals {
    pub lmp: NetPresentValueOfSequence,
    pub essential: NetPresentValueOfSequence,
    pub discretionary: NetPresentValueOfSequence,
}

pub struct SPAWNetPresentValue {
    pub savings: NetPresentValueOfSequence,
    pub withdrawals: NetPresentValueForWithdrawals,
    pub legacy: NetPresentValueOfSequence,
}

pub struct PreCalculationsForSPAW {
    pub net_present_value: SPAWNetPresentValue,
    pub cumulative_1_plus_g_over_1_plus_r: Vec<f64>,
}

pub struct TPAWNetPresentValue {
    pub savings: NetPresentValueOfSequence,
    pub withdrawals: NetPresentValueForWithdrawals,
}

pub struct PreCalculationsForTPAW {
    pub net_present_value: TPAWNetPresentValue,
}

pub struct PreCalculations {
    pub spaw: PreCalculationsForSPAW,
    pub tpaw: PreCalculationsForTPAW,
}

pub fn pre_calculations(params: &Params) -> PreCalculations {
    PreCalculations {
        spaw: pre_calculations_for_spaw(params),
        tpaw: pre_calculations_for_tpaw(params),
    }
}
fn pre_calculations_for_tpaw(params: &Params) -> PreCalculationsForTPAW {
    let num_years = params.num_years as usize;
    let expected_returns = blend_returns(&params.expected_returns);

    let bonds_rate = vec![expected_returns(0.0); num_years];
    let regular_rate =
        vec![expected_returns(params.target_allocation.regular_portfolio.tpaw); num_years];

    let savings = get_net_present_value(&bonds_rate, &params.by_year.savings);
    let lmp = get_net_present_value(&bonds_rate, &vec![params.lmp; num_years]);
    let essential = get_net_present_value(&bonds_rate, &params.by_year.withdrawals_essential);
    let discretionary =
        get_net_present_value(&regular_rate, &params.by_year.withdrawals_discretionary);

    PreCalculationsForTPAW {
        net_present_value: TPAWNetPresentValue {
            savings,
            withdrawals: NetPresentValueForWithdrawals {
                lmp,
                essential,
                discretionary,
            },
        },
    }
}

fn pre_calculations_for_spaw(params: &Params) -> PreCalculationsForSPAW {
    let expected_returns = blend_returns(&params.expected_returns);
    let rate: Vec<f64> = params
        .target_allocation
        .regular_portfolio
        .spaw
        .iter()
        .map(|x| expected_returns(*x))
        .collect();
    let n = rate.len();
    let savings = get_net_present_value(&rate, &params.by_year.savings);
    let lmp = get_net_present_value(&rate, &vec![params.lmp; n]);
    let essential = get_net_present_value(&rate, &params.by_year.withdrawals_essential);
    let discretionary = get_net_present_value(&rate, &params.by_year.withdrawals_discretionary);
    let mut legacy_amount_by_year = vec![0.0; n];
    legacy_amount_by_year[n - 1] = params.legacy_target / (1.0 + rate[n - 1]);
    let legacy = get_net_present_value(&rate, &legacy_amount_by_year);

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
            params.spending_tilt,
        ),
    }
}

fn get_net_present_value(r: &[f64], amounts: &[f64]) -> NetPresentValueOfSequence {
    let n = amounts.len();
    assert!(r.len() == n);
    let mut with_current_year = vec![0.0; n];
    let mut without_current_year = vec![0.0; n];

    for i in (0..n).rev() {
        without_current_year[i] = if i == n - 1 {
            0.0
        } else {
            with_current_year[i + 1] / (1.0 + r[i])
        };
        with_current_year[i] = amounts[i] + without_current_year[i];
    }

    NetPresentValueOfSequence {
        with_current_year,
        without_current_year,
    }
}

fn cumulative_1_plus_g_over_1_plus_r(r: &[f64], g: f64) -> Vec<f64> {
    let n = r.len();
    let mut result = vec![0.0; n];
    for i in (0..n).rev() {
        let x = if i == n - 1 {
            0.0
        } else {
            (result[i + 1] * (1.0 + g)) / (1.0 + r[i])
        };
        result[i] = 1.0 + x
    }
    result
}
