use core::f64;

use serde::{Deserialize, Serialize};
use tsify::Tsify;

use crate::{
    constants::DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT,
    expected_value_of_returns::annual_non_log_to_monthly_non_log_return_rate,
    linear_fns::linear_fn_from_points,
    plan_params::{self, GlidePath, PlanParams, Risk},
    shared_types::{MonthAndStocks, Stocks, StocksAndBonds},
};

use super::process_returns_stats_for_planning::ReturnsStatsForPlanningProcessed;

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct RiskProcessed_TPAW {
    pub full_glide_path: Vec<TPAWGlidePathEntry>,
    pub allocation: Vec<f64>,
    pub allocation_for_legacy: Stocks,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct RiskProcessed_TPAWAndSPAW {
    pub monthly_spending_tilt: Vec<f64>,
    pub lmp: f64,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct RiskProcessed_SPAWAndSWR {
    pub allocation: Vec<f64>,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(tag = "type", rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub enum RiskProcessed_SWR_Withdrawal {
    #[serde(rename_all = "camelCase")]
    AsPercent { percent: f64 },
    #[serde(rename_all = "camelCase")]
    AsAmount { amount: f64 },
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct RiskProcessed_SWR {
    pub monthly_withdrawal: RiskProcessed_SWR_Withdrawal,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct RiskProcessed {
    pub tpaw: RiskProcessed_TPAW,
    #[serde(rename = "tpawAndSPAW")]
    pub tpaw_and_spaw: RiskProcessed_TPAWAndSPAW,
    #[serde(rename = "spawAndSWR")]
    pub spaw_and_swr: RiskProcessed_SPAWAndSWR,
    pub swr: RiskProcessed_SWR,
}

pub fn process_risk(
    plan_params: &PlanParams,
    returns_stats_for_planning: &ReturnsStatsForPlanningProcessed,
) -> RiskProcessed {
    let PlanParams {
        constants,
        ages,
        risk,
        ..
    } = plan_params;
    let r_mu_sigma_pow2 = _RMuSigmaPow2::new(&returns_stats_for_planning);
    let tpaw_glide_path = _get_tpaw_glide_path(constants, ages, risk, &r_mu_sigma_pow2);

    let tpaw_and_spaw = RiskProcessed_TPAWAndSPAW {
        monthly_spending_tilt: match plan_params.advanced.strategy {
            plan_params::Strategy::SPAW => {
                vec![
                    annual_non_log_to_monthly_non_log_return_rate(risk.spaw.annual_spending_tilt);
                    ages.simulation_months.num_months as usize
                ]
            }
            _ => tpaw_glide_path
                .iter()
                .map(|x| match &x.clamped {
                    Some(clamped) => clamped.monthly_spending_tilt,
                    None => x.unclamped.monthly_spending_tilt,
                })
                .collect(),
        },
        // TODO: Remove. No processing is happening
        lmp: risk.tpaw_and_spaw.lmp,
    };

    let tpaw = RiskProcessed_TPAW {
        allocation: tpaw_glide_path
            .iter()
            .map(|x| match &x.clamped {
                Some(clamped) => clamped.stock_allocation,
                None => x.unclamped.stock_allocation,
            })
            .collect(),
        allocation_for_legacy: Stocks {
            stocks: _unclamped_mertons_formula(
                &r_mu_sigma_pow2,
                {
                    let rra = risk.tpaw.risk_tolerance.at20
                        + risk.tpaw.risk_tolerance.for_legacy_as_delta_from_at20;
                    _risk_tolerance_to_rra_with_infinity_at_zero(constants, rra)
                },
                0.0, // Does not matter
                0.0, // Does not matter
            )
            .stock_allocation
            .clamp(0.0, 1.0),
        },
        full_glide_path: tpaw_glide_path,
    };

    let spaw_and_swr = RiskProcessed_SPAWAndSWR {
        allocation: _process_glide_path(&risk.spaw_and_swr.allocation, &ages.simulation_months),
    };

    let swr = RiskProcessed_SWR {
        monthly_withdrawal: match &risk.swr.withdrawal {
            plan_params::Risk_SWR_Withdrawal::Default => RiskProcessed_SWR_Withdrawal::AsPercent {
                percent: DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(
                    ages.simulation_months.num_withdrawal_months,
                ) / 12.0,
            },
            plan_params::Risk_SWR_Withdrawal::AsPercentPerYear { percent_per_year } => {
                RiskProcessed_SWR_Withdrawal::AsPercent {
                    percent: percent_per_year / 12.0,
                }
            }
            plan_params::Risk_SWR_Withdrawal::AsAmountPerMonth { amount_per_month } => {
                RiskProcessed_SWR_Withdrawal::AsAmount {
                    amount: *amount_per_month,
                }
            }
        },
    };

    RiskProcessed {
        tpaw,
        tpaw_and_spaw,
        spaw_and_swr,
        swr,
    }
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TPAWGlidePathEntryUnclampedInfo {
    pub risk_tolerance: f64,
    pub rra: f64,
    pub monthly_spending_tilt: f64,
    pub stock_allocation: f64,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TPAWGlidePathEntryClampedInfo {
    pub risk_tolerance_by_inversion: f64,
    pub rra: f64,
    pub monthly_spending_tilt: f64,
    pub stock_allocation: f64,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TPAWGlidePathEntry {
    #[serde(rename = "minRRA")]
    pub min_rra: f64,
    pub unclamped: TPAWGlidePathEntryUnclampedInfo,
    pub clamped: Option<TPAWGlidePathEntryClampedInfo>,
}

pub fn _get_tpaw_glide_path(
    constants: &plan_params::Constants,
    ages: &plan_params::Ages,
    risk: &plan_params::Risk,
    r_mu_sigma_pow2: &_RMuSigmaPow2,
) -> Vec<TPAWGlidePathEntry> {
    let num_months = ages.simulation_months.num_months;
    let get_risk_tolerance_from_mfn = |mfn: i64| {
        0.0_f64.max(if ages.longer_lived_person.max_age_in_months > 20 * 12 {
            linear_fn_from_points(
                20.0 * 12.0,
                risk.tpaw.risk_tolerance.at20,
                ages.longer_lived_person.max_age_in_months as f64,
                risk.tpaw.risk_tolerance.at20 + risk.tpaw.risk_tolerance.delta_at_max_age,
            )((mfn + ages.longer_lived_person.current_age_in_months) as f64)
        } else {
            risk.tpaw.risk_tolerance.at20
        })
    };

    let _curr_mertons_formula = |rra: f64| {
        _unclamped_mertons_formula(
            r_mu_sigma_pow2,
            rra,
            risk.tpaw.time_preference,
            risk.tpaw.additional_annual_spending_tilt,
        )
    };

    let mertons_formula_by_mfn = |mfn: i64, min_rra: f64| -> TPAWGlidePathEntry {
        let unclamped = {
            let risk_tolerance = get_risk_tolerance_from_mfn(mfn);
            let rra = _risk_tolerance_to_rra_with_infinity_at_zero(constants, risk_tolerance);
            let _MertonsFormulaResult {
                monthly_spending_tilt,
                stock_allocation,
            } = _curr_mertons_formula(rra);
            TPAWGlidePathEntryUnclampedInfo {
                risk_tolerance,
                rra,
                monthly_spending_tilt,
                stock_allocation,
            }
        };
        if min_rra > unclamped.rra {
            TPAWGlidePathEntry {
                min_rra,
                clamped: Some({
                    let rra = min_rra.max(unclamped.rra);
                    let _MertonsFormulaResult {
                        monthly_spending_tilt,
                        stock_allocation,
                    } = _curr_mertons_formula(rra);
                    TPAWGlidePathEntryClampedInfo {
                        rra,
                        risk_tolerance_by_inversion:
                            _risk_tolerance_to_rra_without_infinity_at_zero_inverse(constants, rra),
                        monthly_spending_tilt,
                        stock_allocation,
                    }
                }),
                unclamped,
            }
        } else {
            TPAWGlidePathEntry {
                min_rra,
                unclamped: TPAWGlidePathEntryUnclampedInfo {
                    risk_tolerance: unclamped.risk_tolerance,
                    rra: unclamped.rra,
                    monthly_spending_tilt: unclamped.monthly_spending_tilt,
                    stock_allocation: {
                        let nominally_clamped = unclamped.stock_allocation.clamp(0.0, 1.0);
                        // stockAllocation should already be in the range for [0, 1], but for
                        // floating point imprecision.
                        assert!(nominally_clamped - unclamped.stock_allocation < 1e-10);
                        nominally_clamped
                    },
                },
                clamped: None,
            }
        }
    };

    // mu:stocks, r:bonds.
    let equity_premium = r_mu_sigma_pow2.mu - r_mu_sigma_pow2.r;
    match equity_premium.total_cmp(&0.0) {
        std::cmp::Ordering::Less => {
            // Mertons formula yields a stock allocation of -Infinity and 0 for rra of 0
            // and Infinity respectively. But we don't really want to handle this case
            // using Merton's formula, because negative stock allocation means leverage,
            // which we don't allow. We want instead to completely ignore stocks. We do
            // this by bringing equity premium to 0.
            _get_tpaw_glide_path(
                constants,
                ages,
                risk,
                &_RMuSigmaPow2 {
                    // bonds
                    r: r_mu_sigma_pow2.r,
                    // stocks
                    mu: r_mu_sigma_pow2.r, // Note this is r, not mu
                    sigma_pow2: r_mu_sigma_pow2.sigma_pow2,
                },
            )
        }
        std::cmp::Ordering::Equal => {
            // Mertons formula yields a stock allocation of 0 for the full rra range
            // from 0 to Infinity. This is always in range, so no need to clamp.
            (0..num_months)
                .map(|mfn| mertons_formula_by_mfn(mfn, 0.0))
                .collect()
        }
        std::cmp::Ordering::Greater => {
            // Mertons formula yields a stock allocation of Infinity and 0 for rra of 0
            // and Infinity respectively. We can go through mertons formula, but we clamp
            // rra to the rra that results in a stock allocation of 1.
            let min_rra = _get_rra_by_inverting_unclamped_mertons_formula(r_mu_sigma_pow2, 1.0);
            (0..num_months)
                .map(|mfn| mertons_formula_by_mfn(mfn, min_rra))
                .collect()
        }
    }
}

pub struct _RMuSigmaPow2 {
    r: f64,
    mu: f64,
    sigma_pow2: f64,
}

impl _RMuSigmaPow2 {
    pub fn new(returns_stats_for_planning_processed: &ReturnsStatsForPlanningProcessed) -> Self {
        let StocksAndBonds { stocks, bonds } = returns_stats_for_planning_processed;
        _RMuSigmaPow2 {
            r: bonds.empirical_annual_non_log_expected_return_info.value,
            mu: stocks.empirical_annual_non_log_expected_return_info.value,
            sigma_pow2: stocks.empirical_annual_log_variance,
        }
    }
}

struct _MertonsFormulaResult {
    monthly_spending_tilt: f64,
    stock_allocation: f64,
}

// Does not clamp stock allocation.
fn _unclamped_mertons_formula(
    r_mu_sigma_pow2: &_RMuSigmaPow2,
    // Take in rra not riskTolerance because we need to support rra that are not
    // in range of the defined riskTolerances because we might get this rra from
    // running the inverse().
    rra: f64,
    time_preference: f64,
    additional_annual_spending_tilt: f64,
) -> _MertonsFormulaResult {
    let _RMuSigmaPow2 { r, mu, sigma_pow2 } = r_mu_sigma_pow2;
    let get_result = |annual_spending_tilt: f64, stock_allocation: f64| _MertonsFormulaResult {
        monthly_spending_tilt: annual_non_log_to_monthly_non_log_return_rate(
            annual_spending_tilt + additional_annual_spending_tilt,
        ),
        stock_allocation,
    };
    if rra.is_infinite() {
        return get_result(0.0, 0.0);
    };

    let gamma = rra;
    let stock_allocation = (mu - r) / (sigma_pow2 * gamma);
    let rho = time_preference;
    let nu = (rho - (1.0 - gamma) * ((mu - r).powi(2) / (2.0 * sigma_pow2 * gamma) + r)) / gamma;

    let r_of_portfolio = mu * stock_allocation + r * (1.0 - stock_allocation);
    let annual_spending_tilt = r_of_portfolio - nu;

    get_result(annual_spending_tilt, stock_allocation)
}

// RRA and not risk tolerance because risk tolerance may be < 0.
fn _get_rra_by_inverting_unclamped_mertons_formula(
    r_mu_sigma_pow2: &_RMuSigmaPow2,
    stock_allocation: f64,
) -> f64 {
    let _RMuSigmaPow2 { r, mu, sigma_pow2 } = r_mu_sigma_pow2;
    (mu - r) / (sigma_pow2 * stock_allocation)
}

fn _process_glide_path(
    glide_path: &GlidePath,
    simulation_month: &plan_params::SimulationMonths,
) -> Vec<f64> {
    let result: Vec<f64> = std::iter::once(
        &(MonthAndStocks {
            month: 0,
            stocks: glide_path.now.stocks,
        }),
    )
    .chain(glide_path.intermediate.iter())
    .chain(std::iter::once(
        &(MonthAndStocks {
            month: simulation_month.last_month_as_mfn,
            stocks: glide_path.end.stocks,
        }),
    ))
    .collect::<Vec<_>>()
    .windows(2)
    .flat_map(|x| {
        let from = x[0];
        let to = x[1];
        let fun = linear_fn_from_points(from.month as f64, from.stocks, to.month as f64, to.stocks);
        (from.month..to.month).map(move |month| fun(month as f64))
    })
    .chain(std::iter::once(glide_path.end.stocks))
    .collect();
    assert!(result.len() == simulation_month.num_months as usize);
    result
}

fn _ln_one_over_x(x: f64) -> f64 {
    (1.0 / x).ln()
}

struct _ShiftAndScale {
    shift: f64,
    scale: f64,
}

fn _risk_tolerance_to_rra_shift_and_scale(constants: &plan_params::Constants) -> _ShiftAndScale {
    let num_points = constants.risk_tolerance_num_integer_values_starting_from_0;
    let start_rra = constants.risk_tolerance_start_rra;
    let end_rra = constants.risk_tolerance_end_rra;
    let shift = _ln_one_over_x(start_rra);
    let scale = (num_points as f64 - 2.0) / (_ln_one_over_x(end_rra) - _ln_one_over_x(start_rra));
    _ShiftAndScale { shift, scale }
}

fn _risk_tolerance_to_rra_without_infinity_at_zero(
    constants: &plan_params::Constants,
    risk_tolerance: f64,
) -> f64 {
    let _ShiftAndScale { shift, scale } = _risk_tolerance_to_rra_shift_and_scale(constants);
    1.0 / ((risk_tolerance - 1.0) / scale + shift).exp()
}

fn _risk_tolerance_to_rra_without_infinity_at_zero_inverse(
    constants: &plan_params::Constants,
    rra: f64,
) -> f64 {
    let _ShiftAndScale { shift, scale } = _risk_tolerance_to_rra_shift_and_scale(constants);
    (_ln_one_over_x(rra) - shift) * scale + 1.0
}

fn _risk_tolerance_to_rra_with_infinity_at_zero(
    constants: &plan_params::Constants,
    risk_tolerance: f64,
) -> f64 {
    if risk_tolerance == 0.0 {
        f64::INFINITY
    } else {
        _risk_tolerance_to_rra_without_infinity_at_zero(constants, risk_tolerance)
    }
}
