#![allow(non_camel_case_types)]

use crate::cuda_bridge::{PlanParamsCuda_Risk, PlanParamsCuda_Risk_SWR, PlanParamsCuda_Risk_SWR_WithdrawalType_Amount, PlanParamsCuda_Risk_SWR_WithdrawalType_Percent, PlanParamsCuda_Risk_TPAW};
use crate::cuda_bridge_utils::f_cuda;
use crate::simulate::plan_params_server::{
    PlanParamsServer_Ages, PlanParamsServer_Ages_SimulationMonths, PlanParamsServer_Constants,
    PlanParamsServer_GlidePath, PlanParamsServer_GlidePath_Intermediate, PlanParamsServer_Risk,
    PlanParamsServer_Risk_SWR_Withdrawal, PlanParamsServer_Risk_TPAW,
};
use crate::utils::{
    expected_value_of_returns::annual_non_log_to_monthly_non_log_return_rate,
    linear_fns::linear_fn_from_points,
};
use crate::wire::{WireRiskProcessed, WireRiskProcessedTpaw};

#[derive(Clone)]
pub struct RiskProcessed_TPAW {
    pub risk_tolerance_by_mfn: Vec<f64>,
    pub rra_unclamped_at_20_including_pos_infinity: f64,
    pub rra_unclamped_including_pos_infinity_by_mfn_f64: Vec<f64>,
    pub rra_unclamped_including_pos_infinity_by_mfn_fcuda: Vec<f_cuda>,
    pub legacy_rra_including_pos_infinity: f64,
}

impl From<RiskProcessed_TPAW> for WireRiskProcessedTpaw {
    fn from(risk: RiskProcessed_TPAW) -> Self {
        Self {
            risk_tolerance_by_mfn: risk.risk_tolerance_by_mfn,
            rra_unclamped_at_20_including_pos_infinity: risk
                .rra_unclamped_at_20_including_pos_infinity,
            rra_unclamped_including_pos_infinity_by_mfn: risk
                .rra_unclamped_including_pos_infinity_by_mfn_f64,
            legacy_rra_including_pos_infinity: risk.legacy_rra_including_pos_infinity,
        }
    }
}

#[derive(Clone)]
pub struct RiskProcessed_SPAW {
    pub monthly_spending_tilt_by_mfn_f64: Vec<f64>,
    pub monthly_spending_tilt_by_mfn_fcuda: Vec<f_cuda>,
}

#[derive(Clone)]
pub struct RiskProcessed_SPAWAndSWR {
    pub stock_allocation_savings_portfolio_by_mfn_f64: Vec<f64>,
    pub stock_allocation_savings_portfolio_by_mfn_fcuda: Vec<f_cuda>,
}

#[derive(Clone)]
pub enum RiskProcessed_SWR_Withdrawal {
    AsPercent { percent: f64 },
    AsAmount { amount: f64 },
}

#[derive(Clone)]
pub struct RiskProcessed_SWR {
    pub monthly_withdrawal: RiskProcessed_SWR_Withdrawal,
}

impl From<&RiskProcessed_SWR> for PlanParamsCuda_Risk_SWR {
    fn from(risk: &RiskProcessed_SWR) -> Self {
        match risk.monthly_withdrawal {
            RiskProcessed_SWR_Withdrawal::AsPercent { percent } => Self {
                withdrawal_type: PlanParamsCuda_Risk_SWR_WithdrawalType_Percent,
                withdrawal_as_percent_or_amount: percent,
            },
            RiskProcessed_SWR_Withdrawal::AsAmount { amount } => Self {
                withdrawal_type: PlanParamsCuda_Risk_SWR_WithdrawalType_Amount,
                withdrawal_as_percent_or_amount: amount,
            },
        }
    }
}

#[derive(Clone)]
pub struct RiskProcessed {
    pub tpaw: RiskProcessed_TPAW,
    pub spaw: RiskProcessed_SPAW,
    pub spaw_and_swr: RiskProcessed_SPAWAndSWR,
    pub swr: RiskProcessed_SWR,
}

impl From<RiskProcessed> for WireRiskProcessed {
    fn from(risk: RiskProcessed) -> Self {
        Self {
            tpaw: risk.tpaw.into(),
        }
    }
}

impl RiskProcessed {
    pub fn into_cuda(
        &self,
        plan_params_server_risk: &PlanParamsServer_Risk,
    ) -> PlanParamsCuda_Risk {
        PlanParamsCuda_Risk {
            tpaw: {
                let src = plan_params_server_risk.tpaw;
                PlanParamsCuda_Risk_TPAW {
                    legacy_rra_including_pos_infinity: self.tpaw.legacy_rra_including_pos_infinity
                        as f_cuda,
                    time_preference: src.time_preference as f_cuda,
                    annual_additional_spending_tilt: src.additional_annual_spending_tilt as f_cuda,
                }
            },
            swr: (&self.swr).into(),
        }
    }
}

pub fn process_risk(
    constants: &PlanParamsServer_Constants,
    ages: &PlanParamsServer_Ages,
    risk: &PlanParamsServer_Risk,
) -> RiskProcessed {
    let tpaw = _process_tpaw(constants, ages, &risk.tpaw);

    let spaw = {
        let monthly_spending_tilt_by_mfn_f64 =
            vec![
                annual_non_log_to_monthly_non_log_return_rate(risk.spaw.annual_spending_tilt);
                ages.simulation_months.num_months as usize
            ];
        RiskProcessed_SPAW {
            monthly_spending_tilt_by_mfn_fcuda: monthly_spending_tilt_by_mfn_f64
                .iter()
                .map(|x| *x as f_cuda)
                .collect(),
            monthly_spending_tilt_by_mfn_f64,
        }
    };

    let spaw_and_swr = {
        let stock_allocation_savings_portfolio_by_mfn_f64 =
            _process_glide_path(&risk.spaw_and_swr.stock_allocation, &ages.simulation_months);
        RiskProcessed_SPAWAndSWR {
            stock_allocation_savings_portfolio_by_mfn_fcuda:
                stock_allocation_savings_portfolio_by_mfn_f64
                    .iter()
                    .map(|x| *x as f_cuda)
                    .collect(),
            stock_allocation_savings_portfolio_by_mfn_f64,
        }
    };

    let swr = RiskProcessed_SWR {
        monthly_withdrawal: match &risk.swr.withdrawal {
            PlanParamsServer_Risk_SWR_Withdrawal::PercentPerYear(percent_per_year) => {
                RiskProcessed_SWR_Withdrawal::AsPercent {
                    percent: percent_per_year / 12.0,
                }
            }
            PlanParamsServer_Risk_SWR_Withdrawal::AmountPerMonth(amount_per_month) => {
                RiskProcessed_SWR_Withdrawal::AsAmount {
                    amount: *amount_per_month,
                }
            }
        },
    };

    RiskProcessed {
        tpaw,
        spaw,
        spaw_and_swr,
        swr,
    }
}

pub fn _process_tpaw(
    constants: &PlanParamsServer_Constants,
    ages: &PlanParamsServer_Ages,
    risk: &PlanParamsServer_Risk_TPAW,
) -> RiskProcessed_TPAW {
    let rra_unclamped_at_20_including_pos_infinity =
        _risk_tolerance_to_rra_with_infinity_at_zero(constants, risk.risk_tolerance.at20);

    let (risk_tolerance_by_mfn, rra_unclamped_including_pos_infinity_by_mfn_f64): (
        Vec<f64>,
        Vec<f64>,
    ) = {
        let num_months = ages.simulation_months.num_months;
        let get_risk_tolerance_from_mfn = |mfn: u32| {
            0.0_f64.max(if ages.longer_lived_person.max_age > 20 * 12 {
                linear_fn_from_points(
                    20.0 * 12.0,
                    risk.risk_tolerance.at20,
                    ages.longer_lived_person.max_age as f64,
                    risk.risk_tolerance.at20 + risk.risk_tolerance.delta_at_max_age,
                )((mfn + ages.longer_lived_person.current_age) as f64)
            } else {
                risk.risk_tolerance.at20
            })
        };
        (0..num_months)
            .map(|mfn| {
                let risk_tolerance = get_risk_tolerance_from_mfn(mfn);
                let rra_unclamped =
                    _risk_tolerance_to_rra_with_infinity_at_zero(constants, risk_tolerance);
                (risk_tolerance, rra_unclamped)
            })
            .unzip()
    };

    let legacy_rra_including_pos_infinity = {
        let risk_tolerance =
            risk.risk_tolerance.at20 + risk.risk_tolerance.for_legacy_as_delta_from_at20;
        _risk_tolerance_to_rra_with_infinity_at_zero(constants, risk_tolerance)
    };

    RiskProcessed_TPAW {
        risk_tolerance_by_mfn,
        rra_unclamped_at_20_including_pos_infinity,
        rra_unclamped_including_pos_infinity_by_mfn_fcuda:
            rra_unclamped_including_pos_infinity_by_mfn_f64
                .iter()
                .map(|x| *x as f_cuda)
                .collect(),
        rra_unclamped_including_pos_infinity_by_mfn_f64,
        legacy_rra_including_pos_infinity,
    }
}

fn _process_glide_path(
    glide_path: &PlanParamsServer_GlidePath,
    simulation_months: &PlanParamsServer_Ages_SimulationMonths,
) -> Vec<f64> {
    let result: Vec<f64> = std::iter::once(
        &(PlanParamsServer_GlidePath_Intermediate {
            month: 0,
            value: glide_path.now,
        }),
    )
    .chain(glide_path.intermediate.iter())
    .chain(std::iter::once(
        &(PlanParamsServer_GlidePath_Intermediate {
            month: simulation_months.last_month(),
            value: glide_path.end,
        }),
    ))
    .collect::<Vec<_>>()
    .windows(2)
    .flat_map(|x| {
        let from = x[0];
        let to = x[1];
        let fun = linear_fn_from_points(from.month as f64, from.value, to.month as f64, to.value);
        (from.month..to.month).map(move |month| fun(month as f64))
    })
    .chain(std::iter::once(glide_path.end))
    .collect();
    assert!(result.len() == simulation_months.num_months as usize);
    result
}

fn _ln_one_over_x(x: f64) -> f64 {
    (1.0 / x).ln()
}

struct _ShiftAndScale {
    shift: f64,
    scale: f64,
}

fn _risk_tolerance_to_rra_shift_and_scale(
    constants: &PlanParamsServer_Constants,
) -> _ShiftAndScale {
    let num_points = constants.risk_tolerance_num_integer_values_starting_from_0;
    let start_rra = constants.risk_tolerance_start_rra;
    let end_rra = constants.risk_tolerance_end_rra;
    let shift = _ln_one_over_x(start_rra);
    let scale = (num_points as f64 - 2.0) / (_ln_one_over_x(end_rra) - _ln_one_over_x(start_rra));
    _ShiftAndScale { shift, scale }
}

fn _risk_tolerance_to_rra_without_infinity_at_zero(
    constants: &PlanParamsServer_Constants,
    risk_tolerance: f64,
) -> f64 {
    let _ShiftAndScale { shift, scale } = _risk_tolerance_to_rra_shift_and_scale(constants);
    1.0 / ((risk_tolerance - 1.0) / scale + shift).exp()
}

fn _risk_tolerance_to_rra_without_infinity_at_zero_inverse(
    constants: &PlanParamsServer_Constants,
    rra: f64,
) -> f64 {
    let _ShiftAndScale { shift, scale } = _risk_tolerance_to_rra_shift_and_scale(constants);
    (_ln_one_over_x(rra) - shift) * scale + 1.0
}

fn _risk_tolerance_to_rra_with_infinity_at_zero(
    constants: &PlanParamsServer_Constants,
    risk_tolerance: f64,
) -> f64 {
    if risk_tolerance == 0.0 {
        f64::INFINITY
    } else {
        _risk_tolerance_to_rra_without_infinity_at_zero(constants, risk_tolerance)
    }
}
