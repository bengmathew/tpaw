#![allow(non_camel_case_types)]

use crate::{
    cuda_bridge::{
        PlanParamsCuda_AdjustmentsToSpending, PlanParamsCuda_AdjustmentsToSpending_TPAWAndSPAW,
    },
    cuda_bridge_utils::f_cuda_currency,
    nominal_to_real::nominal_to_real,
    simulate::plan_params_server::PlanParamsServer,
    wire::{
        WireAdjustmentsToSpendingProcessed, WireAdjustmentsToSpendingProcessedTpawAndSpaw,
        WireAdjustmentsToSpendingProcessedTpawAndSpawLegacy,
    },
};

use super::process_annual_inflation::InflationProcessed;

#[derive(Clone)]
pub struct AdjustmentsToSpendingProcessed_TPAWAndSPAW_Legacy {
    pub external: f64,
    pub from_portfolio: f64,
}

impl From<AdjustmentsToSpendingProcessed_TPAWAndSPAW_Legacy>
    for WireAdjustmentsToSpendingProcessedTpawAndSpawLegacy
{
    fn from(other: AdjustmentsToSpendingProcessed_TPAWAndSPAW_Legacy) -> Self {
        Self {
            external: other.external,
            target: other.from_portfolio,
        }
    }
}

#[derive(Clone)]
pub struct AdjustmentsToSpendingProcessed_TPAWAndSPAW {
    pub spending_ceiling: Option<f64>,
    pub spending_floor: Option<f64>,
    pub legacy: AdjustmentsToSpendingProcessed_TPAWAndSPAW_Legacy,
}

impl From<&AdjustmentsToSpendingProcessed_TPAWAndSPAW>
    for PlanParamsCuda_AdjustmentsToSpending_TPAWAndSPAW
{
    fn from(other: &AdjustmentsToSpendingProcessed_TPAWAndSPAW) -> Self {
        Self {
            spending_ceiling: other.spending_ceiling.into(),
            spending_floor: other.spending_floor.into(),
            legacy: other.legacy.from_portfolio as f_cuda_currency,
        }
    }
}

impl From<AdjustmentsToSpendingProcessed_TPAWAndSPAW>
    for WireAdjustmentsToSpendingProcessedTpawAndSpaw
{
    fn from(other: AdjustmentsToSpendingProcessed_TPAWAndSPAW) -> Self {
        Self {
            legacy: other.legacy.into(),
        }
    }
}

#[derive(Clone)]
pub struct AdjustmentsToSpendingProcessed {
    pub tpaw_and_spaw: AdjustmentsToSpendingProcessed_TPAWAndSPAW,
}

impl From<&AdjustmentsToSpendingProcessed> for PlanParamsCuda_AdjustmentsToSpending {
    fn from(other: &AdjustmentsToSpendingProcessed) -> Self {
        Self {
            tpaw_and_spaw: (&other.tpaw_and_spaw).into(),
        }
    }
}

impl From<AdjustmentsToSpendingProcessed> for WireAdjustmentsToSpendingProcessed {
    fn from(other: AdjustmentsToSpendingProcessed) -> Self {
        Self {
            tpaw_and_spaw: other.tpaw_and_spaw.into(),
        }
    }
}

pub fn process_adjustments_to_spending(
    plan_params_server: &PlanParamsServer,
    inflation_processed: &InflationProcessed,
) -> AdjustmentsToSpendingProcessed {
    AdjustmentsToSpendingProcessed {
        tpaw_and_spaw: {
            let src = &plan_params_server.adjustments_to_spending.tpaw_and_spaw;
            AdjustmentsToSpendingProcessed_TPAWAndSPAW {
                spending_ceiling: src.spending_ceiling.into(),
                spending_floor: src.spending_floor.into(),
                legacy: {
                    let src = &plan_params_server
                        .adjustments_to_spending
                        .tpaw_and_spaw
                        .legacy;
                    let external: f64 = src
                        .external
                        .iter()
                        .map(|x| {
                            nominal_to_real(
                                x.amount,
                                x.is_nominal,
                                inflation_processed.monthly,
                                plan_params_server.ages.simulation_months.num_months,
                            )
                        })
                        .sum();
                    AdjustmentsToSpendingProcessed_TPAWAndSPAW_Legacy {
                        external: external,
                        from_portfolio: (src.total - external).max(0.0),
                    }
                },
            }
        },
    }
}
