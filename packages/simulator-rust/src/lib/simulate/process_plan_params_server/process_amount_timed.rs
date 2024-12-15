#![allow(non_camel_case_types)]
use crate::simulate::plan_params_server::{
    PlanParamsServer_AmountTimed, PlanParamsServer_AmountTimed_DeltaEveryRecurrence,
};
use crate::to_float_wire::ToFloatWire;
use crate::utils::nominal_to_real::nominal_to_real;
use crate::wire::{
    WireAmountTimedProcessed, WireAmountTimedProcessedGroup, WireAmountTimedProcessedSingle,
};
use std::collections::HashMap;

#[derive(Clone)]
pub struct AmountTimedProcessed_Group {
    pub total_by_mfn: Vec<f64>,
    pub by_id: HashMap<String, Vec<f64>>,
}

impl From<AmountTimedProcessed_Group> for WireAmountTimedProcessedGroup {
    fn from(other: AmountTimedProcessed_Group) -> Self {
        Self {
            total_x100: other
                .total_by_mfn
                .iter()
                .map(|x| x.to_float_wire(100))
                .collect(),
            by_id: other
                .by_id
                .iter()
                .map(|(k, v)| WireAmountTimedProcessedSingle {
                    id: k.to_string(),
                    values_x100: v.iter().map(|x| x.to_float_wire(100)).collect(),
                })
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct AmountTimedProcessed {
    pub essential_expenses: AmountTimedProcessed_Group,
    pub discretionary_expenses: AmountTimedProcessed_Group,
    pub income_during_retirement: AmountTimedProcessed_Group,
    pub future_savings: AmountTimedProcessed_Group,
}

impl From<AmountTimedProcessed> for WireAmountTimedProcessed {
    fn from(other: AmountTimedProcessed) -> Self {
        Self {
            wealth_income_during_retirement: other.income_during_retirement.into(),
            wealth_future_savings: other.future_savings.into(),
            extra_expenses_essential: other.essential_expenses.into(),
            extra_expenses_discretionary: other.discretionary_expenses.into(),
        }
    }
}

pub fn process_amount_timed(
    num_months: u32,
    future_savings: &Vec<PlanParamsServer_AmountTimed>,
    income_during_retirement: &Vec<PlanParamsServer_AmountTimed>,
    essential_expenses: &Vec<PlanParamsServer_AmountTimed>,
    discretionary_expenses: &Vec<PlanParamsServer_AmountTimed>,
    monthly_inflation: f64,
) -> AmountTimedProcessed {
    AmountTimedProcessed {
        future_savings: process_entries(future_savings, num_months, monthly_inflation),
        income_during_retirement: process_entries(
            income_during_retirement,
            num_months,
            monthly_inflation,
        ),
        essential_expenses: process_entries(essential_expenses, num_months, monthly_inflation),
        discretionary_expenses: process_entries(
            discretionary_expenses,
            num_months,
            monthly_inflation,
        ),
    }
}

pub fn process_entries(
    amounts: &Vec<PlanParamsServer_AmountTimed>,
    num_months: u32,
    monthly_inflation: f64,
) -> AmountTimedProcessed_Group {
    let by_id: HashMap<String, Vec<f64>> = amounts
        .iter()
        .map(|entry| {
            (
                entry.id.clone(),
                process_entry(entry, num_months, monthly_inflation),
            )
        })
        .collect();

    let mut total_by_mfn = vec![0.0; num_months as usize];
    by_id.values().for_each(|v| {
        for i in 0..num_months as usize {
            total_by_mfn[i] += v[i];
        }
    });

    AmountTimedProcessed_Group {
        total_by_mfn,
        by_id,
    }
}

pub fn process_entry(
    x: &PlanParamsServer_AmountTimed,
    num_months: u32,
    monthly_inflation: f64,
) -> Vec<f64> {
    let PlanParamsServer_AmountTimed {
        is_nominal,
        month_range,
        valid_month_range,
        every_x_months,
        base_amount,
        delta_every_recurrence,
        ..
    } = x;
    let mut values = vec![0.0; num_months as usize];
    let mut next_value = *base_amount;
    if let Some(month_range) = month_range {
        for mfn in month_range.clone().into_iter() {
            let hit = (mfn - month_range.start()) % every_x_months == 0
                && *valid_month_range.start() <= mfn
                && mfn <= *valid_month_range.end();
            if hit {
                values[mfn as usize] =
                    nominal_to_real(next_value, *is_nominal, monthly_inflation, mfn as u32);
                next_value = match delta_every_recurrence {
                    PlanParamsServer_AmountTimed_DeltaEveryRecurrence::Percent(percent) => {
                        next_value * (1.0 + percent)
                    }
                    PlanParamsServer_AmountTimed_DeltaEveryRecurrence::Amount(amount) => {
                        next_value + amount
                    }
                }
            }
        }
    }
    values
}
