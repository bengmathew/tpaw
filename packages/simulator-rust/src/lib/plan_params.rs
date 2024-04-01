pub mod plan_params_rust;
pub mod process_plan_params;

// use serde::{Deserialize, Serialize};
// use std::collections::HashMap;
// use tsify::{declare, Tsify};
// use wasm_bindgen::prelude::*;

// use crate::{calendar_month::CalendarMonth, shared_types::MonthAndStocks};

// #[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
// #[serde(rename_all = "camelCase")]
// pub struct InMonths {
//     pub in_months: i64,
// }

// #[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
// #[serde(tag = "type", rename_all = "camelCase")]
// pub enum Ages {
//     #[serde(rename_all = "camelCase")]
//     RetiredWithNoRetirementDateSpecified {
//         month_of_birth: CalendarMonth,
//         max_age: InMonths,
//     },
//     #[serde(rename_all = "camelCase")]
//     RetirementDateSpecified {
//         month_of_birth: CalendarMonth,
//         retirement_age: InMonths,
//         max_age: InMonths,
//     },
// }

// #[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
// #[serde(rename_all = "camelCase")]
// pub struct Person {
//     pub ages: Ages,
// }

// #[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
// #[serde(untagged, rename_all = "camelCase")]
// pub enum People {
//     #[serde(rename_all = "camelCase")]
//     WithoutPartner {
//         #[tsify(type = "false")]
//         with_partner: Boolean<false>,
//         person1: Person,
//     },
//     #[serde(rename_all = "camelCase")]
//     WithPartner {
//         #[tsify(type = "true")]
//         with_partner: Boolean<true>,
//         person1: Person,
//         person2: Person,
//         withdrawal_start: PersonType,
//     },
// }

// #[derive(Serialize, Deserialize, Tsify)]
// #[serde(rename_all = "camelCase")]
// #[tsify(from_wasm_abi)]
// pub struct PlanParams {
//     #[tsify(type = "27")]
//     pub v: ConstU64<27>,
//     pub timestamp: f64,
//     pub dialog_position_nominal: DialogPosition,
//     pub people: People,
//     pub wealth: Wealth<MonthRange>,
//     pub adjustments_to_spending: AdjustmentsToSpending<MonthRange>,
//     pub risk: Risk<GlidePath<CalendarMonth, Month>>,
//     pub advanced: Advanced,
//     pub results: Option<PlanParamsStoredResults>,
// }
