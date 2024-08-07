pub mod process_plan_params;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::shared_types::{MonthAndStocks, SimpleRange, Stocks};

// -----------------------
// -----CONST SUPPORT ----
// -----------------------
// https://github.com/serde-rs/serde/issues/745#issuecomment-1450072069
#[derive(Clone, Debug)]
pub struct Boolean<const B: bool>;

impl<const B: bool> Serialize for Boolean<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_bool(B)
    }
}
impl<'de, const B: bool> Deserialize<'de> for Boolean<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = bool::deserialize(deserializer)?;
        if value == B {
            Ok(Boolean::<B>)
        } else {
            Err(serde::de::Error::custom("not true or false"))
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConstU64<const U64: u64>;
impl<const U64: u64> Serialize for ConstU64<U64> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(U64)
    }
}
impl<'de, const U64: u64> Deserialize<'de> for ConstU64<U64> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        if value == U64 {
            Ok(ConstU64::<U64>)
        } else {
            Err(serde::de::Error::custom("not constant u64"))
        }
    }
}

// ----------------
// ---- CONSTANTS ----
// ----------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Constants {
    pub max_age: i64,
    pub risk_tolerance_num_integer_values_starting_from_0:i64,
    #[serde(rename = "riskToleranceStartRRA")]
    pub risk_tolerance_start_rra:f64,
    #[serde(rename = "riskToleranceEndRRA")]
    pub risk_tolerance_end_rra:f64,
}

// ----------------
// ---- MISC ----
// ----------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum DialogPosition {
    Age,
    CurrentPortfolioBalance,
    FutureSavings,
    IncomeDuringRetirement,
    ShowResults,
    ShowAllInputs,
    Done,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub enum PersonType {
    Person1,
    Person2,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LabeledAmountUntimed {
    pub label: Option<String>,
    pub amount: f64,
    pub nominal: bool,
    pub id: String,
    pub sort_index: u64,
    pub color_index: u64,
}

// --------------
// ---- AGES ----
// --------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Ages_Person_Retirement {
    #[serde(rename = "ageAsMFNIfInFutureElseNull")]
    pub age_as_mfn_if_in_future_else_null: Option<i64>,
    #[serde(rename = "ageAsMFNIfSpecifiedElseNull")]
    pub age_as_mfn_if_specified_else_null: Option<i64>,
    pub is_retired: bool,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SimulationMonths {
    pub num_months: i64,
    pub num_withdrawal_months: i64,
    #[serde(rename = "lastMonthAsMFN")]
    pub last_month_as_mfn: i64,
    #[serde(rename = "withdrawalStartMonthAsMFN")]
    pub withdrawal_start_month_as_mfn: i64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct Ages_ValidMonthRanges {
    #[serde(rename = "futureSavingsAsMFN")]
    pub future_savings_as_mfn: Option<SimpleRange<i64>>,
    #[serde(rename = "incomeDuringRetirementAsMFN")]
    pub income_during_retirement_as_mfn: SimpleRange<i64>,
    #[serde(rename = "extraSpending")]
    pub extra_spending: SimpleRange<i64>,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct Ages_Person {
    pub max_age_in_months: i64,
    pub current_age_in_months: i64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Ages {
    pub simulation_months: SimulationMonths,
    pub longer_lived_person: Ages_Person,
}

// ----------------
// ---- MONTHS ----
// ----------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AmountAndTiming {
    #[serde(rename_all = "camelCase")]
    OneTime { amount: f64, month: i64 },
    #[serde(rename_all = "camelCase")]
    Recurring {
        base_amount: f64,
        month_range: Option<SimpleRange<i64>>,
        valid_month_range: SimpleRange<i64>,
    },
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LabeledAmountTimed {
    pub id: String,
    pub nominal: bool,
    pub amount_and_timing: AmountAndTiming,
}

// ----------------
// ---- WEALTH ----
// ----------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Wealth {
    pub future_savings: Vec<LabeledAmountTimed>,
    pub income_during_retirement: Vec<LabeledAmountTimed>,
}

// ---------------------------------
// ---- ADJUSTMENTS_TO_SPENDING ----
// ---------------------------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExtraSpending {
    pub essential: Vec<LabeledAmountTimed>,
    pub discretionary: Vec<LabeledAmountTimed>,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Legacy {
    pub total: f64,
    pub external: Vec<LabeledAmountUntimed>,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct AdjustmentsToSpending_TPAWAndSPAW {
    pub monthly_spending_ceiling: Option<f64>,
    pub monthly_spending_floor: Option<f64>,
    pub legacy: Legacy,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AdjustmentsToSpending {
    pub extra_spending: ExtraSpending,
    #[serde(rename = "tpawAndSPAW")]
    pub tpaw_and_spaw: AdjustmentsToSpending_TPAWAndSPAW,
}

// --------------
// ---- RISK ----
// --------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GlidePath {
    pub now: Stocks,
    pub intermediate: Vec<MonthAndStocks>,
    pub end: Stocks,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct Risk_TPAW_RiskTolerance {
    pub at20: f64,
    pub delta_at_max_age: f64,
    pub for_legacy_as_delta_from_at20: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct Risk_TPAW {
    pub risk_tolerance: Risk_TPAW_RiskTolerance,
    pub time_preference: f64,
    pub additional_annual_spending_tilt: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct Risk_TPAWAndSPAW {
    pub lmp: f64,
}
#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct Risk_SPAW {
    pub annual_spending_tilt: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct Risk_SPAWAndSWR {
    pub allocation: GlidePath,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub enum Risk_SWR_Withdrawal {
    #[serde(rename_all = "camelCase")]
    AsPercentPerYear {
        percent_per_year: f64,
    },
    #[serde(rename_all = "camelCase")]
    AsAmountPerMonth {
        amount_per_month: f64,
    },
    Default,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct Risk_SWR {
    pub withdrawal: Risk_SWR_Withdrawal,
}
#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Risk {
    pub tpaw: Risk_TPAW,
    #[serde(rename = "tpawAndSPAW")]
    pub tpaw_and_spaw: Risk_TPAWAndSPAW,
    pub spaw: Risk_SPAW,
    #[serde(rename = "spawAndSWR")]
    pub spaw_and_swr: Risk_SPAWAndSWR,
    pub swr: Risk_SWR,
}

// ------------------
// ---- ADVANCED ----
// ------------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub enum ExpectedReturnsForPlanningCustomStocksBase {
    RegressionPrediction,
    ConservativeEstimate,
    #[serde(rename = "1/CAPE")]
    OneOverCAPE,
    Historical,
}
#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExpectedReturnsForPlanningCustomStocks {
    pub base: ExpectedReturnsForPlanningCustomStocksBase,
    pub delta: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub enum ExpectedReturnsForPlanningCustomBondsBase {
    #[serde(rename = "20YearTIPSYield")]
    TwentyYearTIPSYield,
    Historical,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExpectedReturnsForPlanningCustomBonds {
    pub base: ExpectedReturnsForPlanningCustomBondsBase,
    pub delta: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ExpectedReturnsForPlanning_EmpiricalAnnualNonLog {
    #[serde(rename = "regressionPrediction,20YearTIPSYield")]
    #[allow(non_camel_case_types)]
    RegressionPrediction_20YearTIPSYield,
    #[serde(rename = "conservativeEstimate,20YearTIPSYield")]
    #[allow(non_camel_case_types)]
    ConservativeEstimate_20YearTIPSYield,
    #[serde(rename = "1/CAPE,20YearTIPSYield")]
    #[allow(non_camel_case_types)]
    OneOverCAPE_20YearTIPSYield,
    #[serde(rename = "historical")]
    Historical,
    #[serde(rename_all = "camelCase")]
    FixedEquityPremium { equity_premium: f64 },
    #[serde(rename_all = "camelCase")]
    Custom {
        stocks: ExpectedReturnsForPlanningCustomStocks,
        bonds: ExpectedReturnsForPlanningCustomBonds,
    },
    #[serde(rename_all = "camelCase")]
    Fixed { stocks: f64, bonds: f64 },
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct ExpectedReturnsForPlanning {
    pub empirical_annual_non_log: ExpectedReturnsForPlanning_EmpiricalAnnualNonLog,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Log {
    pub log: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ScaleLog {
    pub scale: Log,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct ReturnsStatsForPlanning_StandardDeviation {
    pub stocks: ScaleLog,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ReturnsStatsForPlanning {
    pub expected_value: ExpectedReturnsForPlanning,
    pub standard_deviation: ReturnsStatsForPlanning_StandardDeviation,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
#[tsify(from_wasm_abi)]
pub struct HistoricalReturnsAdjustment_StandardDeviation {
    pub bonds: ScaleLog,
    pub override_to_fixed_for_testing: bool,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct HistoricalReturnsAdjustment {
    pub standard_deviation: HistoricalReturnsAdjustment_StandardDeviation,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum Sampling {
    #[serde(rename_all = "camelCase")]
    MonteCarlo {
        block_size: u64,
        stagger_run_starts: bool,
    },
    Historical,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AnnualInfaltion {
    #[serde(rename_all = "camelCase")]
    Suggested,
    #[serde(rename_all = "camelCase")]
    Manual { value: f64 },
}
#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
pub enum Strategy {
    TPAW,
    SPAW,
    SWR,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Advanced {
    pub returns_stats_for_planning: ReturnsStatsForPlanning,
    pub historical_returns_adjustment: HistoricalReturnsAdjustment,
    pub sampling: Sampling,
    pub annual_inflation: AnnualInfaltion,
    pub strategy: Strategy,
}

// ---------------------
// ---- PLAN PARAMS ----
// ---------------------
#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename = "PlanParamsRust", rename_all = "camelCase")]
#[tsify(from_wasm_abi, into_wasm_abi)]
pub struct PlanParams {
    pub constants: Constants,
    pub ages: Ages,
    pub wealth: Wealth,
    pub adjustments_to_spending: AdjustmentsToSpending,
    pub risk: Risk,
    pub advanced: Advanced,
}
