use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tsify::{declare, Tsify};
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
pub struct LabeledAmount {
    pub label: Option<String>,
    pub value: f64,
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
pub struct Ages_Person {
    #[serde(rename = "monthOfBirthAsMFN")]
    pub month_of_birth_as_mfn: i64,
    #[serde(rename = "maxAgeAsMFN")]
    pub max_age_as_mfn: i64,
    pub retirement: Ages_Person_Retirement,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Ages_SimulationMonths {
    pub num_months: i64,
    #[serde(rename = "lastMonthAsMFN")]
    pub last_month_as_mfn: i64,
    #[serde(rename = "withdrawalStartMonthAsMFN")]
    pub withdrawal_start_month_as_mfn: i64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
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
pub struct Ages {
    pub person1: Ages_Person,
    pub person2: Option<Ages_Person>,
    pub simulation_months: Ages_SimulationMonths,
    pub valid_month_ranges: Ages_ValidMonthRanges,
}

// ----------------
// ---- MONTHS ----
// ----------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ValueForMonthRange {
    pub label: Option<String>,
    #[serde(rename = "monthRange")]
    pub month_range: SimpleRange<i64>,
    pub value: f64,
    pub nominal: bool,
    pub id: String,
    pub sort_index: u64,
    pub color_index: u64,
}

#[declare]
pub type ValueForMonthRanges = HashMap<String, ValueForMonthRange>;

// ----------------
// ---- WEALTH ----
// ----------------
#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(untagged)]
pub enum PortfolioBalance {
    #[serde(rename = "true", rename_all = "camelCase")]
    UpdatedHere {
        #[tsify(type = "true")]
        updated_here: bool,
        amount: f64,
    },
    #[serde(rename = "false", rename_all = "camelCase")]
    NotUpdatedHere {
        #[tsify(type = "false")]
        updated_here: bool,
        updated_at_id: String,
        updated_to: f64,
        updated_at_timestamp: f64,
    },
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Wealth {
    pub portfolio_balance: PortfolioBalance,
    pub future_savings: ValueForMonthRanges,
    pub income_during_retirement: ValueForMonthRanges,
}

// ---------------------------------
// ---- ADJUSTMENTS_TO_SPENDING ----
// ---------------------------------

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExtraSpending {
    pub essential: ValueForMonthRanges,
    pub discretionary: ValueForMonthRanges,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Legacy {
    pub total: f64,
    pub external: HashMap<String, LabeledAmount>,
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
pub struct GlidePathIntermediateEntry {
    pub id: String,
    pub index_to_sort_by_added: i64,
    pub month: i64,
    pub stocks: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GlidePath {
    pub start: MonthAndStocks,
    pub intermediate: HashMap<String, GlidePathIntermediateEntry>,
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
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ExpectedReturnsForPlanning {
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
    Manual { stocks: f64, bonds: f64 },
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct HistoricalMonthlyLogReturnsAdjustment_StandardDeviation_Stocks {
    pub scale: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
pub struct HistoricalMonthlyLogReturnsAdjustment_StandardDeviation_Bonds {
    pub enable_volatility: bool,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[allow(non_camel_case_types)]
#[tsify(from_wasm_abi)]
pub struct HistoricalMonthlyLogReturnsAdjustment_StandardDeviation {
    pub stocks: HistoricalMonthlyLogReturnsAdjustment_StandardDeviation_Stocks,
    pub bonds: HistoricalMonthlyLogReturnsAdjustment_StandardDeviation_Bonds,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct HistoricalMonthlyLogReturnsAdjustment {
    pub standard_deviation: HistoricalMonthlyLogReturnsAdjustment_StandardDeviation,
    pub override_to_fixed_for_testing: bool,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub enum SamplingType {
    MonteCarlo,
    Historical,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ForMonteCarlo {
    pub block_size: u64,
    pub stagger_run_starts: bool,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct Sampling {
    #[serde(rename = "type")]
    pub kind: SamplingType,
    pub for_monte_carlo: ForMonteCarlo,
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
    pub expected_returns_for_planning: ExpectedReturnsForPlanning,
    pub historical_monthly_log_returns_adjustment: HistoricalMonthlyLogReturnsAdjustment,
    pub sampling: Sampling,
    pub annual_inflation: AnnualInfaltion,
    pub strategy: Strategy,
}

// -----------------
// ---- RESULTS ----
// ------------------
#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DisplayedAssetAllocation {
    pub stocks: f64,
}

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PlanParamsStoredResults {
    pub displayed_asset_allocation: DisplayedAssetAllocation,
}

// ---------------------
// ---- PLAN PARAMS ----
// ---------------------
#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, into_wasm_abi)]
pub struct PlanParamsNormalized {
    #[tsify(type = "27")]
    pub v: ConstU64<27>,
    pub timestamp: f64,
    pub dialog_position_nominal: DialogPosition,
    pub ages: Ages,
    pub wealth: Wealth,
    pub adjustments_to_spending: AdjustmentsToSpending,
    pub risk: Risk,
    pub advanced: Advanced,
    pub results: Option<PlanParamsStoredResults>,
}
