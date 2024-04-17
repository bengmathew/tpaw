use serde::Serialize;
use simulator::{
    historical_monthly_returns::{
        HistoricalMonthlyLogReturnsAdjustedInfoArgs, HistoricalMonthlyLogReturnsAdjustedStats,
    },
    plan_params::process_plan_params::{
        plan_params_processed::{self, PlanParamsProcessed},
        process_by_month_params,
    },
    utils::{
        return_series::Stats,
        shared_types::{LogAndNonLog, SimpleRange, StocksAndBonds},
    },
};

#[derive(Serialize)]
pub struct ProcessedValueForMonthRange {
    pub id: String,
    pub values: Vec<f64>,
    pub valid_range: Option<SimpleRange<i64>>,
}

#[derive(Serialize)]
pub struct ProcessedValueForMonthRanges {
    pub by_id: Vec<ProcessedValueForMonthRange>,
    pub total: Vec<f64>,
}

fn for_month_range(
    ppp: process_by_month_params::ProcessedValueForMonthRange,
) -> ProcessedValueForMonthRange {
    ProcessedValueForMonthRange {
        id: ppp.id,
        values: ppp.values.into_iter().map(|v| v as f64).collect(),
        valid_range: ppp.valid_range,
    }
}

fn for_month_ranges(
    ppp: process_by_month_params::ProcessedValueForMonthRanges,
) -> ProcessedValueForMonthRanges {
    ProcessedValueForMonthRanges {
        by_id: ppp.by_id.into_iter().map(for_month_range).collect(),
        total: ppp.total.into_iter().map(|v| v as f64).collect(),
    }
}

#[derive(Serialize)]
pub struct ProcessedByMonthParamsWealth {
    pub future_savings: ProcessedValueForMonthRanges,
    pub income_during_retirement: ProcessedValueForMonthRanges,
    pub total: Vec<f64>,
}

#[derive(Serialize)]
pub struct ProcessedByMonthParamsAdjustmentsToSpendingExtraSpending {
    pub essential: ProcessedValueForMonthRanges,
    pub discretionary: ProcessedValueForMonthRanges,
}

#[derive(Serialize)]
pub struct ProcessedByMonthParamsAdjustmentsToSpending {
    pub extra_spending: ProcessedByMonthParamsAdjustmentsToSpendingExtraSpending,
}

#[derive(Serialize)]
pub struct ProcessedByMonthParamsRiskTPAWAndSPAW {
    pub lmp: Vec<f64>,
}

#[derive(Serialize)]
pub struct ProcessedByMonthParamsRisk {
    pub tpaw_and_spaw: ProcessedByMonthParamsRiskTPAWAndSPAW,
}

#[derive(Serialize)]
pub struct ProcessedByMonthParams {
    pub wealth: ProcessedByMonthParamsWealth,
    pub adjustments_to_spending: ProcessedByMonthParamsAdjustmentsToSpending,
    pub risk: ProcessedByMonthParamsRisk,
}

#[derive(Serialize)]
pub struct HistoricalMonthlyLogReturnsAdjustedInfo {
    // pub log_series: Vec<f64>,
    // pub non_log_series: Vec<f64>,
    pub stats: HistoricalMonthlyLogReturnsAdjustedStats,
    pub args: HistoricalMonthlyLogReturnsAdjustedInfoArgs,
    pub src_annualized_stats: LogAndNonLog<Stats>,
}

// --------------------------- RISK ------------------------

#[derive(Serialize)]
pub struct TPAWGlidePathEntryClampedInfo {
    pub risk_tolerance_by_inversion: f64,
    pub rra: f64,
    pub monthly_spending_tilt: f64,
    pub stock_allocation: f64,
}

#[derive(Serialize)]
pub struct TPAWGlidePathEntryUnclampedInfo {
    pub risk_tolerance: f64,
    pub rra: f64,
    pub monthly_spending_tilt: f64,
    pub stock_allocation: f64,
}

#[derive(Serialize)]
pub struct TPAWGlidePathEntry {
    #[serde(rename = "minRRA")]
    pub min_rra: f64,
    pub unclamped: TPAWGlidePathEntryUnclampedInfo,
    pub clamped: Option<TPAWGlidePathEntryClampedInfo>,
}

#[derive(Serialize)]
pub struct PPPSmall {
    pub historical_returns_adjusted: StocksAndBonds<HistoricalMonthlyLogReturnsAdjustedInfo>,
    // pub risk: ProcessedByMonthParamsRisk,
    pub by_month: ProcessedByMonthParams,
}

impl PPPSmall {
    pub fn new(ppp: PlanParamsProcessed) -> PPPSmall {
        Self {
            historical_returns_adjusted: StocksAndBonds {
                stocks: HistoricalMonthlyLogReturnsAdjustedInfo {
                    stats: ppp.historical_returns_adjusted.stocks.stats,
                    args: ppp.historical_returns_adjusted.stocks.args,
                    src_annualized_stats: ppp
                        .historical_returns_adjusted
                        .stocks
                        .src_annualized_stats,
                },
                bonds: HistoricalMonthlyLogReturnsAdjustedInfo {
                    stats: ppp.historical_returns_adjusted.bonds.stats,
                    args: ppp.historical_returns_adjusted.bonds.args,
                    src_annualized_stats: ppp
                        .historical_returns_adjusted
                        .bonds
                        .src_annualized_stats,
                },
            },
            by_month: ProcessedByMonthParams {
                wealth: ProcessedByMonthParamsWealth {
                    future_savings: for_month_ranges(ppp.by_month.wealth.future_savings),
                    income_during_retirement: for_month_ranges(
                        ppp.by_month.wealth.income_during_retirement,
                    ),
                    total: ppp
                        .by_month
                        .wealth
                        .total
                        .into_iter()
                        .map(|v| v as f64)
                        .collect(),
                },
                adjustments_to_spending: ProcessedByMonthParamsAdjustmentsToSpending {
                    extra_spending: ProcessedByMonthParamsAdjustmentsToSpendingExtraSpending {
                        essential: for_month_ranges(
                            ppp.by_month
                                .adjustments_to_spending
                                .extra_spending
                                .essential,
                        ),
                        discretionary: for_month_ranges(
                            ppp.by_month
                                .adjustments_to_spending
                                .extra_spending
                                .discretionary,
                        ),
                    },
                },
                risk: ProcessedByMonthParamsRisk {
                    tpaw_and_spaw: ProcessedByMonthParamsRiskTPAWAndSPAW {
                        lmp: ppp
                            .by_month
                            .risk
                            .tpaw_and_spaw
                            .lmp
                            .into_iter()
                            .map(|v| v as f64)
                            .collect(),
                    },
                },
            },
        }
    }
}
