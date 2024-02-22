pub mod average_annual_real_earnings_for_sp500_for_10_years;
pub mod v1;
pub mod v2;

use crate::shared_types::{Log, StocksAndBonds};
use crate::{
    random::generate_random_index_sequences,
    return_series::{periodize_log_returns, Mean, Stats},
    shared_types::SlopeAndIntercept,
};
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

pub struct RawMonthlyNonLogSeriesEntry {
    pub stocks: f32,
    pub bonds: f32,
}

impl RawMonthlyNonLogSeriesEntry {
    // Don't actually store year and month to reduce bundle size.
    pub const fn new(
        _year: u16,
        _month: u8,
        stocks: f32,
        bonds: f32,
    ) -> RawMonthlyNonLogSeriesEntry {
        Self { stocks, bonds }
    }
}

pub fn process_raw_monthly_non_log_series(
    data: &[RawMonthlyNonLogSeriesEntry],
) -> Log<StocksAndBonds<Vec<f64>>> {
    let stocks: Vec<f64> = data.iter().map(|x| (x.stocks as f64).ln_1p()).collect();
    let bonds: Vec<f64> = data.iter().map(|x| (x.bonds as f64).ln_1p()).collect();
    Log {
        log: StocksAndBonds { stocks, bonds },
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RawCAPESeriesEntry {
    pub year: u16,
    pub month: u8,
    pub cape: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct FiveTenTwentyThirtyYearsSlopeAndIntercept {
    pub five_year: SlopeAndIntercept,
    pub ten_year: SlopeAndIntercept,
    pub twenty_year: SlopeAndIntercept,
    pub thirty_year: SlopeAndIntercept,
}

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FiveTenTwentyThirtyYearsF64 {
    pub five_year: f64,
    pub ten_year: f64,
    pub twenty_year: f64,
    pub thirty_year: f64,
}

impl FiveTenTwentyThirtyYearsF64 {
    pub fn iter(&self) -> impl Iterator<Item = f64> {
        vec![
            self.five_year,
            self.ten_year,
            self.twenty_year,
            self.thirty_year,
        ]
        .into_iter()
    }
}

#[derive(Serialize, Deserialize, Copy, Clone, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct CAPEBasedRegressionResults {
    pub full: FiveTenTwentyThirtyYearsF64,
    pub restricted: FiveTenTwentyThirtyYearsF64,
}

impl CAPEBasedRegressionResults {
    pub fn map<F: Fn(f64) -> f64>(&self, fun: F) -> CAPEBasedRegressionResults {
        CAPEBasedRegressionResults {
            full: FiveTenTwentyThirtyYearsF64 {
                five_year: fun(self.full.five_year),
                ten_year: fun(self.full.ten_year),
                twenty_year: fun(self.full.twenty_year),
                thirty_year: fun(self.full.thirty_year),
            },
            restricted: FiveTenTwentyThirtyYearsF64 {
                five_year: fun(self.restricted.five_year),
                ten_year: fun(self.restricted.ten_year),
                twenty_year: fun(self.restricted.twenty_year),
                thirty_year: fun(self.restricted.thirty_year),
            },
        }
    }
}

#[derive(Debug)]
pub struct AnnualLogMeanFromOneOverCAPERegressionInfo {
    pub full: FiveTenTwentyThirtyYearsSlopeAndIntercept,
    pub restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept,
}

impl AnnualLogMeanFromOneOverCAPERegressionInfo {
    pub fn new_zero() -> Self {
        let slope_and_intercept = SlopeAndIntercept {
            slope: 0.0,
            intercept: 0.0,
        };
        let for_window = FiveTenTwentyThirtyYearsSlopeAndIntercept {
            five_year: slope_and_intercept,
            ten_year: slope_and_intercept,
            twenty_year: slope_and_intercept,
            thirty_year: slope_and_intercept,
        };

        Self {
            full: for_window,
            restricted: for_window,
        }
    }

    pub fn y(&self, log_one_over_cape: f64) -> CAPEBasedRegressionResults {
        CAPEBasedRegressionResults {
            full: FiveTenTwentyThirtyYearsF64 {
                five_year: self.full.five_year.y(log_one_over_cape),
                ten_year: self.full.ten_year.y(log_one_over_cape),
                twenty_year: self.full.twenty_year.y(log_one_over_cape),
                thirty_year: self.full.thirty_year.y(log_one_over_cape),
            },
            restricted: FiveTenTwentyThirtyYearsF64 {
                five_year: self.restricted.five_year.y(log_one_over_cape),
                ten_year: self.restricted.ten_year.y(log_one_over_cape),
                twenty_year: self.restricted.twenty_year.y(log_one_over_cape),
                thirty_year: self.restricted.thirty_year.y(log_one_over_cape),
            },
        }
    }
}
impl CAPEBasedRegressionResults {
    pub fn iter(&self) -> impl Iterator<Item = f64> {
        self.full.iter().chain(self.restricted.iter())
    }
}

pub struct EmpiricalStats64 {
    pub annual_non_log_returns_mean: f64,
    pub annual_log_returns_variance: f64,
}

#[derive(Debug)]
pub struct EmpiricalStats32 {
    pub annual_non_log_returns_mean: f32,
    pub annual_log_returns_variance: f32,
}

impl EmpiricalStats32 {
    pub const fn new(annual_non_log_returns_mean: f32, annual_log_returns_variance: f32) -> Self {
        Self {
            annual_non_log_returns_mean,
            annual_log_returns_variance,
        }
    }
}

pub fn get_empirical_stats_for_block_size(
    monthly_log_returns: &[f64],
    block_size: usize,
    num_runs: usize,
    months_per_run: usize,
    stagger_run_starts: bool,
    seed: u64,
) -> EmpiricalStats32 {
    let mut sampled_annual_log_returns = {
        let sampled_monthly_log_returns: Vec<f64> = {
            let indexes = generate_random_index_sequences(
                seed,
                0,
                num_runs,
                months_per_run,
                block_size,
                monthly_log_returns.len(),
                stagger_run_starts,
            );
            indexes
                .into_iter()
                .flatten()
                .map(|i| monthly_log_returns[i])
                .collect()
        };
        periodize_log_returns(&sampled_monthly_log_returns, 12)
    };

    let annual_log_returns_variance =
        Stats::from_series(&sampled_annual_log_returns).variance as f32;

    let annual_non_log_returns_mean = {
        for i in 0..sampled_annual_log_returns.len() {
            sampled_annual_log_returns[i] = sampled_annual_log_returns[i].exp_m1();
        }
        sampled_annual_log_returns.iter().mean() as f32
    };
    return EmpiricalStats32 {
        annual_non_log_returns_mean,
        annual_log_returns_variance,
    };
}
