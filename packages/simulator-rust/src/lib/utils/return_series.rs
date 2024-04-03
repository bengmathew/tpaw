use core::iter::Iterator;
use js_sys::*;
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub mean: f64,
    pub variance: f64,
    pub standard_deviation: f64,
    pub n: usize,
}

impl Stats {
    pub fn zero() -> Stats {
        Stats {
            mean: 0.0,
            variance: 0.0,
            standard_deviation: 0.0,
            n: 0,
        }
    }
    pub fn from_series(series: &[f64]) -> Stats {
        let n = series.len();
        let sum = series.iter().sum::<f64>();
        let mean = sum / n as f64;
        let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let standard_deviation = variance.sqrt();
        Stats {
            mean,
            variance,
            standard_deviation,
            n,
        }
    }
}

// https://stackoverflow.com/a/43925422/2771609
pub trait Mean {
    fn mean(self) -> f64;
}

impl<F, T> Mean for T
where
    T: std::iter::Iterator<Item = F>,
    F: std::borrow::Borrow<f64>,
{
    fn mean(self) -> f64 {
        self.zip(1..).fold(0., |s, (e, i)| {
            (*e.borrow() + s * (i - 1) as f64) / i as f64
        })
    }
}

pub struct SeriesAndStats {
    pub series: Vec<f64>,
    pub stats: Stats,
}

impl SeriesAndStats {
    pub fn from_series(series: Vec<f64>) -> SeriesAndStats {
        SeriesAndStats {
            stats: Stats::from_series(&series),
            series,
        }
    }
}

#[inline(always)]
pub fn ln_1p_series(series: &[f64]) -> Vec<f64> {
    series.iter().map(|x| x.ln_1p()).collect()
}

#[inline(always)]
pub fn exp_m1_series(series: &[f64]) -> Vec<f64> {
    series.iter().map(|x| x.exp_m1()).collect()
}

pub fn periodize_log_returns(log_returns: &[f64], period_size: usize) -> Vec<f64> {
    let mut prev = 0.0;
    let mut prev_gross = log_returns[0..(period_size - 1)].iter().sum::<f64>();
    let n = log_returns.len() - period_size + 1;
    log_returns[0..n]
        .iter()
        .enumerate()
        .map(|(i, curr)| {
            prev_gross = prev_gross - prev + log_returns[i + period_size - 1];
            prev = *curr;
            return prev_gross;
        })
        .collect()
}

pub fn adjust_log_returns(
    log_returns_and_stats: &SeriesAndStats,
    target_mean_of_log: f64,
    standard_deviation_of_log_scale: f64,
) -> Vec<f64> {
    let delta = log_returns_and_stats.stats.mean - target_mean_of_log;
    log_returns_and_stats
        .series
        .iter()
        .map(|x| {
            let unscaled = x - delta;
            target_mean_of_log + (unscaled - target_mean_of_log) * standard_deviation_of_log_scale
        })
        .collect::<Vec<f64>>()
}

// ---- TESTS ----
#[cfg(test)]
mod tests {
    use crate::{
        historical_monthly_returns::{
            data::v1::V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
            get_historical_monthly_returns_info,
        },
        return_series::{adjust_log_returns, exp_m1_series, periodize_log_returns, SeriesAndStats},
    };

    #[test]
    fn test_periodize_log_returns() {
        fn alt_periodize(series: &[f64], period_size: usize) -> Vec<f64> {
            (0..series.len() - period_size + 1)
                .map(|i| {
                    series[i..i + period_size]
                        .iter()
                        .map(|x| 1.0 + x)
                        .product::<f64>()
                        - 1.0
                })
                .collect::<Vec<f64>>()
        }
        let original = &get_historical_monthly_returns_info(
            V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
        )
        .returns
        .stocks
        .log;

        let annualized1 = alt_periodize(&exp_m1_series(&original.series), 3);
        let annualized2 = exp_m1_series(&periodize_log_returns(&original.series, 3));
        let diff: Vec<f64> = annualized1
            .iter()
            .zip(annualized2.iter())
            .map(|(x, y)| (x - y).abs())
            .collect();
        assert!(diff.iter().all(|x| (*x).abs() < 1e-10));
    }

    #[test]
    fn test_adjust_log_returns_mean() {
        let original = &get_historical_monthly_returns_info(
            V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
        )
        .returns
        .stocks
        .log;
        let target_mean = original.stats.mean + 1.5;
        let adjusted = SeriesAndStats::from_series(adjust_log_returns(&original, target_mean, 1.0));
        assert!((adjusted.stats.mean - target_mean).abs() < 1e-10);
        assert!(
            (adjusted.stats.standard_deviation - original.stats.standard_deviation).abs() < 1e-10
        );
    }

    #[test]
    fn test_adjust_log_returns_scale() {
        let original = &get_historical_monthly_returns_info(
            V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
        )
        .returns
        .stocks
        .log;
        let scale = 0.5;
        let adjusted =
            SeriesAndStats::from_series(adjust_log_returns(&original, original.stats.mean, scale));
        assert!((adjusted.stats.mean - original.stats.mean).abs() < 1e-10);
        assert!(
            (adjusted.stats.standard_deviation - original.stats.standard_deviation * scale).abs()
                < 1e-10
        );
    }

    #[test]
    fn test_adjust_log_returns_mean_and_scale() {
        let original = &get_historical_monthly_returns_info(
            V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
        )
        .returns
        .stocks
        .log;

        let target_mean = original.stats.mean + 1.5;
        let scale = 0.5;
        let adjusted =
            SeriesAndStats::from_series(adjust_log_returns(&original, target_mean, scale));
        assert!((adjusted.stats.mean - target_mean).abs() < 1e-10);
        assert!(
            (adjusted.stats.standard_deviation - original.stats.standard_deviation * scale).abs()
                < 1e-10
        );
    }
}
