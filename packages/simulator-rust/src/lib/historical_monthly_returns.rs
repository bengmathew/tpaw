pub mod data;

use self::data::process_raw_monthly_non_log_series;
use self::data::v1::v1_raw_monthly_non_log_series::V1_RAW_MONTHLY_NON_LOG_SERIES;
use self::data::v1::V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS;
use self::data::v2::v2_annual_log_mean_from_one_over_cape_regression_info_stocks::V2_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS;
use self::data::v2::v2_empirical_stats_by_block_size_bonds::V2_EMPIRICAL_STATS_BY_BLOCK_SIZE_BONDS;
use self::data::v2::v2_empirical_stats_by_block_size_stocks::V2_EMPIRICAL_STATS_BY_BLOCK_SIZE_STOCKS;
use self::data::v2::v2_raw_monthly_non_log_series::V2_RAW_MONTHLY_NON_LOG_SERIES;
use self::data::v2::V2_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS;
use self::data::v3::v3_annual_log_mean_from_one_over_cape_regression_info_stocks::V3_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS;
use self::data::v3::v3_empirical_stats_by_block_size_bonds::V3_EMPIRICAL_STATS_BY_BLOCK_SIZE_BONDS;
use self::data::v3::v3_empirical_stats_by_block_size_stocks::V3_EMPIRICAL_STATS_BY_BLOCK_SIZE_STOCKS;
use self::data::v3::v3_raw_monthly_non_log_series::V3_RAW_MONTHLY_NON_LOG_SERIES;
use self::data::v3::V3_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS;
use self::data::{AnnualLogMeanFromOneOverCAPERegressionInfo, EmpiricalStats64};
use crate::historical_monthly_returns::data::v1::v1_raw_monthly_non_log_series::V1_RAW_MONTHLY_NON_LOG_SERIES_START;
use crate::historical_monthly_returns::data::v2::v2_raw_monthly_non_log_series::V2_RAW_MONTHLY_NON_LOG_SERIES_START;
use crate::historical_monthly_returns::data::v3::v3_raw_monthly_non_log_series::V3_RAW_MONTHLY_NON_LOG_SERIES_START;
use crate::shared_types::{SimpleRange, YearAndMonth};
use crate::{
    expected_value_of_returns::EmpiricalAnnualNonLogExpectedValueInfo,
    historical_monthly_returns::data::{
        v1::{
            v1_annual_log_mean_from_one_over_cape_regression_info_stocks::V1_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS,
            v1_empirical_stats_by_block_size_bonds::V1_EMPIRICAL_STATS_BY_BLOCK_SIZE_BONDS,
            v1_empirical_stats_by_block_size_stocks::V1_EMPIRICAL_STATS_BY_BLOCK_SIZE_STOCKS,
        },
        EmpiricalStats32,
    },
    return_series::{adjust_log_returns, periodize_log_returns, SeriesAndStats, Stats},
    shared_types::{LogAndNonLog, StocksAndBonds},
};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tsify::Tsify;
use wasm_bindgen::prelude::*;

pub struct HistoricalMonthlyReturns {
    pub log: SeriesAndStats,
    pub annualized_stats: LogAndNonLog<Stats>,
    pub empirical_stats_by_block_size: HashMap<usize, EmpiricalStats64>,
    pub annual_log_mean_from_one_over_cape_regression_info:
        AnnualLogMeanFromOneOverCAPERegressionInfo,
}

#[derive(Serialize, Deserialize, Tsify, Copy, Clone)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct HistoricalMonthlyLogReturnsAdjustedStats {
    pub log: Stats,
    pub annualized: LogAndNonLog<Stats>,
    pub empirical_annual_log_variance: f64,
    pub unadjusted_annualized: LogAndNonLog<Stats>,
}

#[derive(Serialize, Deserialize, Tsify, Clone)]
#[serde(rename_all = "camelCase")]
pub struct HistoricalMonthlyLogReturnsAdjustedInfo {
    #[serde(skip)]
    pub log_series: Vec<f64>,
    pub stats: HistoricalMonthlyLogReturnsAdjustedStats,
}

impl HistoricalMonthlyReturns {
    pub fn new(
        monthly_log_series: Vec<f64>,
        empirical_stats_by_block_size_arr: &[EmpiricalStats32],
        annual_log_mean_from_one_over_cape_regression_info:
        AnnualLogMeanFromOneOverCAPERegressionInfo,
    ) -> Self {
        let annualized_log_series = periodize_log_returns(&monthly_log_series, 12);

        Self {
            annualized_stats: LogAndNonLog {
                log: Stats::from_series(&annualized_log_series),
                non_log: {
                    let series: Vec<f64> =
                        annualized_log_series.iter().map(|x| x.exp_m1()).collect();
                    Stats::from_series(&series)
                },
            },
            log: SeriesAndStats::from_series(monthly_log_series),
            empirical_stats_by_block_size: {
                let mut result: HashMap<usize, EmpiricalStats64> = HashMap::new();
                empirical_stats_by_block_size_arr
                    .iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        if i != 0 {
                            result.insert(
                                i,
                                EmpiricalStats64 {
                                    annual_non_log_returns_mean: x.annual_non_log_returns_mean
                                        as f64,
                                    annual_log_returns_variance: x.annual_log_returns_variance
                                        as f64,
                                },
                            );
                        }
                    });
                result
            },
            annual_log_mean_from_one_over_cape_regression_info,
        }
    }

    pub fn get_shift_correction(
        &self,
        block_size: &Option<usize>,
        log_volatility_scale: f64,
    ) -> f64 {
        let empirical_unadjusted_annual_non_log_mean = match block_size {
            Some(block_size) => {
                self.empirical_stats_by_block_size
                    .get(&block_size)
                    .unwrap()
                    .annual_non_log_returns_mean
            }
            None => self.annualized_stats.non_log.mean,
        };
        (empirical_unadjusted_annual_non_log_mean.ln_1p() / 12.0 - self.log.stats.mean)
            * log_volatility_scale.powi(2)
    }

    pub fn get_empirical_annual_non_log_from_log_monthly_expected_value(
        &self,
        log_monthly_expected_value: f64,
        block_size: Option<usize>,
        log_volatility_scale: f64,
    ) -> EmpiricalAnnualNonLogExpectedValueInfo {
        let correction = self.get_shift_correction(&block_size, log_volatility_scale);
        EmpiricalAnnualNonLogExpectedValueInfo {
            value: ((log_monthly_expected_value + correction) * 12.0).exp_m1(),
            block_size,
            log_volatility_scale,
        }
    }

    pub fn adjust_log_returns(
        &self,
        empirical_annual_non_log_expected_value: &EmpiricalAnnualNonLogExpectedValueInfo,
    ) -> Vec<f64> {
        let EmpiricalAnnualNonLogExpectedValueInfo {
            block_size,
            log_volatility_scale,
            value: target_annual_non_log_mean,
        } = empirical_annual_non_log_expected_value;
        let correction = self.get_shift_correction(block_size, *log_volatility_scale);
        let monthly_log_mean = target_annual_non_log_mean.ln_1p() / 12.0 - correction;
        adjust_log_returns(&self.log, monthly_log_mean, *log_volatility_scale)
    }

    pub fn adjust_log_returns_detailed(
        &self,
        empirical_annual_non_log_return_info: &EmpiricalAnnualNonLogExpectedValueInfo,
    ) -> HistoricalMonthlyLogReturnsAdjustedInfo {
        let log_series = self.adjust_log_returns(empirical_annual_non_log_return_info);
        let annalized_log_series = &periodize_log_returns(&log_series, 12);
        let stats = HistoricalMonthlyLogReturnsAdjustedStats {
            log: Stats::from_series(&log_series),
            annualized: LogAndNonLog {
                log: Stats::from_series(&annalized_log_series),
                non_log: Stats::from_series(
                    &annalized_log_series
                        .iter()
                        .map(|x| x.exp_m1())
                        .collect::<Vec<f64>>(),
                ),
            },
            empirical_annual_log_variance: {
                let unscaled = match empirical_annual_non_log_return_info.block_size {
                    Some(block_size) => {
                        self.empirical_stats_by_block_size
                            .get(&block_size)
                            .unwrap()
                            .annual_log_returns_variance
                    }
                    None => self.annualized_stats.log.variance,
                };
                unscaled
                    * empirical_annual_non_log_return_info
                        .log_volatility_scale
                        .powi(2)
            },
            unadjusted_annualized: LogAndNonLog {
                log: self.annualized_stats.log,
                non_log: self.annualized_stats.non_log,
            },
        };
        HistoricalMonthlyLogReturnsAdjustedInfo { log_series, stats }
    }
}

pub enum HistoricalReturnsId {
    V1,
    V2,
    V3,
}

impl HistoricalReturnsId {
    pub fn as_str(&self) -> &'static str {
        match self {
            HistoricalReturnsId::V1 => "v1",
            HistoricalReturnsId::V2 => "v2",
            HistoricalReturnsId::V3 => "v3",
        }
    }
}
pub struct HistoricalReturnsInfo {
    pub id: HistoricalReturnsId,
    pub timestamp_ms: i64,
    pub month_range: SimpleRange<YearAndMonth>,
    pub returns: StocksAndBonds<HistoricalMonthlyReturns>,
}

fn get_all_historical_returns_infos() -> Vec<HistoricalReturnsInfo> {
    fn get_month_range(start: &YearAndMonth, n: usize) -> SimpleRange<YearAndMonth> {
        let end = start.add_months(n as i64 - 1);
        SimpleRange {
            start: start.clone(),
            end,
        }
    }
    let v1 = {
        let monthly_series = process_raw_monthly_non_log_series(&V1_RAW_MONTHLY_NON_LOG_SERIES);
        HistoricalReturnsInfo {
            id: HistoricalReturnsId::V1,
            timestamp_ms: V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
            month_range: get_month_range(
                &V1_RAW_MONTHLY_NON_LOG_SERIES_START,
                V1_RAW_MONTHLY_NON_LOG_SERIES.len(),
            ),
            returns: StocksAndBonds {
                stocks: HistoricalMonthlyReturns::new(
                    monthly_series.log.stocks,
                    &V1_EMPIRICAL_STATS_BY_BLOCK_SIZE_STOCKS,
                    V1_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS,
                ),
                bonds: HistoricalMonthlyReturns::new(
                    monthly_series.log.bonds,
                    &V1_EMPIRICAL_STATS_BY_BLOCK_SIZE_BONDS,
                    AnnualLogMeanFromOneOverCAPERegressionInfo::new_zero(),
                ),
            },
        }
    };

    let v2 = {
        let monthly_series = process_raw_monthly_non_log_series(&V2_RAW_MONTHLY_NON_LOG_SERIES);
        HistoricalReturnsInfo {
            id: HistoricalReturnsId::V2,
            timestamp_ms: V2_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
            month_range: get_month_range(
                &V2_RAW_MONTHLY_NON_LOG_SERIES_START,
                V2_RAW_MONTHLY_NON_LOG_SERIES.len(),
            ),
            returns: StocksAndBonds {
                stocks: HistoricalMonthlyReturns::new(
                    monthly_series.log.stocks,
                    &V2_EMPIRICAL_STATS_BY_BLOCK_SIZE_STOCKS,
                    V2_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS,
                ),
                bonds: HistoricalMonthlyReturns::new(
                    monthly_series.log.bonds,
                    &V2_EMPIRICAL_STATS_BY_BLOCK_SIZE_BONDS,
                    AnnualLogMeanFromOneOverCAPERegressionInfo::new_zero(),
                ),
            },
        }
    };

    let v3 = {
        let monthly_series = process_raw_monthly_non_log_series(&V3_RAW_MONTHLY_NON_LOG_SERIES);
        HistoricalReturnsInfo {
            id: HistoricalReturnsId::V3,
            timestamp_ms: V3_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
            month_range: get_month_range(
                &V3_RAW_MONTHLY_NON_LOG_SERIES_START,
                V3_RAW_MONTHLY_NON_LOG_SERIES.len(),
            ),
            returns: StocksAndBonds {
                stocks: HistoricalMonthlyReturns::new(
                    monthly_series.log.stocks,
                    &V3_EMPIRICAL_STATS_BY_BLOCK_SIZE_STOCKS,
                    V3_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS,
                ),
                bonds: HistoricalMonthlyReturns::new(
                    monthly_series.log.bonds,
                    &V3_EMPIRICAL_STATS_BY_BLOCK_SIZE_BONDS,
                    AnnualLogMeanFromOneOverCAPERegressionInfo::new_zero(),
                ),
            },
        }
    };
    let result = vec![v1, v2, v3];
    assert!(result
        .windows(2)
        .all(|x| (x[0].timestamp_ms < x[1].timestamp_ms)));
    result
}

lazy_static! {
    pub static ref HISTORICAL_MONTHLY_RETURNS: Vec<HistoricalReturnsInfo> =
        get_all_historical_returns_infos();
}

pub fn get_historical_monthly_returns_info(timestamp_ms: i64) -> &'static HistoricalReturnsInfo {
    let x1: &'static Vec<HistoricalReturnsInfo> = &HISTORICAL_MONTHLY_RETURNS;
    let result = &x1
        .iter()
        .rev()
        .find(|x| x.timestamp_ms <= timestamp_ms)
        .unwrap();
    &result
}

#[cfg(test)]
mod tests {
    use rand::{
        distributions::{Distribution, Uniform},
        SeedableRng,
    };
    use rand_chacha::ChaCha20Rng;

    use crate::{
        expected_value_of_returns::EmpiricalAnnualNonLogExpectedValueInfo,
        historical_monthly_returns::data::get_empirical_stats_for_block_size,
    };

    use super::{
        data::{
            v1::V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
            v2::V2_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
            v3::V3_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
        },
        get_historical_monthly_returns_info, HistoricalReturnsId,
    };
    use rstest::*;

    enum StocksOrBonds {
        Stocks,
        Bonds,
    }

    fn resolve_historical_monthly_returns(
        returns_id: &HistoricalReturnsId,
        src: &StocksOrBonds,
    ) -> &'static crate::historical_monthly_returns::HistoricalMonthlyReturns {
        let timestamp_ms = match returns_id {
            HistoricalReturnsId::V1 => V1_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
            HistoricalReturnsId::V2 => V2_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
            HistoricalReturnsId::V3 => V3_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
        };
        let h = &get_historical_monthly_returns_info(timestamp_ms).returns;
        match src {
            StocksOrBonds::Stocks => &h.stocks,
            StocksOrBonds::Bonds => &h.bonds,
        }
    }

    #[rstest]
    fn get_empirical_annual_non_log_from_log_monthly_expected_value(
        #[values(
            HistoricalReturnsId::V1,
            HistoricalReturnsId::V2,
            HistoricalReturnsId::V3
        )]
        returns_id: HistoricalReturnsId,
        #[values(StocksOrBonds::Stocks, StocksOrBonds::Bonds)] src: StocksOrBonds,
        #[values(0.01, 0.03, 0.07, 0.09, 0.1)] annual_mean: f64,
        #[values(None, Some(1), Some(6), Some(12), Some(36), Some(60), Some(120))]
        block_size: Option<usize>,
        #[values(0.5, 0.75, 1.0, 1.25, 1.5)] volatility_scale: f64,
    ) {
        let h = resolve_historical_monthly_returns(&returns_id, &src);
        let log_monthly_mean = annual_mean.ln_1p() / 12.0;
        let info = h.get_empirical_annual_non_log_from_log_monthly_expected_value(
            log_monthly_mean,
            block_size,
            volatility_scale,
        );
        let adjusted_details = h.adjust_log_returns_detailed(&info);
        let mean_diff = adjusted_details.stats.log.mean - log_monthly_mean;
        assert!(mean_diff < 0.00000000001, "mean_diff: {}", mean_diff);
    }

    #[rstest]
    fn historical_expected_return_does_not_adjust(
        #[values(
            HistoricalReturnsId::V1,
            HistoricalReturnsId::V2,
            HistoricalReturnsId::V3
        )]
        returns_id: HistoricalReturnsId,
        #[values(StocksOrBonds::Stocks, StocksOrBonds::Bonds)] src: StocksOrBonds,
        #[values(None, Some(1), Some(6), Some(12), Some(36), Some(60), Some(120))]
        block_size: Option<usize>,
        #[values(0.5, 0.75, 1.0, 1.25, 1.5)] volatility_scale: f64,
    ) {
        let h = resolve_historical_monthly_returns(&returns_id, &src);
        let info = h.get_empirical_annual_non_log_from_log_monthly_expected_value(
            h.log.stats.mean,
            block_size,
            volatility_scale,
        );
        let adjusted_details = h.adjust_log_returns_detailed(&info);
        let mean_diff = adjusted_details.stats.log.mean - h.log.stats.mean;
        assert!(mean_diff < 0.00000000001, "mean_diff: {}", mean_diff);
    }

    #[rstest]
    fn adjust_log_returns_historical(
        #[values(
            HistoricalReturnsId::V1,
            HistoricalReturnsId::V2,
            HistoricalReturnsId::V3
        )]
        returns_id: HistoricalReturnsId,
        #[values(StocksOrBonds::Stocks, StocksOrBonds::Bonds)] src: StocksOrBonds,
        #[values(0.1, 0.2, 0.5, 0.7, 0.9, 0.1)] target_mean: f64,
        #[values(0.5, 0.75, 1.0, 1.25, 1.5)] volatility_scale: f64,
    ) {
        let h = resolve_historical_monthly_returns(&returns_id, &src);
        let adjusted_info = h.adjust_log_returns_detailed(
            &(EmpiricalAnnualNonLogExpectedValueInfo {
                value: target_mean,
                block_size: None,
                log_volatility_scale: volatility_scale,
            }),
        );

        let mean_diff = adjusted_info.stats.annualized.non_log.mean - target_mean;
        let var_diff = adjusted_info.stats.annualized.log.variance
            - adjusted_info.stats.empirical_annual_log_variance;
        assert!(mean_diff < 0.00099, "mean_diff: {}", mean_diff);
        assert!(var_diff < 0.00099, "var_diff: {}", var_diff);
    }

    #[rstest]
    fn adjust_log_returns_monte_carlo(
        #[values(
            HistoricalReturnsId::V1,
            HistoricalReturnsId::V2,
            HistoricalReturnsId::V3
        )]
        returns_id: HistoricalReturnsId,
        #[values(StocksOrBonds::Stocks, StocksOrBonds::Bonds)] src: StocksOrBonds,
        #[values(0.5, 0.1)] target_mean: f64,
        #[values(0.5, 1.0, 1.5)] volatility_scale: f64,
        #[values(1, 6, 12, 36, 60, 120)] block_size: usize,
        #[values(0, 1, 2)] seed_index: usize,
    ) {
        let seed = Uniform::from(0..u64::MAX)
            .sample_iter(ChaCha20Rng::seed_from_u64(19902393848483))
            .take(seed_index + 1)
            .collect::<Vec<u64>>()[seed_index];

        let h = resolve_historical_monthly_returns(&returns_id, &src);
        let adjusted_info = h.adjust_log_returns_detailed(
            &(EmpiricalAnnualNonLogExpectedValueInfo {
                value: target_mean,
                block_size: Some(block_size as usize),
                log_volatility_scale: volatility_scale,
            }),
        );

        let sampled_adjusted = get_empirical_stats_for_block_size(
            &adjusted_info.log_series,
            block_size,
            10 * 1000,
            55 * 12,
            true,
            seed,
        );
        let mean_diff = sampled_adjusted.annual_non_log_returns_mean as f64 - target_mean;
        let var_diff = sampled_adjusted.annual_log_returns_variance as f64
            - adjusted_info.stats.empirical_annual_log_variance;
        assert!(mean_diff < 0.001, "mean_diff: {}", mean_diff);
        assert!(var_diff < 0.0005, "var_diff: {}", var_diff);
    }
}
