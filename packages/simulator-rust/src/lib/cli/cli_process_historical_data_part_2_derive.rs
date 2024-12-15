use linreg::linear_regression;
use rayon::prelude::*;
use crate::{
    constants::MAX_AGE_IN_MONTHS, historical_monthly_returns::{
        data::{
            get_empirical_stats_for_block_size, process_raw_monthly_non_log_series,
            v1::v1_raw_monthly_non_log_series::{
                V1_RAW_MONTHLY_NON_LOG_SERIES, V1_RAW_MONTHLY_NON_LOG_SERIES_START,
            },
            v2::v2_raw_monthly_non_log_series::{
                V2_RAW_MONTHLY_NON_LOG_SERIES, V2_RAW_MONTHLY_NON_LOG_SERIES_START,
            },
            v3::v3_raw_monthly_non_log_series::{V3_RAW_MONTHLY_NON_LOG_SERIES, V3_RAW_MONTHLY_NON_LOG_SERIES_START},
            v4::v4_raw_monthly_non_log_series::{V4_RAW_MONTHLY_NON_LOG_SERIES, V4_RAW_MONTHLY_NON_LOG_SERIES_START},
            AnnualLogMeanFromOneOverCAPERegressionInfo, EmpiricalStats32,
            FiveTenTwentyThirtyYearsSlopeAndIntercept, RawCAPESeriesEntry,
            RawMonthlyNonLogSeriesEntry,
        },
        HistoricalReturnsId,
    }, utils::{return_series::periodize_log_returns, shared_types::SlopeAndIntercept}
};
use std::{
    fs::{self, File},
    time::Instant,
};

const NUM_RUNS: usize = 500 * 1000;
const MONTHS_PER_RUN: usize = 50 * 12;
const STAGGER_RUN_STARTS: bool = true;
const MAX_BLOCK_SIZE: usize = MAX_AGE_IN_MONTHS;
const SEED: u64 = 15998927836015553044;

pub fn cli_process_historical_data_part_2_derive(returns_id: HistoricalReturnsId) {
    let base_dir = format!(
        "./src/lib/historical_monthly_returns/data/{}",
        returns_id.as_str()
    );
    generate_regressions(&returns_id, &base_dir);
    generate_empirical_stats(&returns_id, &base_dir);
}

fn get_raw_monthly_non_log_series(
    returns_id: &HistoricalReturnsId,
) -> &[RawMonthlyNonLogSeriesEntry] {
    match returns_id {
        HistoricalReturnsId::V1 => &V1_RAW_MONTHLY_NON_LOG_SERIES,
        HistoricalReturnsId::V2 => &V2_RAW_MONTHLY_NON_LOG_SERIES,
        HistoricalReturnsId::V3 => &V3_RAW_MONTHLY_NON_LOG_SERIES,
        HistoricalReturnsId::V4 => &V4_RAW_MONTHLY_NON_LOG_SERIES,
    }
}

fn generate_regressions(returns_id: &HistoricalReturnsId, base_dir: &str) {
    let cape_series: Vec<RawCAPESeriesEntry> = {
        let raw_cape_file = File::open(format!(
            "{base_dir}/{}_raw_cape_series.json",
            returns_id.as_str()
        ))
        .unwrap();
        serde_json::from_reader(raw_cape_file).unwrap()
    };

    let monthly_series =
        process_raw_monthly_non_log_series(get_raw_monthly_non_log_series(returns_id));
    let raw_monthly_series_start = match returns_id {
        HistoricalReturnsId::V1 => &V1_RAW_MONTHLY_NON_LOG_SERIES_START,
        HistoricalReturnsId::V2 => &V2_RAW_MONTHLY_NON_LOG_SERIES_START,
        HistoricalReturnsId::V3 => &V3_RAW_MONTHLY_NON_LOG_SERIES_START,
        HistoricalReturnsId::V4 => &V4_RAW_MONTHLY_NON_LOG_SERIES_START,
    };

    assert!(
        cape_series.len() == monthly_series.log.stocks.len(),
        "{} {}",
        cape_series.len(),
        monthly_series.log.stocks.len()
    );
    assert!(cape_series[0].year == raw_monthly_series_start.year);
    assert!(cape_series[0].month == raw_monthly_series_start.month);

    let full_start_index = cape_series.iter().position(|x| x.cape.is_some()).unwrap();

    let restricted_start_index = cape_series
        .iter()
        .position(|x| x.year == 1950 && x.month == 1)
        .unwrap();

    let regress = |start_index: usize, window_years: usize| {
        do_regression(
            &cape_series,
            &monthly_series.log.stocks,
            start_index,
            window_years,
        )
    };

    let stocks = AnnualLogMeanFromOneOverCAPERegressionInfo {
        full: FiveTenTwentyThirtyYearsSlopeAndIntercept {
            five_year: regress(full_start_index, 5),
            ten_year: regress(full_start_index, 10),
            twenty_year: regress(full_start_index, 20),
            thirty_year: regress(full_start_index, 30),
        },
        restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {
            five_year: regress(restricted_start_index, 5),
            ten_year: regress(restricted_start_index, 10),
            twenty_year: regress(restricted_start_index, 20),
            thirty_year: regress(restricted_start_index, 30),
        },
    };

    write_regession(returns_id, base_dir, "stocks", &stocks);
}

fn do_regression(
    raw_cape_series_full: &[RawCAPESeriesEntry],
    monthly_log_return_series_full: &[f64],
    start_index: usize,
    window_years: usize,
) -> SlopeAndIntercept {
    let raw_cape_series = &raw_cape_series_full[start_index..];
    let monthly_log_return_series = &monthly_log_return_series_full[start_index..];
    let log_one_over_cape_series = raw_cape_series
        .iter()
        .map(|x| (1.0 / x.cape.unwrap()).ln_1p())
        .collect::<Vec<_>>();

    let periodized_log_return_series =
        periodize_log_returns(&monthly_log_return_series, window_years * 12)
            .iter()
            .map(|x| x / window_years as f64)
            .collect::<Vec<_>>();
    let log_one_over_cape_series_cut =
        &log_one_over_cape_series[..periodized_log_return_series.len()];

    let (slope, intercept) = linear_regression::<f64, f64, f64>(
        log_one_over_cape_series_cut,
        &periodized_log_return_series,
    )
    .unwrap();
    SlopeAndIntercept { slope, intercept }
}

fn write_regession(
    returns_id: &HistoricalReturnsId,
    base_dir: &str,
    kind: &str,
    result: &AnnualLogMeanFromOneOverCAPERegressionInfo,
) {
    let contents = format!(
        r#"use crate::{{
    historical_monthly_returns::data::{{
        AnnualLogMeanFromOneOverCAPERegressionInfo, FiveTenTwentyThirtyYearsSlopeAndIntercept,
    }},
    shared_types::SlopeAndIntercept,
}};

pub const {returns_id_caps}_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_{kind_caps}:
    AnnualLogMeanFromOneOverCAPERegressionInfo = AnnualLogMeanFromOneOverCAPERegressionInfo {{
    full: FiveTenTwentyThirtyYearsSlopeAndIntercept {{
        five_year: SlopeAndIntercept {{
            slope: {full_five_year_slope},
            intercept: {full_five_year_intercept},
        }},
        ten_year: SlopeAndIntercept {{
            slope: {full_ten_year_slope},
            intercept: {full_ten_year_intercept},
        }},
        twenty_year: SlopeAndIntercept {{
            slope: {full_twenty_year_slope},
            intercept: {full_twenty_year_intercept},
        }},
        thirty_year: SlopeAndIntercept {{
            slope: {full_thirty_year_slope},
            intercept: {full_thirty_year_intercept},
        }},
    }},
    restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {{
        five_year: SlopeAndIntercept {{
            slope: {restricted_five_year_slope},
            intercept: {restricted_five_year_intercept},
        }},
        ten_year: SlopeAndIntercept {{
            slope: {restricted_ten_year_slope},
            intercept: {restricted_ten_year_intercept},
        }},
        twenty_year: SlopeAndIntercept {{
            slope: {restricted_twenty_year_slope},
            intercept: {restricted_twenty_year_intercept},
        }},
        thirty_year: SlopeAndIntercept {{
            slope: {restricted_thirty_year_slope},
            intercept: {restricted_thirty_year_intercept},
        }},
    }},
}};        
        "#,
        returns_id_caps = returns_id.as_str().to_uppercase(),
        kind_caps = kind.to_uppercase(),
        full_five_year_slope = result.full.five_year.slope,
        full_five_year_intercept = result.full.five_year.intercept,
        full_ten_year_slope = result.full.ten_year.slope,
        full_ten_year_intercept = result.full.ten_year.intercept,
        full_twenty_year_slope = result.full.twenty_year.slope,
        full_twenty_year_intercept = result.full.twenty_year.intercept,
        full_thirty_year_slope = result.full.thirty_year.slope,
        full_thirty_year_intercept = result.full.thirty_year.intercept,
        restricted_five_year_slope = result.restricted.five_year.slope,
        restricted_five_year_intercept = result.restricted.five_year.intercept,
        restricted_ten_year_slope = result.restricted.ten_year.slope,
        restricted_ten_year_intercept = result.restricted.ten_year.intercept,
        restricted_twenty_year_slope = result.restricted.twenty_year.slope,
        restricted_twenty_year_intercept = result.restricted.twenty_year.intercept,
        restricted_thirty_year_slope = result.restricted.thirty_year.slope,
        restricted_thirty_year_intercept = result.restricted.thirty_year.intercept,
    );
    fs::write(
        format!(
            "{base_dir}/{returns_id}_annual_log_mean_from_one_over_cape_regression_info_{kind}.rs",
            returns_id = returns_id.as_str()
        ),
        contents,
    )
    .unwrap();
}

fn generate_empirical_stats(returns_id: &HistoricalReturnsId, base_dir: &str) {
    let raw_monthly_non_log_series = get_raw_monthly_non_log_series(returns_id);
    let monthly_series = process_raw_monthly_non_log_series(raw_monthly_non_log_series);

    write_empirical_stats_file(
        &monthly_series.log.stocks,
        &base_dir,
        &returns_id,
        "stocks".to_string(),
    );
    write_empirical_stats_file(
        &monthly_series.log.bonds,
        &base_dir,
        &returns_id,
        "bonds".to_string(),
    );
}

fn write_empirical_stats_file(
    monthly_log_returns: &[f64],
    base_dir: &str,
    returns_id: &HistoricalReturnsId,
    kind: String,
) {
    let start = Instant::now();
    let blocks_sizes = 1..=MAX_BLOCK_SIZE;

    let result: Vec<EmpiricalStats32> = blocks_sizes
        .into_par_iter()
        .map(|block_size| {
            get_empirical_stats_for_block_size(
                monthly_log_returns,
                block_size,
                NUM_RUNS,
                MONTHS_PER_RUN,
                STAGGER_RUN_STARTS,
                SEED,
            )
        })
        .collect();

    {
        let array_contents = result
            .iter()
            .map(|x| {
                format!(
                    "    EmpiricalStats32::new({}, {}),",
                    x.annual_non_log_returns_mean, x.annual_log_returns_variance
                )
            })
            .collect::<Vec<String>>()
            .join("\n");

        let contents = format!(
            r#"use crate::historical_monthly_returns::data::EmpiricalStats32;

const _NUM_RUNS: usize = {NUM_RUNS};
const _MONTHS_PER_RUN: usize = {MONTHS_PER_RUN};
const _STAGGER_RUN_STARTS: bool = {STAGGER_RUN_STARTS};
const _SEED: u64 = {SEED};
pub const {returns_id_caps}_EMPIRICAL_STATS_BY_BLOCK_SIZE_{kind_caps}: [EmpiricalStats32; {num_items}] = [
    EmpiricalStats32::new(0.0, 0.0), // Block size 0 is undefined.
{array_contents}
];
"#,
            returns_id_caps = returns_id.as_str().to_uppercase(),
            kind_caps = kind.to_uppercase(),
            num_items = MAX_BLOCK_SIZE + 1
        );
        fs::write(
            format!(
                "{base_dir}/{returns_id}_empirical_stats_by_block_size_{kind}.rs",
                returns_id = returns_id.as_str()
            ),
            contents,
        )
        .unwrap();
    }

    let duration = start.elapsed();
    println!("Time: Empirical stats {kind}: {:?}", duration);
}
