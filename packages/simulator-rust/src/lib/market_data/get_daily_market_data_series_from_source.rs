use chrono::Datelike;
use futures::future::join_all;
use itertools::Itertools;

use crate::config::CONFIG;
use crate::market_data::market_data_defs::DailyMarketDataForPresets_Inflation;
use crate::ny_tz::get_market_closing_time_in_ny_tz_from_naive_date;
use crate::{constants::MIN_PLAN_PARAM_TIME_MS, utils::round::RoundP};
use chrono::DateTime;
use chrono::Duration;
use chrono::NaiveDate;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::market_data::market_data_defs::DailyMarketDataForPresets;
use crate::market_data::market_data_defs::DailyMarketDataForPresets_BondRates;
use crate::market_data::market_data_defs::DailyMarketDataForPresets_SP500;
use crate::market_data::market_data_defs::VTAndBNDData;

use crate::utils::ny_tz::get_now_in_ny_tz;

use super::market_data_defs::VTAndBNDData_PercentageChangeFromLastClose;

const LOOKBACK_IN_DAYS: i32 = 30;
fn get_data_start_time_in_utc() -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp_millis(MIN_PLAN_PARAM_TIME_MS).unwrap()
        - Duration::days(LOOKBACK_IN_DAYS.into())
}

pub async fn get_daily_market_data_series_from_source(
) -> (Vec<DailyMarketDataForPresets>, Vec<VTAndBNDData>) {
    let (inflation, sp500, bond_rates, vt_and_bnd) = futures::join!(
        get_inflation(),
        get_sp500(),
        get_bond_rates(),
        get_vt_and_bnd()
    );
    (combine_streams(inflation, sp500, bond_rates), vt_and_bnd)
}

async fn get_vt_and_bnd() -> Vec<VTAndBNDData> {
    let (bnd, vt) = futures::join!(get_from_eod("BND.US"), get_from_eod("VT.US"));

    bnd.iter()
        // Zip will stop at the shortest length, which is what we want.
        .zip(vt)
        .map(|(bnd, vt)| VTAndBNDData {
            closing_time_ms: {
                assert!(bnd.closing_time_ms == vt.closing_time_ms);
                bnd.closing_time_ms
            },
            percentage_change_from_last_close: VTAndBNDData_PercentageChangeFromLastClose {
                // Returns are calculated from adjusted close, not close, because
                // the adjusted close includes dividends and stock splits.
                vt: vt.percentage_change_from_last_adjusted_close,
                bnd: bnd.percentage_change_from_last_adjusted_close,
            },
        })
        .collect()
}

async fn get_bond_rates() -> Vec<DailyMarketDataForPresets_BondRates> {
    async fn for_year(year: i32) -> Vec<DailyMarketDataForPresets_BondRates> {
        let text = reqwest::Client::new()
            .get(format!(
                "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all",
                year = year
            ))
            .query(&[
                ("type", "daily_treasury_real_yield_curve"),
            ])
            .header("Cache-Control", "no-cache")
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();

        let rows: Vec<Vec<&str>> = text
            .split("\n")
            .map(|x| x.split(",").collect::<Vec<&str>>())
            .collect();

        assert!(
            rows[0]
                == [
                    "Date",
                    "\"5 YR\"",
                    "\"7 YR\"",
                    "\"10 YR\"",
                    "\"20 YR\"",
                    "\"30 YR\"",
                ]
        );

        rows.iter()
            .skip(1)
            .map(|cols| DailyMarketDataForPresets_BondRates {
                closing_time_ms: date_to_market_closing_time_ms(&cols[0], "%m/%d/%Y"),
                five_year: parse_percent_string(cols[1]).unwrap(),
                seven_year: parse_percent_string(cols[2]).unwrap(),
                ten_year: parse_percent_string(cols[3]).unwrap(),
                twenty_year: parse_percent_string(cols[4]).unwrap(),
                thirty_year: parse_percent_string(cols[5]).unwrap(),
            })
            .collect()
    }

    let now_ny = get_now_in_ny_tz();
    join_all((get_data_start_time_in_utc().year()..=now_ny.year()).map(|year| for_year(year)))
        .await
        .into_iter()
        .flatten()
        .sorted_by(|a, b| a.closing_time_ms.cmp(&b.closing_time_ms))
        .collect()
}

async fn get_sp500() -> Vec<DailyMarketDataForPresets_SP500> {
    get_from_eod("GSPC.INDX")
        .await
        .iter()
        .map(|x| DailyMarketDataForPresets_SP500 {
            closing_time_ms: x.closing_time_ms,
            // This data is used for CAPE calcualtions, which is the price to earnings ratio.
            // The price of the SP500 is given by close and not adjusted_close.
            value: x.close,
        })
        .collect()
}

async fn get_inflation() -> Vec<DailyMarketDataForPresets_Inflation> {
    let body = reqwest::Client::new()
        .get("https://api.stlouisfed.org/fred/series/observations")
        .query(&[
            ("api_key", CONFIG.fred_api_key.as_str()),
            ("series_id", "T10YIE"),
            ("file_type", "json"),
            (
                "observation_start",
                &get_data_start_time_in_utc().format("%Y-%m-%d").to_string(),
            ),
        ])
        .header("Cache-Control", "no-cache")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    #[derive(Serialize, Deserialize, Debug)]
    struct Data {
        date: String,
        value: String,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct Response {
        observations: Vec<Data>,
    }
    serde_json::from_str::<Response>(&body)
        .unwrap()
        .observations
        .iter()
        .map(|x| {
            // Don't unwrap() because value for memorial day 2023-05-29 came
            // back as "." which fails parsing.
            let value = parse_percent_string(&x.value)?;
            Some(DailyMarketDataForPresets_Inflation {
                closing_time_ms: date_to_market_closing_time_ms(&x.date, "%Y-%m-%d"),
                value,
            })
        })
        .filter_map(|x| x)
        .collect::<Vec<DailyMarketDataForPresets_Inflation>>()
}

#[derive(Debug)]
struct EODData {
    closing_time_ms: i64,
    percentage_change_from_last_adjusted_close: f64,
    close: f64,
}

async fn get_from_eod(name: &str) -> Vec<EODData> {
    let current_time_ms = chrono::Utc::now().timestamp_millis();
    let body = reqwest::Client::new()
        .get(format!("https://eodhistoricaldata.com/api/eod/{}", name))
        .query(&[
            ("api_token", CONFIG.eod_api_key.as_str()),
            ("fmt", "json"),
            ("period", "d"), // daily
            ("order", "a"),  // ascending
            (
                "from",
                &get_data_start_time_in_utc().format("%Y-%m-%d").to_string(),
            ),
        ])
        .header("Cache-Control", "no-cache")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    #[derive(Serialize, Deserialize, Debug)]
    struct Data {
        date: String,
        close: f64,
        adjusted_close: f64,
    }

    serde_json::from_str::<Vec<Data>>(&body)
        .unwrap()
        .windows(2)
        .map(|x| {
            let (prev, curr) = (&x[0], &x[1]);
            EODData {
                closing_time_ms: date_to_market_closing_time_ms(&curr.date, "%Y-%m-%d"),
                percentage_change_from_last_adjusted_close: (curr.adjusted_close
                    - prev.adjusted_close)
                    / prev.adjusted_close,
                close: curr.close,
            }
        })
        // For SP500 it can return intra-day data with, which looks like a closing
        // time in the future. Filter these out.
        .filter(|x| x.closing_time_ms < current_time_ms)
        .collect()
}

// We want to convert a string like "1.45" to 0.0145. Naively this would be
// converting to float and then dividing by 100, but that introduces floating
// point error in the division. This function handles the conversion without
// introducing floating point error.
fn parse_percent_string(s: &str) -> Option<f64> {
    let value = s.parse::<f64>().ok()?;
    let parts: Vec<&str> = s.split('.').collect();
    assert!(parts.len() == 1 || parts.len() == 2);
    let precision: i32 = if parts.len() == 1 {
        0
    } else {
        parts[1].len() as i32
    };
    Some((value / 100.0).round_p(precision + 2))
}

fn date_to_market_closing_time_ms(date: &str, format: &str) -> i64 {
    get_market_closing_time_in_ny_tz_from_naive_date(
        &NaiveDate::parse_from_str(date, format).unwrap(),
    )
    .timestamp_millis()
}

fn combine_streams(
    inflation: Vec<DailyMarketDataForPresets_Inflation>,
    sp500: Vec<DailyMarketDataForPresets_SP500>,
    bond_rates: Vec<DailyMarketDataForPresets_BondRates>,
) -> Vec<DailyMarketDataForPresets> {
    let inflation_closing_times_ms = inflation.iter().map(|x| x.closing_time_ms).collect_vec();
    let sp500_closing_times_ms = sp500.iter().map(|x| x.closing_time_ms).collect_vec();
    let bond_rates_closing_times_ms = bond_rates.iter().map(|x| x.closing_time_ms).collect_vec();

    let mut inflation_ptr = 0;
    let mut sp500_ptr = 0;
    let mut bond_rates_ptr = 0;

    let combined_start_closing_time_ms = vec![
        inflation_closing_times_ms[0],
        sp500_closing_times_ms[0],
        bond_rates_closing_times_ms[0],
    ]
    .into_iter()
    .max()
    .unwrap();

    let combined_closing_times_ms = inflation_closing_times_ms
        .iter()
        .chain(sp500_closing_times_ms.iter())
        .chain(bond_rates_closing_times_ms.iter())
        .map(|x| *x)
        .unique()
        .filter(|x| *x >= combined_start_closing_time_ms)
        .sorted()
        .collect_vec();

    combined_closing_times_ms
        .iter()
        .map(|timestamp| {
            fn advance_ptr(closing_times_ms: &Vec<i64>, curr: usize, timestamp: i64) -> usize {
                if curr == closing_times_ms.len() - 1 || closing_times_ms[curr + 1] > timestamp {
                    curr
                } else {
                    advance_ptr(closing_times_ms, curr + 1, timestamp)
                }
            }
            inflation_ptr = advance_ptr(&inflation_closing_times_ms, inflation_ptr, *timestamp);
            sp500_ptr = advance_ptr(&sp500_closing_times_ms, sp500_ptr, *timestamp);
            bond_rates_ptr = advance_ptr(&bond_rates_closing_times_ms, bond_rates_ptr, *timestamp);
            DailyMarketDataForPresets {
                closing_time_ms: *timestamp,
                inflation: inflation[inflation_ptr],
                sp500: sp500[sp500_ptr],
                bond_rates: bond_rates[bond_rates_ptr],
            }
        })
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn test_parse_percent_string() {
        assert_eq!(parse_percent_string("1.45"), Some(0.0145));
        assert_eq!(parse_percent_string("1.4567"), Some(0.014567));
        assert_eq!(parse_percent_string("0.14567"), Some(0.0014567));
        assert_eq!(parse_percent_string("0.014567"), Some(0.00014567));
    }
}
