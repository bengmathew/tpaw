use std::fs::{self, File};

use crate::historical_monthly_returns::{data::RawCAPESeriesEntry, HistoricalReturnsId};

#[derive(Debug)]
struct Row {
    year: u16,
    month: u8,
    cape: Option<f64>,
    stocks: f64,
    bonds: f64,
}

pub fn cli_process_historical_data_part_1_parse_csv(returns_id: HistoricalReturnsId) {
    let base_dir = format!(
        "./src/lib/historical_monthly_returns/data/{0}",
        returns_id.as_str()
    );
    let data = read_csv(&returns_id, &base_dir);
    write_non_log_series(&data, &returns_id, &base_dir);
    write_cape_data(&data, &returns_id, &base_dir);
}

fn read_csv(returns_id: &HistoricalReturnsId, base_dir: &str) -> Vec<Row> {
    let file = File::open(format!("{base_dir}/{}_raw_data.csv", returns_id.as_str())).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut data: Vec<Row> = Vec::new();

    for record in rdr.records() {
        let record = record.unwrap();
        data.push(Row {
            year: record[0].parse().unwrap(),
            month: record[1].parse().unwrap(),
            cape: if record[2] == *"NA" {
                None
            } else {
                Some(record[2].parse().unwrap())
            },
            stocks: record[3].parse().unwrap(),
            bonds: record[4].parse().unwrap(),
        });
    }

    data.iter().enumerate().for_each(|(i, curr)| {
        if i > 0 {
            let prev = &data[i - 1];
            assert!(curr.year >= prev.year);
            assert!(if curr.month == 1 {
                prev.month == 12
            } else {
                curr.month == prev.month + 1
            });
        }
    });

    data
}

fn write_non_log_series(data: &[Row], returns_id: &HistoricalReturnsId, base_dir: &str) {
    let returns_id_caps = returns_id.as_str().to_uppercase();

    let array_contents = data
        .iter()
        .map(|x| {
            format!(
                "    RawMonthlyNonLogSeriesEntry::new({year}, {month}, {stocks}, {bonds}),",
                year = x.year,
                month = x.month,
                stocks = x.stocks,
                bonds = x.bonds,
            )
        })
        .collect::<Vec<String>>()
        .join("\n");

    let contents = format!(
        r#"use crate::{{
    historical_monthly_returns::data::RawMonthlyNonLogSeriesEntry,
    shared_types::YearAndMonth,
}};

pub const {returns_id_caps}_RAW_MONTHLY_NON_LOG_SERIES_START: YearAndMonth = YearAndMonth {{
    year: {year},
    month: {month},
}};

pub const {returns_id_caps}_RAW_MONTHLY_NON_LOG_SERIES: [RawMonthlyNonLogSeriesEntry; {num_months}] = [
{array_contents}
];
"#,
        year = data[0].year,
        month = data[0].month,
        num_months = data.len(),
    );

    fs::write(
        format!(
            "{base_dir}/{returns_id}_raw_monthly_non_log_series.rs",
            returns_id = returns_id.as_str(),
        ),
        contents,
    )
    .unwrap();
}

fn write_cape_data(data: &[Row], returns_id: &HistoricalReturnsId, base_dir: &str) {
    let cape_data = data
        .iter()
        .map(|x| RawCAPESeriesEntry {
            year: x.year as u16,
            month: x.month as u8,
            cape: x.cape,
        })
        .collect::<Vec<_>>();
    let cape_data_str = serde_json::to_string(&cape_data).unwrap();

    fs::write(
        format!(
            "{base_dir}/{returns_id}_raw_cape_series.json",
            returns_id = returns_id.as_str(),
        ),
        cape_data_str,
    )
    .unwrap();
}
