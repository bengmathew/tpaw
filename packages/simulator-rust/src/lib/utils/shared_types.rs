use std::ops::RangeInclusive;
use serde::{Deserialize, Serialize};

use crate::wire::{WireMonthRange, WireYearAndMonth};

#[derive(Serialize, Deserialize, Copy, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LogAndNonLog<T> {
    pub log: T,
    pub non_log: T,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct StocksAndBonds<T> {
    pub stocks: T,
    pub bonds: T,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct MonthAndStocks {
    pub month: i64,
    pub stocks: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Stocks {
    pub stocks: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Monthly<T> {
    pub monthly: T,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Log<T> {
    pub log: T,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseAndLog<T> {
    pub base: T,
    pub log: T,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SlopeAndIntercept {
    pub slope: f64,
    pub intercept: f64,
}

impl SlopeAndIntercept {
    pub fn y(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

pub enum StocksOrBonds {
    Stocks,
    Bonds,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct YearAndMonth {
    // TODO: After duration matching, make this i64.
    pub year: u16,
    pub month: u8,
}

impl From<&YearAndMonth> for WireYearAndMonth {
    fn from(value: &YearAndMonth) -> Self {
        Self {
            year: value.year as i64,
            month: value.month as i64,
        }
    }
}

impl YearAndMonth {
    pub fn add_months(&self, months_to_add: i64) -> YearAndMonth {
        let as_months_since0 = self.year as i64 * 12 + self.month as i64 - 1;
        let new_as_months_since0 = as_months_since0 + months_to_add;
        assert!(new_as_months_since0 >= 0); // Mod math doesn't work with negative numbers.
        let year = new_as_months_since0 / 12;
        let month = (new_as_months_since0 % 12) + 1;
        YearAndMonth {
            year: year as u16,
            month: month as u8,
        }
    }
}


impl From<RangeInclusive<YearAndMonth>> for WireMonthRange {
    fn from(value: RangeInclusive<YearAndMonth>) -> Self {
        Self {
            start: value.start().into(),
            end: value.end().into(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(2020, 1, 11, 2020, 12)]
    #[case(2020, 1, 12, 2021, 1)]
    #[case(2020, 12, 1, 2021, 1)]
    #[case(2020, 12, 13, 2022, 1)]
    #[case (2020, 2, -1, 2020, 1)]
    #[case (2020, 2, -12, 2019, 2)]
    #[case (2020, 2, -13, 2019, 1)]
    #[case (2020, 2, -14, 2018, 12)]
    fn test_add_months(
        #[case] year: u16,
        #[case] month: u8,
        #[case] months_to_add: i64,
        #[case] result_year: u16,
        #[case] result_month: u8,
    ) {
        let start = YearAndMonth {
            year: year,
            month: month,
        };

        let result = YearAndMonth {
            year: result_year,
            month: result_month,
        };
        assert_eq!(start.add_months(months_to_add), result);
    }
}
