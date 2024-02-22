use serde::{Deserialize, Serialize};
use tsify::Tsify;

#[derive(Serialize, Deserialize, Tsify, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CalendarMonth {
    pub year: i64,
    pub month: i64,
}

impl CalendarMonth {
    pub fn minus_in_months(&self, other: &CalendarMonth) -> i64 {
        (self.year - other.year) * 12 + self.month - other.month
    }

    pub fn add_years(&self, years: i64) -> CalendarMonth {
        CalendarMonth {
            year: self.year + years,
            month: self.month,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(2020, 5, 2019, 4, 13)]
    #[case(2020, 5, 2019, 5, 12)]
    #[case(2020, 5, 2019, 6, 11)]
    #[case(2020, 5, 2020, 4, 1)]
    #[case(2020, 5, 2020, 5, 0)]
    #[case(2020, 5, 2020, 6, -1)]
    #[case(2020, 5, 2021, 4, -11)]
    #[case(2020, 5, 2021, 5, -12)]
    #[case(2020, 5, 2021, 6, -13)]
    fn test_minus_in_months(
        #[case] start_year: i64,
        #[case] start_month: i64,
        #[case] end_year: i64,
        #[case] end_month: i64,
        #[case] expected: i64,
    ) {
        let start = CalendarMonth {
            year: start_year,
            month: start_month,
        };

        let end = CalendarMonth {
            year: end_year,
            month: end_month,
        };
        assert_eq!(start.minus_in_months(&end), expected);
    }
}
