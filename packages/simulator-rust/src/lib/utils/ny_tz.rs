use std::ops::RangeInclusive;

use chrono::DateTime;
use chrono::Datelike;
use chrono::NaiveDate;
use chrono::NaiveDateTime;
use chrono::TimeZone;
use chrono::Utc;
use chrono_tz;

pub fn get_now_in_ny_tz() -> DateTime<chrono_tz::Tz> {
    chrono_tz::America::New_York.from_utc_datetime(&Utc::now().naive_utc())
}

// This does not change the timestamp.
pub fn interpret_timestamp_as_ny_tz(timestamp_ms: i64) -> DateTime<chrono_tz::Tz> {
    chrono_tz::America::New_York.from_utc_datetime(
        &DateTime::from_timestamp_millis(timestamp_ms)
            .unwrap()
            .naive_utc(),
    )
}

// NaiveDateTime does not have a timestamp associated with it, it needs a
// timezone to resolve to a timestamp. This interprets the NaiveDateTime as
// being in America/New_York time. So 12/24/2024 4:00 PM is interpreted as
// 12/24/2024 4:00 PM in NY time. This will be a different timestamp than if it
// was interpreted in UTC.
pub fn interpret_naive_date_time_as_ny_tz(
    naive_date_time: &NaiveDateTime,
) -> DateTime<chrono_tz::Tz> {
    chrono_tz::America::New_York
        .from_local_datetime(naive_date_time)
        .unwrap()
}

// Assumes there is a market close on that day.
pub fn get_market_closing_time_in_ny_tz_from_naive_date(
    naive_date: &NaiveDate,
) -> DateTime<chrono_tz::Tz> {
    interpret_naive_date_time_as_ny_tz(&naive_date.and_hms_opt(16, 0, 0).unwrap())
}

pub fn get_next_weekday(date_time: DateTime<chrono_tz::Tz>) -> DateTime<chrono_tz::Tz> {
    let delta = if date_time.weekday() == chrono::Weekday::Fri {
        3
    } else if date_time.weekday() == chrono::Weekday::Sat {
        2
    } else {
        1
    };
    date_time
        .checked_add_days(chrono::Days::new(delta))
        .unwrap()
}

// start of range should be a market close.
pub fn get_conservative_market_closes_between_timestamps_ms(
    range: RangeInclusive<i64>,
) -> Vec<DateTime<chrono_tz::Tz>> {
    let mut result = vec![];
    let mut current = interpret_timestamp_as_ny_tz(*range.start());
    while *range.end() >= current.timestamp_millis() {
        result.push(current);
        current = get_next_weekday(current);
    }
    result
}

#[cfg(test)]
mod tests {
    use chrono::{Datelike, NaiveDate};
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn test_get_timestamp_in_ny_tz() {
        let timestamp_ms = 1714406400000;
        let dt = interpret_timestamp_as_ny_tz(timestamp_ms);
        assert_eq!(dt.timestamp_millis(), timestamp_ms);
    }

    #[rstest]
    fn test_get_conservative_market_closes_between_timestamps_ms() {
        fn _from_ymd(year: i32, month: u32, day: u32) -> DateTime<chrono_tz::Tz> {
            get_market_closing_time_in_ny_tz_from_naive_date(
                &NaiveDate::from_ymd_opt(year, month, day).unwrap(),
            )
        }
        let start = _from_ymd(2024, 10, 31); 
        assert_eq!(start.weekday(), chrono::Weekday::Thu);
        let market_closes: Vec<DateTime<chrono_tz::Tz>> =
            get_conservative_market_closes_between_timestamps_ms(
                start.timestamp_millis()..=start.timestamp_millis() + 1000 * 60 * 60 * 24 * 5,
            );
        assert_eq!(market_closes.len(), 3);
        assert_eq!(market_closes[0].weekday(), chrono::Weekday::Thu);
        assert_eq!(market_closes[0], _from_ymd(2024, 10, 31));
        assert_eq!(market_closes[1].weekday(), chrono::Weekday::Fri);
        assert_eq!(market_closes[1], _from_ymd(2024, 11, 1)); 
        assert_eq!(market_closes[2].weekday(), chrono::Weekday::Mon);
        assert_eq!(market_closes[2], _from_ymd(2024, 11, 4)); 
    }
}
