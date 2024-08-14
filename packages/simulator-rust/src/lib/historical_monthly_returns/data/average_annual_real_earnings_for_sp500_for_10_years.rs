use crate::constants::MIN_PLAN_PARAM_TIME_MS;
use crate::shared_types::YearAndMonth;
use serde::{Deserialize, Serialize};
use tsify::Tsify;

#[derive(Serialize, Deserialize, Clone, Copy, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct TenYearDuration {
    pub start: YearAndMonth,
    pub end: YearAndMonth,
}

#[derive(Serialize, Deserialize, Clone, Copy, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct AverageAnnualRealEarningsForSP500For10Years {
    pub added_date_ms: i64,
    pub ten_year_duration: TenYearDuration,
    pub value: f64,
}

// FEATURE: Version this the next time it is updated.
// NOTE: added_date_ms is the date it was *added* to the array, not when it was true in the
// world.
pub const AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS:
    [AverageAnnualRealEarningsForSP500For10Years; 5] = [
    AverageAnnualRealEarningsForSP500For10Years {
        added_date_ms: MIN_PLAN_PARAM_TIME_MS - 30 * 30 * 24 * 60 * 60 * 1000, // ~`30 months before.
        ten_year_duration: TenYearDuration {
            start: YearAndMonth {
                year: 2012,
                month: 10,
            },
            end: YearAndMonth {
                year: 2022,
                month: 9,
            },
        },
        value: 136.71,
    },
    AverageAnnualRealEarningsForSP500For10Years {
        // Monday, October 2, 2023 11:12:29.500 AM PDT
        added_date_ms: 1696270349500,
        ten_year_duration: TenYearDuration {
            start: YearAndMonth {
                year: 2013,
                month: 4,
            },
            end: YearAndMonth {
                year: 2023,
                month: 3,
            },
        },
        value: 143.91,
    },
    AverageAnnualRealEarningsForSP500For10Years {
        // Sun Nov 05 2023 07:53:20 PST
        added_date_ms: 1699199600576,
        ten_year_duration: TenYearDuration {
            start: YearAndMonth {
                year: 2013,
                month: 7,
            },
            end: YearAndMonth {
                year: 2023,
                month: 6,
            },
        },
        value: 146.14,
    },
    AverageAnnualRealEarningsForSP500For10Years {
        //  Wednesday, April 3, 2024 12:37:41.624 PM PDT
        added_date_ms: 1712173061624,
        ten_year_duration: TenYearDuration {
            start: YearAndMonth {
                year: 2014,
                month: 1,
            },
            end: YearAndMonth {
                year: 2023,
                month: 12,
            },
        },
        value: 150.89,
    },
    AverageAnnualRealEarningsForSP500For10Years {
        // Wednesday, August 14, 2024 12:01:00 PM PDT
        added_date_ms: 1723662060000,
        ten_year_duration: TenYearDuration {
            start: YearAndMonth {
                year: 2014,
                month: 4,
            },
            end: YearAndMonth {
                year: 2024,
                month: 3,
            },
        },
        value: 154.61,
    },
];
