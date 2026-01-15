use crate::constants::MIN_PLAN_PARAM_TIME_MS;
use crate::utils::shared_types::YearAndMonth;
use crate::wire::{WireAverageAnnualRealEarningsForSp500For10Years, WireTenYearDuration};
use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TenYearDuration {
    pub start: YearAndMonth,
    pub end: YearAndMonth,
}

impl From<TenYearDuration> for WireTenYearDuration {
    fn from(value: TenYearDuration) -> Self {
        Self {
            start: (&value.start).into(),
            end: (&value.end).into(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AverageAnnualRealEarningsForSP500For10Years {
    pub added_date_ms: i64,
    pub ten_year_duration: TenYearDuration,
    pub value: f64,
}

impl From<AverageAnnualRealEarningsForSP500For10Years>
    for WireAverageAnnualRealEarningsForSp500For10Years
{
    fn from(value: AverageAnnualRealEarningsForSP500For10Years) -> Self {
        Self {
            added_date_ms: value.added_date_ms,
            ten_year_duration: value.ten_year_duration.into(),
            value: value.value,
        }
    }
}

// FEATURE: Version this the next time it is updated.
// NOTE: added_date_ms is the date it was *added* to the array, not when it was true in the
// world.
pub const AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS:
    [AverageAnnualRealEarningsForSP500For10Years; 8] = [
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
    AverageAnnualRealEarningsForSP500For10Years {
        // Tuesday, January 28, 2025 12:30:00 PM PST
        added_date_ms: 1738096200000,
        ten_year_duration: TenYearDuration {
            start: YearAndMonth {
                year: 2014,
                month: 10,
            },
            end: YearAndMonth {
                year: 2024,
                month: 9,
            },
        },
        value: 158.17
    },
    AverageAnnualRealEarningsForSP500For10Years {
        // August 5, 2025 3:00:00 PM GMT-07:00
        added_date_ms: 1754431200000,
        ten_year_duration: TenYearDuration {
            start: YearAndMonth {
                year: 2015,
                month: 4,
            },
            end: YearAndMonth {
                year: 2025,
                month: 3,
            },
        },
        value: 165.20
    },
    AverageAnnualRealEarningsForSP500For10Years {
        // Thursday, January 15, 2026 1:00:00 PM GMT-08:00
        added_date_ms: 1768510800000,
        ten_year_duration: TenYearDuration {
            start: YearAndMonth {
                year: 2015,
                month: 10,
            },
            end: YearAndMonth {
                year: 2025,
                month: 9,
            },
        },
        value: 170.80
    },
];
