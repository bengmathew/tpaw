use crate::{
    historical_monthly_returns::data::{
        AnnualLogMeanFromOneOverCAPERegressionInfo, FiveTenTwentyThirtyYearsSlopeAndIntercept,
    },
    shared_types::SlopeAndIntercept,
};

pub const V3_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS:
    AnnualLogMeanFromOneOverCAPERegressionInfo = AnnualLogMeanFromOneOverCAPERegressionInfo {
    full: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.0566637598571609,
            intercept: -0.006450386098104471,
        },
        ten_year: SlopeAndIntercept {
            slope: 0.9164280996736617,
            intercept: 0.0009509620511941458,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.619325249977248,
            intercept: 0.019291132938121285,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2643478924933897,
            intercept: 0.043813044732791795,
        },
    },
    restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.024597338219725,
            intercept: 0.009776105467621227,
        },
        ten_year: SlopeAndIntercept {
            slope: 1.1440164916236495,
            intercept: -0.004035613248623282,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.8960140437279107,
            intercept: 0.002621888075961011,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2992642962448828,
            intercept: 0.04218934391445492,
        },
    },
};        
        