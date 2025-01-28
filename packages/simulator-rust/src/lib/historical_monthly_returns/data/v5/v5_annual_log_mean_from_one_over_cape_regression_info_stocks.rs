use crate::{
    historical_monthly_returns::data::{
        AnnualLogMeanFromOneOverCAPERegressionInfo, FiveTenTwentyThirtyYearsSlopeAndIntercept,
    },
    shared_types::SlopeAndIntercept,
};

pub const V5_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS:
    AnnualLogMeanFromOneOverCAPERegressionInfo = AnnualLogMeanFromOneOverCAPERegressionInfo {
    full: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.040640095817658,
            intercept: -0.005005583952886916,
        },
        ten_year: SlopeAndIntercept {
            slope: 0.9044909180493473,
            intercept: 0.0020773540712879096,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.6118863769594572,
            intercept: 0.019989613720509627,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.25995791783531136,
            intercept: 0.04425808988171827,
        },
    },
    restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 0.9993307948782261,
            intercept: 0.011831871029791746,
        },
        ten_year: SlopeAndIntercept {
            slope: 1.1214702924678372,
            intercept: -0.0020750399396809727,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.8755576002441773,
            intercept: 0.004399478929091372,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2855866468541192,
            intercept: 0.04345457230302298,
        },
    },
};        
        