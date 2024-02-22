use crate::{
    historical_monthly_returns::data::{
        AnnualLogMeanFromOneOverCAPERegressionInfo, FiveTenTwentyThirtyYearsSlopeAndIntercept,
    },
    shared_types::SlopeAndIntercept,
};

pub const V2_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS:
    AnnualLogMeanFromOneOverCAPERegressionInfo = AnnualLogMeanFromOneOverCAPERegressionInfo {
    full: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.0633342528471696,
            intercept: -0.00705066774526697,
        },
        ten_year: SlopeAndIntercept {
            slope: 0.9217641012864941,
            intercept: 0.000440036102269864,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.6227070319793933,
            intercept: 0.018974019488398705,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2662449195138592,
            intercept: 0.043623828945467,
        },
    },
    restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.0341155536379807,
            intercept: 0.009002231982247103,
        },
        ten_year: SlopeAndIntercept {
            slope: 1.1538153299999194,
            intercept: -0.004906916224730287,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.905964132253405,
            intercept: 0.0017586605872627267,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.3054457002927557,
            intercept: 0.04162683202149632,
        },
    },
};        
        