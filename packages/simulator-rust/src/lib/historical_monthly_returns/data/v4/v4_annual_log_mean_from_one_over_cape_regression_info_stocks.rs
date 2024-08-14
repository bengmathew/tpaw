use crate::{
    historical_monthly_returns::data::{
        AnnualLogMeanFromOneOverCAPERegressionInfo, FiveTenTwentyThirtyYearsSlopeAndIntercept,
    },
    shared_types::SlopeAndIntercept,
};

pub const V4_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS:
    AnnualLogMeanFromOneOverCAPERegressionInfo = AnnualLogMeanFromOneOverCAPERegressionInfo {
    full: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.0515621961683368,
            intercept: -0.0059918424807347825,
        },
        ten_year: SlopeAndIntercept {
            slope: 0.9126629044945024,
            intercept: 0.0013088469454574925,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.6169408694047565,
            intercept: 0.019512671828758814,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.26296739323182616,
            intercept: 0.0439521711872196,
        },
    },
    restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.0166273190921047,
            intercept: 0.010421586857509206,
        },
        ten_year: SlopeAndIntercept {
            slope: 1.136974033875041,
            intercept: -0.0034163763943588915,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.8891448562873637,
            intercept: 0.0032105511917436894,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2948871489091286,
            intercept: 0.042591817126255725,
        },
    },
};        
        