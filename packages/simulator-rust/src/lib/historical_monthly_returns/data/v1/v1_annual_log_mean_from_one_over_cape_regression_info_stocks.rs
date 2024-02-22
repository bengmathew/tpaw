use crate::{
    historical_monthly_returns::data::{
        AnnualLogMeanFromOneOverCAPERegressionInfo, FiveTenTwentyThirtyYearsSlopeAndIntercept,
    },
    shared_types::SlopeAndIntercept,
};

pub const V1_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS:
    AnnualLogMeanFromOneOverCAPERegressionInfo = AnnualLogMeanFromOneOverCAPERegressionInfo {
    full: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.1183777088117077,
            intercept: -0.012081807482717852,
        },
        ten_year: SlopeAndIntercept {
            slope: 0.9607069551169595,
            intercept: -0.003516169985707203,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.6453769628119154,
            intercept: 0.01685632904478314,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2778518081256793,
            intercept: 0.04239954923881026,
        },
    },
    restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.1222721232597273,
            intercept: 0.0016598947032759742,
        },
        ten_year: SlopeAndIntercept {
            slope: 1.2219657457618343,
            intercept: -0.011588255661417635,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.9786668372309346,
            intercept: -0.004497980961483752,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.344898374695848,
            intercept: 0.03782988641028833,
        },
    },
};        
        