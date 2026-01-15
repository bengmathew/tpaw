use crate::{
    historical_monthly_returns::data::{
        AnnualLogMeanFromOneOverCAPERegressionInfo, FiveTenTwentyThirtyYearsSlopeAndIntercept,
    },
    shared_types::SlopeAndIntercept,
};

pub const V7_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS:
    AnnualLogMeanFromOneOverCAPERegressionInfo = AnnualLogMeanFromOneOverCAPERegressionInfo {
    full: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.0186750091739176,
            intercept: -0.0030196513070538944,
        },
        ten_year: SlopeAndIntercept {
            slope: 0.8872973208298874,
            intercept: 0.003678563136717744,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.6013906822070715,
            intercept: 0.0209753430373825,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2534226892459835,
            intercept: 0.044901708118683506,
        },
    },
    restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 0.9649252188371695,
            intercept: 0.014643434263680727,
        },
        ten_year: SlopeAndIntercept {
            slope: 1.088798216190916,
            intercept: 0.0007109393929390362,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.8480459980120469,
            intercept: 0.006791270853428094,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2658049833900813,
            intercept: 0.04523120587150688,
        },
    },
};        
        