use crate::{
    historical_monthly_returns::data::{
        AnnualLogMeanFromOneOverCAPERegressionInfo, FiveTenTwentyThirtyYearsSlopeAndIntercept,
    },
    shared_types::SlopeAndIntercept,
};

pub const V6_ANNUAL_LOG_MEAN_FROM_ONE_OVER_CAPE_REGRESSION_INFO_STOCKS:
    AnnualLogMeanFromOneOverCAPERegressionInfo = AnnualLogMeanFromOneOverCAPERegressionInfo {
    full: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 1.0299402457674502,
            intercept: -0.0040317246752935715,
        },
        ten_year: SlopeAndIntercept {
            slope: 0.8961137681650266,
            intercept: 0.0028528283660981257,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.6067212577586099,
            intercept: 0.02047271029386382,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.2568442240405936,
            intercept: 0.044572761459183916,
        },
    },
    restricted: FiveTenTwentyThirtyYearsSlopeAndIntercept {
        five_year: SlopeAndIntercept {
            slope: 0.9826790697581592,
            intercept: 0.013206790413477373,
        },
        ten_year: SlopeAndIntercept {
            slope: 1.105356822253614,
            intercept: -0.0007131609222229829,
        },
        twenty_year: SlopeAndIntercept {
            slope: 0.8617124521685182,
            intercept: 0.005596499522017122,
        },
        thirty_year: SlopeAndIntercept {
            slope: 0.27617221758398497,
            intercept: 0.04432266195131911,
        },
    },
};        
        