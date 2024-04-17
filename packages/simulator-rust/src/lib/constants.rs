use crate::round::RoundP;

// Prefer to have constants sent with plan_params instead of here.
pub const MAX_AGE_IN_MONTHS: usize = 120 * 12;
pub const MIN_PLAN_PARAM_TIME_MS: i64 = 1680481336120;

#[allow(non_snake_case)]
pub fn DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(retirement_length_in_months: i64) -> f64 {
    (0.7125 * (retirement_length_in_months as f64 / 12.0).powf(-0.859)).round_p(3)
}
