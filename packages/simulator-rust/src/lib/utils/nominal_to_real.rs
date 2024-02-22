#[inline(always)]
pub fn nominal_to_real(
    value: f64,
    is_nominal: bool,
    monthly_inflation: f64,
    months_from_now: i64,
) -> f64 {
    if is_nominal {
        value / (1.0 + monthly_inflation).powi(months_from_now as i32)
    } else {
        value
    }
}

#[cfg(test)]
mod test {
    use crate::utils::nominal_to_real::nominal_to_real;
    use rstest::rstest;

    #[rstest]
    #[case(100.0, true, 0.01, 1, 99.00990099009901)]
    #[case(100.0, true, 0.01, 10, 90.52869546929833)]
    #[case(100.0, false, 0.01, 1, 100.0)]
    fn test_nominal_to_real(
        #[case] value: f64,
        #[case] is_nominal: bool,
        #[case] monthly_inflation: f64,
        #[case] months_from_now: i64,
        #[case] expected: f64,
    ) {
        let diff =
            expected - nominal_to_real(value, is_nominal, monthly_inflation, months_from_now);
        assert!(diff < 0.000000000001, "{} is too high", diff);
    }
}
