
pub trait RoundP<T> {
    fn round_p(&self, precision: i32) -> T;
}

impl RoundP<f64> for f64 {
    fn round_p(&self, precision: i32) -> f64 {
        let factor = 10_f64.powi(precision as i32);
        (self * factor).round() / factor
    }
}