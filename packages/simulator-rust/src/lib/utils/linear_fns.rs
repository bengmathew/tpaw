
pub fn linear_fn_from_points(x0: f64, y0: f64, x1: f64, y1: f64) -> impl Fn(f64) -> f64 {
    let slope = (y1 - y0) / (x1 - x0);
    linear_fn_from_point_and_slope(x0, y0, slope)
}

pub fn linear_fn_from_point_and_slope(x: f64, y: f64, slope: f64) -> impl Fn(f64) -> f64 {
    let intercept = y - slope * x;
    linear_fn_from_slope_and_intercept(slope, intercept)
}

pub fn linear_fn_from_slope_and_intercept(slope: f64, intercept: f64) -> impl Fn(f64) -> f64 {
    move |x: f64| slope * x + intercept
}
