pub struct NetPresentValueByMFN {
    pub with_current_month: Vec<f64>,
    pub without_current_month: Vec<f64>,
}

pub fn get_net_present_value_by_mfn(r: &[f64], amounts: &[f64]) -> NetPresentValueByMFN {
    let n = amounts.len();
    assert!(r.len() == n);
    let mut with_current_month = vec![0.0; n];
    let mut without_current_month = vec![0.0; n];

    for i in (0..n).rev() {
        without_current_month[i] = if i == n - 1 {
            0.0
        } else {
            with_current_month[i + 1] / (1.0 + r[i])
        };
        with_current_month[i] = amounts[i] + without_current_month[i];
    }

    NetPresentValueByMFN {
        with_current_month,
        without_current_month,
    }
}

pub fn cumulative_1_plus_g_over_1_plus_r(r: &[f64], g: &[f64]) -> Vec<f64> {
    let n = r.len();
    let mut result = vec![0.0; n];
    for i in (0..n).rev() {
        let x = if i == n - 1 {
            0.0
        } else {
            (result[i + 1] * (1.0 + g[i])) / (1.0 + r[i])
        };
        result[i] = 1.0 + x
    }
    result
}
