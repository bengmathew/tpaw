pub mod expected_value_of_returns;
pub mod get_net_present_value_by_mfn;
pub mod nominal_to_real;
pub mod random;
pub mod return_series;
pub mod round;
pub mod shared_types;

use self::shared_types::StocksAndBonds;


// Not performant. Don't use in simulator.
pub fn blend_returns(returns: &StocksAndBonds<f64>) -> Box<dyn Fn(f64) -> f64> {
    let stocks = returns.stocks;
    let bonds = returns.bonds;
    let x = Box::new(move |stock_allocation: f64| {
        bonds * (1.0 - stock_allocation) + stocks * stock_allocation
    });
    x
}

pub struct AccountForWithdrawal {
    pub balance: f64,
    pub insufficient_funds: bool,
}

impl AccountForWithdrawal {
    pub fn new(balance: f64) -> Self {
        Self {
            balance,
            insufficient_funds: false,
        }
    }
    pub fn withdraw(&mut self, x: f64) -> f64 {
        let amount = f64::min(x, self.balance);
        self.balance -= amount;
        if amount < x {
            self.insufficient_funds = true
        }
        return amount;
    }
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => (web_sys::console::log_1(&format_args!($($t)*).to_string().into()))
}

pub fn vec_f64_js_view(x: &Vec<f64>) -> js_sys::Float64Array {
    unsafe { js_sys::Float64Array::view(&x[..]) }
}
pub fn vec_i32_js_view(x: &Vec<i32>) -> js_sys::Int32Array {
    unsafe { js_sys::Int32Array::view(&x[..]) }
}
