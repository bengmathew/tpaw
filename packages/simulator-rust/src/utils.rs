use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct ReturnsAtPointInTime {
    pub stocks: f64,
    pub bonds: f64,
}

// Not performant. Don't use in simulator.
pub fn blend_returns(returns: &ReturnsAtPointInTime) -> Box<dyn Fn(f64) -> f64> {
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

lazy_static! {
    static ref RANDOM_STORE: Mutex<Vec<Vec<usize>>> = Mutex::new(Vec::new());
}

pub fn clear_memoized_random_store() {
    let mut store = RANDOM_STORE.lock().unwrap();
    store.clear()
}

use rand::distributions::{Distribution, Uniform};

pub fn memoized_random(
    num_runs: usize,
    num_years: usize,
    max_value: usize,
    run_index: usize,
) -> &'static Vec<usize> {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::from(0..max_value);
    let mut store = RANDOM_STORE.lock().unwrap();
    if store.len() < num_runs {
        store.clear();
        let mut tail = (0..num_runs)
            .map(|_| uniform.sample_iter(&mut rng).take(num_years).collect())
            .collect();
        store.append(&mut tail);
    } else if store[0].len() < num_years {
        let shortfall = num_years - store[0].len();
        store.iter_mut().for_each(|x| {
            let mut tail = uniform.sample_iter(&mut rng).take(shortfall).collect();
            x.append(&mut tail);
        })
    }

    let result = &store[run_index];
    unsafe { std::mem::transmute(result) }
}

#[wasm_bindgen]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub mean: f64,
    pub variance: f64,
    pub standard_deviation: f64,
    pub n: usize,
}
pub fn stats(data: Box<[f64]>) -> Stats {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data
        .iter()
        .map(|value| {
            let diff = mean - value;
            diff.powi(2)
        })
        .sum::<f64>()
        / (n - 1) as f64;
    let standard_deviation = variance.sqrt();
    return Stats {
        mean,
        variance,
        standard_deviation,
        n,
    };
}
