use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::{ sync::Mutex};


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
}

impl AccountForWithdrawal {
    pub fn withdraw(&mut self, x: f64) -> f64 {
        let amount = f64::min(x, self.balance);
        self.balance -= amount;
        amount
    }
}

lazy_static! {
    static ref RANDOM_STORE: Mutex<Vec<Vec<usize>>> = Mutex::new(Vec::new());
}

use rand::distributions::{Distribution, Uniform};

pub fn memoized_random(
    num_runs: i32,
    num_years: usize,
    max_value: usize,
    run_index: usize,
) -> &'static Vec<usize> {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::from(0..max_value);
    let mut store = RANDOM_STORE.lock().unwrap();
    if store.len() < num_runs as usize {
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
