use lazy_static::lazy_static;
use rand::SeedableRng;
use rand_chacha::{ChaCha20Rng, ChaCha8Rng};
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

struct Store {
    data: Vec<Vec<usize>>,
    // This should be all the args into generate_random_index_sequences.
    seed: u64,
    start_run: usize,
    num_runs: usize,
    max_num_months: usize,
    block_size: usize,
    max_value: usize,
}

lazy_static! {
    static ref RANDOM_STORE: Mutex<Store> = Mutex::new(Store {
        data: Vec::new(),
        seed: 0,
        start_run: 0,
        num_runs: 0,
        max_num_months: 0,
        block_size: 0,
        max_value: 0,
    });
}

use rand::distributions::{Distribution, Uniform};

pub fn generate_random_index_sequences(
    seed: u64,
    start_run: usize,
    num_runs: usize,
    max_num_months: usize,
    block_size: usize,
    max_value: usize,
) -> Vec<Vec<usize>> {
    let run_seeds: Vec<u64> = Uniform::from(0..u64::MAX)
        .sample_iter(ChaCha20Rng::seed_from_u64(seed))
        .take(start_run + num_runs)
        .collect();
    let uniform = Uniform::from(0..max_value);
    let num_blocks = max_num_months / block_size + 1 + 1; // Extra +1 to account for staggering.

    return (start_run..(start_run + num_runs))
        .map(|run_index| {
            let block_starting_month: Vec<usize> = uniform
                .sample_iter(ChaCha8Rng::seed_from_u64(run_seeds[run_index]))
                .take(num_blocks)
                .collect();
            return (0..max_num_months)
                .map(|i| {
                    // Staggering the i's so that block don't change at the same month
                    // accross different runs.
                    let staggered_i = i + (run_index % block_size);
                    let block_index = staggered_i / block_size;
                    (block_starting_month[block_index] + staggered_i % block_size) % max_value
                })
                .collect();
        })
        .collect();
}

pub fn memoized_random(
    seed: u64,
    start_run: usize,
    num_runs: usize,
    max_num_months: usize,
    block_size: usize,
    max_value: usize,
) -> &'static Vec<Vec<usize>> {
    let mut store = RANDOM_STORE.lock().unwrap();

    if store.seed != seed
        || store.start_run != start_run
        || store.num_runs != num_runs
        || store.block_size != block_size
        || store.max_num_months != max_num_months
        || store.max_value != max_value
    {
        store.seed = seed;
        store.start_run = start_run;
        store.num_runs = num_runs;
        store.max_num_months = max_num_months;
        store.block_size = block_size;
        store.max_value = max_value;
        store.data.clear();

        let mut tail = generate_random_index_sequences(
            seed,
            start_run,
            num_runs,
            max_num_months,
            block_size,
            max_value,
        );

        // web_sys::console::log_1(&wasm_bindgen::JsValue::from_serde(&(&tail)).unwrap());
        store.data.append(&mut tail);
    }

    unsafe { std::mem::transmute(&store.data) }
}

#[wasm_bindgen]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub mean: f64,
    pub variance: f64,
    pub standard_deviation: f64,
    pub n: usize,
}
pub fn get_stats(data: &Vec<f64>) -> Stats {
    let n = data.len();
    let mean = get_mean(&data);
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
pub fn get_mean(data: &Vec<f64>) -> f64 {
    data.iter().sum::<f64>() / (data.len() as f64)
}

pub fn get_log_returns(returns: &Vec<f64>) -> Vec<f64> {
    returns.iter().map(|x| (1.0 + x).ln()).collect()
}

#[wasm_bindgen]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct StatsForWindowSize {
    pub n: usize,
    pub of_base: Stats,
    pub of_log: Stats,
}

pub fn get_stats_for_window_size_from_log_returns(
    ln_returns: &Vec<f64>,
    window_size: usize,
) -> StatsForWindowSize {
    let mut prev = 0.0;
    let mut prev_gross = ln_returns[0..(window_size - 1)].iter().sum::<f64>();
    let n = ln_returns.len() - window_size - 1;
    let gross: Vec<f64> = ln_returns[0..n]
        .iter()
        .enumerate()
        .map(|(i, curr)| {
            prev_gross = prev_gross - prev + ln_returns[i + window_size - 1];
            prev = *curr;
            return prev_gross;
        })
        .collect();
    return StatsForWindowSize {
        n: gross.len(),
        of_base: get_stats(&gross.iter().map(|x| x.exp() - 1.0).collect()),
        of_log: get_stats(&gross),
    };
}
