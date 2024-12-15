use lazy_static::lazy_static;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Mutex;

// TODO: After duration matching, remove this.
struct Store {
    data: Vec<Vec<usize>>,
    // This should be all the args into generate_random_index_sequences.
    seed: u64,
    start_run: usize,
    num_runs: usize,
    max_num_months: usize,
    block_size: usize,
    max_value: usize,
    stagger_run_starts: bool,
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
        stagger_run_starts: true,
    });
}

use rand::distributions::{Distribution, Uniform};

pub fn generate_random_index_sequences(
    seed: u64,
    start_run: usize,
    num_runs: usize,
    months_per_run: usize,
    block_size: usize,
    max_value: usize,
    stagger_run_starts: bool,
) -> Vec<Vec<usize>> {
    let run_seeds: Vec<u64> = Uniform::from(0..u64::MAX)
        .sample_iter(ChaCha8Rng::seed_from_u64(seed))
        .take(start_run + num_runs)
        .collect();
    // Important to not use usize for uniform because it is platform dependent.
    // This will lead to different random values for the same seed on eg. wasm
    // and native.
    let uniform = Uniform::from(0..(max_value as u64));
    let num_blocks = months_per_run / block_size + 1 + 1; // Extra +1 to account for staggering.

    let result: Vec<Vec<usize>> = (start_run..(start_run + num_runs))
        .map(|run_index| {
            let block_starting_month = uniform
                .sample_iter(ChaCha8Rng::seed_from_u64(run_seeds[run_index]))
                .take(num_blocks)
                .map(|x| x as usize)
                .collect::<Vec<usize>>();

            let stagger = if stagger_run_starts {
                run_index % block_size
            } else {
                0
            };
            return (0..months_per_run)
                .map(|i| {
                    // Staggering the i's so that block don't change at the same month
                    // accross different runs.
                    let staggered_i = i + stagger;
                    let block_index = staggered_i / block_size;
                    (block_starting_month[block_index] + staggered_i % block_size) % max_value
                })
                .collect();
        })
        .collect();
    result
}

pub fn memoized_random(
    seed: u64,
    start_run: usize,
    num_runs: usize,
    max_num_months: usize,
    block_size: usize,
    max_value: usize,
    stagger_run_starts: bool,
) -> &'static Vec<Vec<usize>> {
    let mut store = RANDOM_STORE.lock().unwrap();

    if store.seed != seed
        || store.start_run != start_run
        || store.num_runs != num_runs
        || store.block_size != block_size
        || store.max_num_months != max_num_months
        || store.max_value != max_value
        || store.stagger_run_starts != stagger_run_starts
    {
        store.seed = seed;
        store.start_run = start_run;
        store.num_runs = num_runs;
        store.max_num_months = max_num_months;
        store.block_size = block_size;
        store.max_value = max_value;
        store.stagger_run_starts = stagger_run_starts;
        store.data.clear();

        let mut tail = generate_random_index_sequences(
            seed,
            start_run,
            num_runs,
            max_num_months,
            block_size,
            max_value,
            stagger_run_starts,
        );

        store.data.append(&mut tail);
    }

    unsafe { std::mem::transmute(&store.data) }
}
