use criterion::{black_box, criterion_group, Criterion};
use rand::Rng;
use simulator::utils::fget_item_at_or_before_key::FGetItemAtOrBeforeKey;

pub fn criterion_benchmark(c: &mut Criterion) {
    let n: i64 = 100_000;
    let step: i64 = 100;
    let data: Vec<i64> = (0..n).step_by(step as usize).collect();
    let mut rng = rand::thread_rng();
    c.bench_function("binary_search", |b| {
        b.iter_with_setup(
            || rng.gen_range(0..n) * step,
            |x| {
                black_box(data.binary_search(&x).unwrap_or_else(|x| x - 1));
            },
        )
    });

    struct DataForBy {
        timestamp_ms: i64,
    }
    let data_for_by: Vec<DataForBy> = (0..n)
        .step_by(step as usize)
        .map(|x| DataForBy {
            timestamp_ms: x * step,
        })
        .collect();
    c.bench_function("fget_item_at_or_before_key", |b| {
        b.iter_with_setup(
            || rng.gen_range(0..n) * step,
            |x| {
                black_box(data_for_by.fget_item_at_or_before_key(x, |x| x.timestamp_ms));
            },
        )
    });
}

criterion_group!(bench_utils, criterion_benchmark);
