use std::time::Instant;

use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::{ChaCha20Rng, ChaCha8Rng};
use simulator::historical_monthly_returns::get_historical_monthly_returns_info;

pub fn cli_scratch() {
    let x = get_historical_monthly_returns_info(1712093932265);
    // print x version
    println!("{:?}", x.returns.bonds.annualized_stats.log);
    let y = get_historical_monthly_returns_info(1712093932265 - 24 * 60 * 60 * 1000);
    // print x version
    println!("{:?}", y.returns.bonds.annualized_stats.log);
}
