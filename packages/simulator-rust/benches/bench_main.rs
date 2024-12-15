use criterion::criterion_main;

mod bench_estimate_portfolio_balance;
mod bench_process_plan_params_server;
mod bench_simulate;
mod bench_utils;
criterion_main!(
    bench_process_plan_params_server::bench_process_plan_params_server,
    bench_simulate::bench_simulate,
    bench_utils::bench_utils,
    bench_estimate_portfolio_balance::bench_estimate_portfolio_balance
);
