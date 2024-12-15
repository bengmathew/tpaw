use criterion::{black_box, criterion_group, Criterion};
use dotenv::dotenv;
use simulator::{
    market_data::downloaded_market_data::get_market_data_series_for_simulation,
    simulate::{self, plan_params_server::get_test_plan_params_server},
};
#[tokio::main]
pub async fn criterion_benchmark(c: &mut Criterion) {
    dotenv().ok();
    let market_data_series_for_simulation = get_market_data_series_for_simulation(None).await;

    let current_portfolio_balance: f64 = 1000000.0;
    let plan_params_server =
        get_test_plan_params_server(1000, chrono::Utc::now().timestamp_millis());
    let num_months_to_simulate = plan_params_server.ages.simulation_months.num_months;
    let percentiles: Vec<u32> = vec![5u32, 50u32, 95u32];

    c.bench_function("simulate", |b| {
        b.iter(|| {
            let (result, plan_params_processed) = simulate::simulate(
                current_portfolio_balance,
                &percentiles,
                num_months_to_simulate,
                &plan_params_server,
                plan_params_server.evaluation_timestamp_ms,
                &market_data_series_for_simulation,
            );
            black_box(result.into_wire(None, plan_params_processed, 0, 0, 0));
        })
    });
    let (result, plan_params_processed) = simulate::simulate(
        current_portfolio_balance,
        &percentiles,
        num_months_to_simulate,
        &plan_params_server,
        plan_params_server.evaluation_timestamp_ms,
        &market_data_series_for_simulation,
    );
    c.bench_function("simulate_into_wire", |b| {
        b.iter_with_setup(
            || (result.clone(), plan_params_processed.clone()),
            |(result, plan_params_processed)| {
                black_box(result.into_wire(None, plan_params_processed, 0, 0, 0));
            },
        )
    });
}

criterion_group!(bench_simulate, criterion_benchmark);
