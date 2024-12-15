use criterion::{black_box, criterion_group, Criterion};
use simulator::{
    estimate_portfolio_balance::{
        estimate_portfolio_balance,
        portfolio_balance_estimation_args::PortfolioBalanceEstimationArgs_ParamsForNonMarketAction,
    },
    historical_monthly_returns::{
        data::average_annual_real_earnings_for_sp500_for_10_years::AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS,
        HISTORICAL_MONTHLY_RETURNS,
    },
    market_data::market_data_defs::{
        DailyMarketDataForPresets, MarketDataSeriesForPortfolioBalanceEstimation,
        MarketDataSeriesForSimulation, VTAndBNDData, VTAndBNDData_PercentageChangeFromLastClose,
    },
    simulate::plan_params_server::get_test_plan_params_server,
    wire::WirePortfolioBalanceEstimationArgsNonMarketActionType,
};

#[tokio::main]
pub async fn criterion_benchmark(c: &mut Criterion) {
    let portfolio_balance_at_history_start: f64 = 1000000.0;
    let start_timestamp_ms = chrono::Utc::now().timestamp_millis();

    let mut plan_params_for_actions_unsorted: Vec<
        PortfolioBalanceEstimationArgs_ParamsForNonMarketAction,
    > = vec![];
    let mut vt_and_bnd_series: Vec<VTAndBNDData> = vec![];
    let mut daily_market_data_for_presets_series: Vec<DailyMarketDataForPresets> = vec![];

    for days in 0..100 {
        let day_of_month = days % 30;
        let closing_time_ms = start_timestamp_ms + days * 24 * 60 * 60 * 1000;
        let timestamp_ms = closing_time_ms + 1;

        plan_params_for_actions_unsorted.push(
            PortfolioBalanceEstimationArgs_ParamsForNonMarketAction {
                id: "".to_string(),
                plan_params: get_test_plan_params_server(1000, timestamp_ms),
                action_type: if day_of_month == 28 {
                    WirePortfolioBalanceEstimationArgsNonMarketActionType::WithdrawalAndContribution
                } else if day_of_month == 2 {
                    WirePortfolioBalanceEstimationArgsNonMarketActionType::MonthlyRebalance
                } else {
                    WirePortfolioBalanceEstimationArgsNonMarketActionType::PlanChange
                },
            },
        );
        vt_and_bnd_series.push(VTAndBNDData {
            closing_time_ms,
            percentage_change_from_last_close: VTAndBNDData_PercentageChangeFromLastClose {
                vt: 0.02,
                bnd: 0.01,
            },
        });

        daily_market_data_for_presets_series
            .push(DailyMarketDataForPresets::new_for_testing(closing_time_ms));
    }

    let market_data_series_for_portfolio_balance_estimation =
        MarketDataSeriesForPortfolioBalanceEstimation {
            vt_and_bnd_series: &vt_and_bnd_series,
        };

    let market_data_series_for_simulation = MarketDataSeriesForSimulation {
        daily_market_data_for_presets_series: &daily_market_data_for_presets_series,
        historical_monthly_returns_info_series: &HISTORICAL_MONTHLY_RETURNS,
        average_annual_real_earnings_for_sp500_for_10_years_series:
            &AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS,
    };

    let end_timestamp_ms = plan_params_for_actions_unsorted
        .last()
        .unwrap()
        .plan_params
        .evaluation_timestamp_ms;

    c.bench_function("estimate_portfolio_balance", |b| {
        b.iter_with_setup(
            || plan_params_for_actions_unsorted.clone(),
            |plan_params_for_actions_unsorted| {
                black_box(estimate_portfolio_balance(
                    portfolio_balance_at_history_start,
                    plan_params_for_actions_unsorted,
                    end_timestamp_ms,
                    &market_data_series_for_portfolio_balance_estimation,
                    &market_data_series_for_simulation,
                ));
            },
        )
    });
}

criterion_group!(bench_estimate_portfolio_balance, criterion_benchmark);
