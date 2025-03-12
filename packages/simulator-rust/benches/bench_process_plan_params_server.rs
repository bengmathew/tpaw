use criterion::{criterion_group, Criterion};
use simulator::{
    historical_monthly_returns::{
        data::average_annual_real_earnings_for_sp500_for_10_years::AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS,
        HISTORICAL_MONTHLY_RETURNS,
    },
    market_data::market_data_defs::{
        DailyMarketDataForPresets, MarketDataAtTimestampForSimulation,
    },
    simulate::{
        plan_params_server::{
            PlanParamsServer_AmountTimed, PlanParamsServer_AmountTimed_DeltaEveryRecurrence,
            PlanParamsServer_AnnualInflation, PlanParamsServer_ExpectedReturnsForPlanning,
            PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog,
            PlanParamsServer_HistoricalReturnsAdjustment,
            PlanParamsServer_HistoricalReturnsAdjustment_StandardDeviation,
            PlanParamsServer_ReturnStatsForPlanning,
            PlanParamsServer_ReturnStatsForPlanning_StandardDeviation, PlanParamsServer_Sampling,
        },
        process_plan_params_server::{
            process_amount_timed::process_amount_timed,
            process_annual_inflation::process_annual_inflation,
            process_historical_returns::process_historical_returns,
            process_market_data_for_presets::process_market_data_for_presets,
            process_returns_stats_for_planning::process_returns_stats_for_planning,
        },
    },
    wire::{
        self, wire_plan_params_server_historical_returns_adjustment, WireLogDouble,
        WirePlanParamsServerSamplingMonteCarlo, WireScaleLogDouble,
    },
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let num_months: u32 = 12 * 120;

    // Market Data

    let market_data_at_timestamp_for_simulation = MarketDataAtTimestampForSimulation {
        daily_market_data_for_presets: &DailyMarketDataForPresets::new_for_testing(0),
        historical_monthly_returns_info: &HISTORICAL_MONTHLY_RETURNS.last().unwrap(),
        average_annual_real_earnings_for_sp500_for_10_years:
            &AVERAGE_ANNUAL_REAL_EARNINGS_FOR_SP500_FOR_10_YEARS
                .last()
                .unwrap(),
    };

    c.bench_function("process_plan_params_server_market_data_for_presets", |b| {
        b.iter(|| {
            process_market_data_for_presets(&market_data_at_timestamp_for_simulation);
        })
    });
    let market_data = process_market_data_for_presets(&market_data_at_timestamp_for_simulation);

    let returns_stats_for_planning_src = PlanParamsServer_ReturnStatsForPlanning {
        expected_value: PlanParamsServer_ExpectedReturnsForPlanning {
            empirical_annual_non_log:
                PlanParamsServer_ExpectedReturnsForPlanning_EmpiricalAnnualNonLog::Fixed {
                    stocks: 0.05,
                    bonds: 0.02,
                },
        },
        standard_deviation: PlanParamsServer_ReturnStatsForPlanning_StandardDeviation {
            stocks: WireScaleLogDouble {
                scale: WireLogDouble { log: 1.15 },
            },
        },
    };
    let sampling_src =
        PlanParamsServer_Sampling::MonteCarlo(WirePlanParamsServerSamplingMonteCarlo {
            seed: 1,
            num_runs: 1000,
            block_size: 5 * 60,
            stagger_run_starts: true,
        });

    c.bench_function(
        "process_plan_params_server_returns_stats_for_planning",
        |b| {
            b.iter(|| {
                process_returns_stats_for_planning(
                    &returns_stats_for_planning_src,
                    &sampling_src,
                    &market_data,
                    &market_data_at_timestamp_for_simulation.historical_monthly_returns_info,
                );
            })
        },
    );
    let returns_stats_for_planning = process_returns_stats_for_planning(
        &returns_stats_for_planning_src,
        &sampling_src,
        &market_data,
        &market_data_at_timestamp_for_simulation.historical_monthly_returns_info,
    );

    let inflation_src = PlanParamsServer_AnnualInflation::Manual(0.02);
    c.bench_function("process_plan_params_server_inflation", |b| {
        b.iter(|| {
            process_annual_inflation(&inflation_src, &market_data);
        })
    });
    let inflation = process_annual_inflation(&inflation_src, &market_data);

    // Amount Timed
    let amount_times_entry = PlanParamsServer_AmountTimed {
        id: "1".to_string(),
        is_nominal: true,
        month_range: Some(12 * 5..=12 * 50),
        valid_month_range: 0..=(num_months as i32),
        every_x_months: 1,
        base_amount: 1000.0,
        delta_every_recurrence: PlanParamsServer_AmountTimed_DeltaEveryRecurrence::Amount(0.0),
    };
    let future_savings = vec![amount_times_entry.clone()];
    let income_during_retirement = vec![amount_times_entry.clone()];
    let essential_expenses = vec![amount_times_entry.clone()];
    let discretionary_expenses = vec![amount_times_entry.clone()];
    c.bench_function("process_plan_params_server_amount_timed", |b| {
        b.iter(|| {
            process_amount_timed(
                num_months,
                &future_savings,
                &income_during_retirement,
                &essential_expenses,
                &discretionary_expenses,
                inflation.monthly,
            );
        })
    });

    let historical_returns_adjustment_src = PlanParamsServer_HistoricalReturnsAdjustment {
        standard_deviation: PlanParamsServer_HistoricalReturnsAdjustment_StandardDeviation {
            bonds: WireScaleLogDouble {
                scale: WireLogDouble { log: 1.0 },
            },
        },
        override_to_fixed_for_testing:
            wire_plan_params_server_historical_returns_adjustment::OverrideToFixedForTesting::None(
                wire::NoMessage {},
            ),
    };
    c.bench_function("process_plan_params_server_historical_returns", |b| {
        b.iter(|| {
            process_historical_returns(
                &returns_stats_for_planning,
                &historical_returns_adjustment_src,
                &market_data_at_timestamp_for_simulation.historical_monthly_returns_info,
            );
        })
    });
}

criterion_group!(bench_process_plan_params_server, criterion_benchmark);
