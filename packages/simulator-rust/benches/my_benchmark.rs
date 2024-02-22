use criterion::{black_box, criterion_group, criterion_main, Criterion};
use simulator::{
    constants::MAX_AGE_IN_MONTHS,
    historical_monthly_returns::{
        data::v2::V2_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS,
        get_historical_monthly_returns, HistoricalReturnsId,
    },
    params::ParamsStrategy,
    plan_params::{
        normalize_plan_params::plan_params_normalized,
        process_plan_params::{
            process_by_month_params::process_by_month_params,
            process_expected_returns_for_planning::{
                process_expected_returns_for_planning,
                process_market_data_for_expected_returns_for_planning_presets,
            },
            process_historical_monthly_log_returns_adjustment::process_historical_monthly_log_returns_adjustment,
            process_plan_params,
        },
    },
    run, ParamsSWRWithdrawalType,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let plan_params_norm =
        serde_json::from_str::<plan_params_normalized::PlanParamsNormalized>(PLAN_PARAMS_NORM_STR)
            .unwrap();
    let market_data = serde_json::from_str::<
        simulator::data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues,
    >(MARKET_DATA_STR)
    .unwrap();

    c.bench_function("process_plan_params", |b| {
        b.iter(|| {
            process_plan_params(&plan_params_norm, &market_data);
        })
    });

    c.bench_function(
        "process_market_data_for_expected_returns_for_planning_presets",
        |b| {
            b.iter(|| {
                process_market_data_for_expected_returns_for_planning_presets(
                    &market_data,
                    &plan_params_norm.advanced.sampling,
                    &plan_params_norm
                        .advanced
                        .historical_monthly_log_returns_adjustment
                        .standard_deviation,
                );
            })
        },
    );

    c.bench_function("process_expected_returns_for_planning", |b| {
        b.iter(|| {
            process_expected_returns_for_planning(
                &plan_params_norm.advanced.expected_returns_for_planning,
                &plan_params_norm.advanced.sampling,
                &plan_params_norm
                    .advanced
                    .historical_monthly_log_returns_adjustment
                    .standard_deviation,
                &market_data,
            );
        })
    });

    let expected_returns_for_planning = process_expected_returns_for_planning(
        &plan_params_norm.advanced.expected_returns_for_planning,
        &plan_params_norm.advanced.sampling,
        &plan_params_norm
            .advanced
            .historical_monthly_log_returns_adjustment
            .standard_deviation,
        &market_data,
    );

    c.bench_function("process_historical_monthly_log_returns_adjustment", |b| {
        b.iter(|| {
            process_historical_monthly_log_returns_adjustment(
                &expected_returns_for_planning,
                &market_data,
                false,
            );
        })
    });

    c.bench_function("process_by_month_params", |b| {
        b.iter(|| {
            process_by_month_params(&plan_params_norm, 0.01);
        })
    });

    c.bench_function("run", |b| {
        let h =
            get_historical_monthly_returns(V2_HISTORICAL_MONTHLY_RETURNS_EFFECTIVE_TIMESTAMP_MS);
        b.iter(|| {
            run(
                ParamsStrategy::TPAW,
                0,
                500,
                50 * 12,
                50 * 12,
                0,
                0.05,
                0.02,
                h.stocks.log.series.to_vec().into_boxed_slice(),
                h.bonds.log.series.to_vec().into_boxed_slice(),
                10000.0,
                vec![0.5; 50 * 12].into_boxed_slice(),
                vec![0.5; 50 * 12].into_boxed_slice(),
                0.5,
                ParamsSWRWithdrawalType::AsPercent,
                0.04,
                vec![0.0; 50 * 12].into_boxed_slice(),
                vec![0.0; 50 * 12].into_boxed_slice(),
                vec![0.0; 50 * 12].into_boxed_slice(),
                vec![0.0; 50 * 12].into_boxed_slice(),
                0.0,
                0.0,
                vec![0.01; 50 * 12].into_boxed_slice(),
                None,
                None,
                true,
                5 * 12,
                true,
                MAX_AGE_IN_MONTHS,
                256893116,
                None,
                None,
            );
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

const PLAN_PARAMS_NORM_STR: &str = r#"{"v":27,"timestamp":1708130248368,"dialogPositionNominal":"done","ages":{"person1":{"monthOfBirthAsMFN":-289,"maxAgeAsMFN":398,"retirement":{"ageAsMFNIfInFutureElseNull":null,"ageAsMFNIfSpecifiedElseNull":null,"isRetired":true}},"person2":null,"simulationMonths":{"numMonths":399,"lastMonthAsMFN":398,"withdrawalStartMonthAsMFN":0},"validMonthRanges":{"futureSavingsAsMFN":null,"incomeDuringRetirementAsMFN":{"start":0,"end":398},"extraSpending":{"start":0,"end":398}}},"wealth":{"portfolioBalance":{"updatedTo":26000,"updatedAtId":"c2ee245d-959a-49e3-bb28-bb61f49f982f","updatedHere":false,"updatedAtTimestamp":1707941306220},"futureSavings":{},"incomeDuringRetirement":{"dcuvjlbclk":{"id":"dcuvjlbclk","sortIndex":0,"label":null,"value":200,"nominal":false,"colorIndex":0,"monthRange":{"start":0,"end":398}}}},"adjustmentsToSpending":{"extraSpending":{"essential":{"aphtemlzqn":{"id":"aphtemlzqn","sortIndex":0,"label":null,"value":100,"nominal":false,"colorIndex":0,"monthRange":{"start":0,"end":59}}},"discretionary":{"ugrgljsglh":{"id":"ugrgljsglh","sortIndex":0,"label":null,"value":100,"nominal":false,"colorIndex":1,"monthRange":{"start":0,"end":59}}}},"tpawAndSPAW":{"legacy":{"total":0,"external":{}},"monthlySpendingFloor":null,"monthlySpendingCeiling":null}},"risk":{"tpaw":{"riskTolerance":{"at20":12,"deltaAtMaxAge":-2,"forLegacyAsDeltaFromAt20":2},"timePreference":0,"additionalAnnualSpendingTilt":0},"tpawAndSPAW":{"lmp":0},"spaw":{"annualSpendingTilt":0.008},"spawAndSWR":{"allocation":{"start":{"month":0,"stocks":0.5},"intermediate":{},"end":{"stocks":0.5}}},"swr":{"withdrawal":{"type":"default"}}},"advanced":{"sampling":{"type":"historical","forMonteCarlo":{"blockSize":240,"staggerRunStarts":true}},"strategy":"TPAW","annualInflation":{"type":"suggested"},"expectedReturnsForPlanning":{"type":"historical"},"historicalMonthlyLogReturnsAdjustment":{"overrideToFixedForTesting":false,"standardDeviation":{"bonds":{"enableVolatility":false},"stocks":{"scale":1.5}}}},"results":{"displayedAssetAllocation":{"stocks":0.6844}}}"#;
const MARKET_DATA_STR: &str = r#"{"inflation":{"value":0.0239},"sp500":{"closingTime":1706648400000,"value":4924.9702},"bondRates":{"closingTime":1706648400000,"fiveYear":0.0157,"sevenYear":0.0152,"tenYear":0.0149,"twentyYear":0.015,"thirtyYear":0.0154},"returnsId":"v2"}"#;
