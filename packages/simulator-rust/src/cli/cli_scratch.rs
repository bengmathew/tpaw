use std::time::Instant;

use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::{ChaCha20Rng, ChaCha8Rng};
use simulator::plan_params::{self, process_plan_params::process_plan_params};
use std::io::prelude::*;

pub fn cli_scratch() {
    let plan_plan_params =
        serde_json::from_str::<plan_params::PlanParams>(PLAN_PARAMS_STR).unwrap();
    let market_data = serde_json::from_str::<
        simulator::data_for_market_based_plan_param_values::DataForMarketBasedPlanParamValues,
    >(MARKET_DATA_STR)
    .unwrap();

    let ppp = process_plan_params(&plan_plan_params, &market_data);
    // let ppp_small = PPPSmall::new(ppp);

    // for_buffer(
    //     "AVRO",
    //     &PPPAvro::process(process_plan_params(&plan_plan_params, &market_data).without_arrays),
    // );

    for_buffer("JSON", &serde_json::to_vec(&ppp).unwrap());

    for_buffer(
        "MessagePack",
        &rmp_serde::encode::to_vec_named(&ppp).unwrap(),
    );
}

fn for_buffer(name: &str, buf: &Vec<u8>) {
    println!("");
    println!("----{}", name);
    println!("buf: {}", buf.len() as f64 / 1000.0);

    // let mut e = ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    // e.write_all(&buf).unwrap();
    // let compressed = e.finish().unwrap();
    // println!("zlib: {}", compressed.len() as f64 / 1000.0);

    // let mut e = XzEncoder::new(Vec::new(), 9);
    // e.write_all(&buf).unwrap();
    // let compressed = e.finish().unwrap();
    // println!("lzma: {}", compressed.len() as f64 / 1000.0);
}

const PLAN_PARAMS_STR: &str = r#"{"constants":{"maxAge":1440,"riskToleranceNumIntegerValuesStartingFrom0":25,"riskToleranceStartRRA":16,"riskToleranceEndRRA":0.5},"ages":{"simulationMonths":{"numMonths":742,"numWithdrawalMonths":457,"lastMonthAsMFN":741,"withdrawalStartMonthAsMFN":285},"longerLivedPerson":{"maxAgeInMonths":1236,"currentAgeInMonths":495}},"wealth":{"futureSavings":[{"id":"feebiphizs","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":10000,"validMonthRange":{"start":0,"end":284},"monthRange":{"start":0,"end":284}}}],"incomeDuringRetirement":[{"id":"hrzsmdujpd","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":5000,"validMonthRange":{"start":283,"end":741},"monthRange":{"start":285,"end":741}}}]},"adjustmentsToSpending":{"extraSpending":{"essential":[{"id":"srjjqswpnk","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":4000,"validMonthRange":{"start":0,"end":741},"monthRange":{"start":0,"end":352}}},{"id":"fvpxstjfml","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":2000,"validMonthRange":{"start":0,"end":741},"monthRange":{"start":0,"end":172}}},{"id":"lzykdntqcg","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":500,"validMonthRange":{"start":0,"end":741},"monthRange":{"start":0,"end":148}}},{"id":"vkwtzracsk","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":2000,"validMonthRange":{"start":0,"end":741},"monthRange":{"start":0,"end":4}}},{"id":"uqavaxaxyg","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":500,"validMonthRange":{"start":0,"end":741},"monthRange":{"start":0,"end":148}}},{"id":"vicjhlgzau","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":800,"validMonthRange":{"start":0,"end":741},"monthRange":{"start":0,"end":148}}},{"id":"ibcehvmksf","nominal":false,"amountAndTiming":{"type":"recurring","baseAmount":800,"validMonthRange":{"start":0,"end":741},"monthRange":{"start":0,"end":172}}}],"discretionary":[]},"tpawAndSPAW":{"monthlySpendingCeiling":null,"monthlySpendingFloor":null,"legacy":{"total":0,"external":[{"id":"eunijmppod","label":null,"amount":0,"nominal":false,"sortIndex":0,"colorIndex":0}]}}},"risk":{"tpaw":{"riskTolerance":{"at20":13,"deltaAtMaxAge":-2,"forLegacyAsDeltaFromAt20":2},"timePreference":0,"additionalAnnualSpendingTilt":0},"tpawAndSPAW":{"lmp":0},"spaw":{"annualSpendingTilt":0.004},"spawAndSWR":{"allocation":{"now":{"stocks":0.5},"intermediate":[],"end":{"stocks":0.5}}},"swr":{"withdrawal":{"type":"asPercentPerYear","percentPerYear":0.034}}},"advanced":{"returnsStatsForPlanning":{"expectedValue":{"empiricalAnnualNonLog":{"type":"fixed","bonds":0.023,"stocks":0.049}},"standardDeviation":{"stocks":{"scale":{"log":1.14}},"bonds":{"scale":{"log":0}}}},"historicalReturnsAdjustment":{"standardDeviation":{"bonds":{"scale":{"log":1}},"overrideToFixedForTesting":false}},"sampling":{"type":"monteCarlo","blockSize":60,"staggerRunStarts":true},"annualInflation":{"type":"suggested"},"strategy":"TPAW"}}"#;
const MARKET_DATA_STR: &str = r#"{"closingTime":1706648400000,"inflation":{"closingTime":1706648400000,"value":0.0239},"sp500":{"closingTime":1706648400000,"value":4924.9702},"bondRates":{"closingTime":1706648400000,"fiveYear":0.0157,"sevenYear":0.0152,"tenYear":0.0149,"twentyYear":0.015,"thirtyYear":0.0154},"timestampForMarketData":9007199254740991}"#;
