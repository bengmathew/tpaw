use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::{
    blend_returns,
    get_net_present_value_by_mfn::get_net_present_value_by_mfn,
    nominal_to_real::nominal_to_real,
    plan_params::normalize_plan_params::plan_params_normalized::{
        PlanParamsNormalized, ValueForMonthRange,
    },
    shared_types::{SimpleRange, StocksAndBonds},
};

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedByMonthParamsWealth {
    pub future_savings: ProcessedValueForMonthRanges,
    pub income_during_retirement: ProcessedValueForMonthRanges,
    #[serde(skip)]
    pub total: Vec<f64>,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedByMonthParamsAdjustmentsToSpendingExtraSpending {
    pub essential: ProcessedValueForMonthRanges,
    pub discretionary: ProcessedValueForMonthRanges,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedByMonthParamsAdjustmentsToSpending {
    pub extra_spending: ProcessedByMonthParamsAdjustmentsToSpendingExtraSpending,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedByMonthParamsRiskTPAWAndSPAW {
    pub lmp: Vec<f64>,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedByMonthParamsRisk {
    #[serde(rename = "tpawAndSPAW")]
    pub tpaw_and_spaw: ProcessedByMonthParamsRiskTPAWAndSPAW,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedByMonthParams {
    pub wealth: ProcessedByMonthParamsWealth,
    pub adjustments_to_spending: ProcessedByMonthParamsAdjustmentsToSpending,
    pub risk: ProcessedByMonthParamsRisk,
}

pub fn process_by_month_params(
    plan_params_norm: &PlanParamsNormalized,
    monthly_inflation: f64,
    // expected_monthly_non_log_returns: &StocksAndBonds<f64>,
) -> ProcessedByMonthParams {
    let num_months = plan_params_norm.ages.simulation_months.num_months;
    // let get_expected_returns_for_allocation = blend_returns(expected_monthly_non_log_returns);

    // let bonds_rate_by_mfn = vec![get_expected_returns_for_allocation(0.0); num_months as usize];

    ProcessedByMonthParams {
        wealth: {
            let future_savings = from_value_for_month_ranges(
                &plan_params_norm.wealth.future_savings,
                &plan_params_norm
                    .ages
                    .valid_month_ranges
                    .future_savings_as_mfn,
                num_months,
                monthly_inflation,
                // &bonds_rate_by_mfn,
            );
            let income_during_retirement = from_value_for_month_ranges(
                &plan_params_norm.wealth.income_during_retirement,
                &Some(
                    plan_params_norm
                        .ages
                        .valid_month_ranges
                        .income_during_retirement_as_mfn,
                ),
                num_months,
                monthly_inflation,
                // &bonds_rate_by_mfn,
            );
            ProcessedByMonthParamsWealth {
                total: future_savings
                    .total
                    .iter()
                    .zip(income_during_retirement.total.iter())
                    .map(|(x, y)| x + y)
                    .collect(),
                future_savings,
                income_during_retirement,
            }
        },
        adjustments_to_spending: ProcessedByMonthParamsAdjustmentsToSpending {
            extra_spending: ProcessedByMonthParamsAdjustmentsToSpendingExtraSpending {
                essential: from_value_for_month_ranges(
                    &plan_params_norm
                        .adjustments_to_spending
                        .extra_spending
                        .essential,
                    &Some(plan_params_norm.ages.valid_month_ranges.extra_spending),
                    num_months,
                    monthly_inflation,
                    // &bonds_rate_by_mfn
                ),
                discretionary: from_value_for_month_ranges(
                    &plan_params_norm
                        .adjustments_to_spending
                        .extra_spending
                        .discretionary,
                    &Some(plan_params_norm.ages.valid_month_ranges.extra_spending),
                    num_months,
                    monthly_inflation,
                ),
            },
        },
        risk: ProcessedByMonthParamsRisk {
            tpaw_and_spaw: ProcessedByMonthParamsRiskTPAWAndSPAW {
                lmp: {
                    let n = num_months as usize;
                    let mut lmp = vec![0.0; n];
                    for i in (plan_params_norm
                        .ages
                        .simulation_months
                        .withdrawal_start_month_as_mfn) as usize..n
                    {
                        lmp[i] = plan_params_norm.risk.tpaw_and_spaw.lmp;
                    }
                    lmp
                },
            },
        },
    }
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedValueForMonthRange_NetPresentValue_WithCurrentMonth {
    with_current_month: f64,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedValueForMonthRange_NetPresentValue {
    tpaw: ProcessedValueForMonthRange_NetPresentValue_WithCurrentMonth,
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedValueForMonthRange {
    #[serde(skip)]
    pub values: Vec<f64>,
    pub valid_range: Option<SimpleRange<i64>>,
    // pub net_present_value: ProcessedValueForMonthRange_NetPresentValue,
}

impl ProcessedValueForMonthRange {
    pub fn zeros(num_months: i64) -> Self {
        Self {
            values: vec![0.0; num_months as usize],
            valid_range: None,
            // net_present_value: ProcessedValueForMonthRange_NetPresentValue {
            //     tpaw: ProcessedValueForMonthRange_NetPresentValue_WithCurrentMonth {
            //         with_current_month: 0.0,
            //     },
            // },
        }
    }
}

#[derive(Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedValueForMonthRanges {
    pub by_id: HashMap<String, ProcessedValueForMonthRange>,
    #[serde(skip)]
    pub total: Vec<f64>,
}

fn from_value_for_month_range(
    value_for_month_range: &ValueForMonthRange,
    allowable_month_range: &Option<SimpleRange<i64>>,
    num_months: i64,
    monthly_inflation: f64,
    // tpaw_rate_array_for_net_present_value: &[f64],
) -> ProcessedValueForMonthRange {
    let ValueForMonthRange {
        month_range,
        value,
        nominal: is_nominal,
        ..
    } = value_for_month_range;
    let mut values = vec![0.0; num_months as usize];

    match allowable_month_range {
        None => ProcessedValueForMonthRange::zeros(num_months),
        Some(allowable_month_range) => {
            if allowable_month_range.end < month_range.start {
                return ProcessedValueForMonthRange::zeros(num_months);
            }

            let start = month_range.start.max(allowable_month_range.start);
            if month_range.end < start {
                return ProcessedValueForMonthRange::zeros(num_months);
            }
            let end = month_range.end.min(allowable_month_range.end);

            for mfn in start..=end {
                values[mfn as usize] = nominal_to_real(*value, *is_nominal, monthly_inflation, mfn);
            }

            // let net_present_value = ProcessedValueForMonthRange_NetPresentValue {
            //     tpaw: ProcessedValueForMonthRange_NetPresentValue_WithCurrentMonth {
            //         with_current_month: get_net_present_value_by_mfn(
            //             tpaw_rate_array_for_net_present_value,
            //             &values,
            //         )
            //         .with_current_month[0],
            //     },
            // };

            ProcessedValueForMonthRange {
                values,
                valid_range: Some(SimpleRange { start, end }),
                // net_present_value,
            }
        }
    }
}

fn from_value_for_month_ranges(
    value_for_month_ranges: &HashMap<String, ValueForMonthRange>,
    allowable_month_range: &Option<SimpleRange<i64>>,
    num_months: i64,
    monthly_inflation: f64,
    // tpaw_rate_array_for_net_present_value: &[f64],
) -> ProcessedValueForMonthRanges {
    let by_id: HashMap<String, ProcessedValueForMonthRange> = value_for_month_ranges
        .iter()
        .map(|(id, value_for_month_range)| {
            (
                id.clone(),
                from_value_for_month_range(
                    value_for_month_range,
                    allowable_month_range,
                    num_months,
                    monthly_inflation,
                    // tpaw_rate_array_for_net_present_value,
                ),
            )
        })
        .collect();

    let mut total = vec![0.0; num_months as usize];
    for i in 0..(num_months as usize) {
        total[i] = by_id.values().map(|x| x.values[i]).sum();
    }
    ProcessedValueForMonthRanges { by_id, total }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use rstest::*;

    use crate::{
        nominal_to_real::nominal_to_real,
        plan_params::normalize_plan_params::plan_params_normalized::ValueForMonthRange,
        shared_types::SimpleRange,
    };

    const VALUE: f64 = 100.0;
    const MONTHLY_INFLATION: f64 = 0.01;
    const NUM_MONTHS: i64 = 10;
    const VALID_MONTH_RANGE: Option<SimpleRange<i64>> = Some(SimpleRange { start: 2, end: 5 });
    fn get_value_for_month_range(month_range: SimpleRange<i64>) -> ValueForMonthRange {
        ValueForMonthRange {
            month_range,
            value: VALUE,
            nominal: true,
            label: None,
            id: "".to_string(),
            sort_index: 1,
            color_index: 1,
        }
    }
    fn get_expected_value(mfn: usize) -> (usize, f64) {
        (
            mfn,
            nominal_to_real(VALUE, true, MONTHLY_INFLATION, mfn as i64),
        )
    }

    #[rstest]
    #[case(SimpleRange{start:0, end:1}, &[],None )]
    #[case(SimpleRange{start:0, end:2}, &[get_expected_value(2)],Some(SimpleRange{start:2, end:2})  )]
    #[case(SimpleRange{start:2, end:3}, &[get_expected_value(2), get_expected_value(3)],Some(SimpleRange{start:2, end:3}) )]
    #[case(SimpleRange{start:4, end:6}, &[get_expected_value(4), get_expected_value(5)],Some(SimpleRange{start:4, end:5})  )]
    #[case(SimpleRange{start:5, end:6}, &[get_expected_value(5)],Some(SimpleRange{start:5, end:5})  )]
    #[case(SimpleRange{start:6, end:7}, &[],None  )]
    #[case(SimpleRange{start:0, end:-1}, &[],None )]
    #[case(SimpleRange{start:0, end:-2}, &[],None  )]
    #[case(SimpleRange{start:2, end:1}, &[],None  )]
    #[case(SimpleRange{start:4, end:3}, &[],None  )]
    #[case(SimpleRange{start:5, end:4}, &[],None  )]
    #[case(SimpleRange{start:6, end:1}, &[],None  )]
    fn test_from_value_for_month_range(
        #[case] month_range: SimpleRange<i64>,
        #[case] non_zero_expected_values: &[(usize, f64)],
        #[case] expected_valid_range: Option<SimpleRange<i64>>,
    ) {
        let value_for_month_range = get_value_for_month_range(month_range);

        let result = super::from_value_for_month_range(
            &value_for_month_range,
            &VALID_MONTH_RANGE,
            NUM_MONTHS,
            MONTHLY_INFLATION,
        );
        let non_zero_expected_values_map: HashMap<usize, f64> =
            non_zero_expected_values.iter().cloned().collect();

        for (i, v) in result.values.iter().enumerate() {
            let expected = if let Some(x) = non_zero_expected_values_map.get(&i) {
                *x
            } else {
                0.0
            };
            assert_eq!(*v, expected);
        }
        assert_eq!(result.values.len(), NUM_MONTHS as usize);
        assert_eq!(result.valid_range, expected_valid_range);
    }

    #[rstest]
    fn test_from_value_for_month_range_no_range() {
        let value_for_month_range = get_value_for_month_range(SimpleRange { start: 0, end: 5 });

        let result = super::from_value_for_month_range(
            &value_for_month_range,
            &None,
            NUM_MONTHS,
            MONTHLY_INFLATION,
        );

        for (i, v) in result.values.iter().enumerate() {
            assert_eq!(*v, 0.0);
        }
        assert_eq!(result.values.len(), NUM_MONTHS as usize);
        assert_eq!(result.valid_range, None);
    }

    #[rstest]
    fn test_from_value_for_month_ranges() {
        let mut value_for_month_ranges: HashMap<String, ValueForMonthRange> = HashMap::new();
        value_for_month_ranges.insert(
            "1".to_string(),
            get_value_for_month_range(SimpleRange { start: 0, end: 0 }),
        );
        value_for_month_ranges.insert(
            "2".to_string(),
            get_value_for_month_range(SimpleRange { start: 0, end: 0 }),
        );
        let result = super::from_value_for_month_ranges(
            &value_for_month_ranges,
            &Some(SimpleRange { start: 0, end: 0 }),
            NUM_MONTHS,
            MONTHLY_INFLATION,
        );
        assert_eq!(result.by_id.len(), 2);
        assert_eq!(result.total[0], 200.0);
        assert_eq!(result.total[1], 0.0);
    }
}
