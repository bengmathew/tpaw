pub mod plan_params_normalized;

// use std::collections::HashMap;

// use crate::{console_log, shared_types::SimpleRange};

// use self::plan_params_normalized::{
//     NormalizedAges, NormalizedAges_Person, NormalizedAges_Person_Retirement,
//     NormalizedAges_SimulationMonths, NormalizedAges_ValidMonthRanges, NormalizedValueForMonthRange,
//     PlanParamsNormalized,
// };

// use super::{
//     AdjustmentsToSpending, CalendarMonth, ExtraSpending, GlidePath, GlidePathIntermediateEntry,
//     Month, MonthRange, People, PersonType, PlanParams, Risk_SPAWAndSWR, ValueForMonthRange,
//     ValueForMonthRanges, Wealth,
// };


// pub fn normalize_plan_params(
//     plan_params: &PlanParams,
//     now_as_calendar_month: &CalendarMonth,
// ) -> PlanParamsNormalized {
//     let ages = normalize_ages(&plan_params.people, now_as_calendar_month);

//     let (wealth, adjustments_to_spending, risk) = {
//         let normalize_month =
//             get_normalize_month(&ages.person1, &ages.person2, now_as_calendar_month);
//         let normalize_month_range = get_normalize_month_range(&normalize_month);

//         let normalize_value_for_month_range =
//             |orig: &ValueForMonthRange<MonthRange>| NormalizedValueForMonthRange {
//                 value: orig.value,
//                 month_range: normalize_month_range(&orig.month_range),
//                 label: orig.label.clone(),
//                 nominal: orig.nominal,
//                 id: orig.id.clone(),
//                 sort_index: orig.sort_index,
//                 color_index: orig.color_index,
//             };
//         let normalize_value_for_month_ranges = |orig: &ValueForMonthRanges<MonthRange>| -> HashMap<
//             String,
//             NormalizedValueForMonthRange,
//         > {
//             orig.into_iter()
//                 .map(|(k, v)| (k.clone(), normalize_value_for_month_range(&v)))
//                 .collect()
//         };

//         let normalize_glide_path = |orig: &GlidePath<CalendarMonth, Month>| -> GlidePath<i64, i64> {
//             GlidePath {
//                 start: super::MonthAndStocks {
//                     month: orig.start.month.minus_in_months(now_as_calendar_month),
//                     stocks: orig.start.stocks,
//                 },
//                 intermediate: orig
//                     .intermediate
//                     .iter()
//                     .map(|(k, v)| {
//                         (
//                             k.clone(),
//                             GlidePathIntermediateEntry {
//                                 id: v.id.clone(),
//                                 month: normalize_month(&v.month),
//                                 index_to_sort_by_added: v.index_to_sort_by_added,
//                                 stocks: v.stocks,
//                             },
//                         )
//                     })
//                     .collect(),
//                 end: orig.end.clone(),
//             }
//         };

//         let wealth = Wealth {
//             portfolio_balance: plan_params.wealth.portfolio_balance.clone(),
//             future_savings: normalize_value_for_month_ranges(&plan_params.wealth.future_savings),
//             income_during_retirement: normalize_value_for_month_ranges(
//                 &plan_params.wealth.income_during_retirement,
//             ),
//         };

//         let adjustments_to_spending = AdjustmentsToSpending {
//             extra_spending: ExtraSpending {
//                 essential: normalize_value_for_month_ranges(
//                     &plan_params.adjustments_to_spending.extra_spending.essential,
//                 ),
//                 discretionary: normalize_value_for_month_ranges(
//                     &plan_params
//                         .adjustments_to_spending
//                         .extra_spending
//                         .discretionary,
//                 ),
//             },
//             tpaw_and_spaw: plan_params.adjustments_to_spending.tpaw_and_spaw.clone(),
//         };

//         let risk = super::Risk {
//             tpaw: plan_params.risk.tpaw.clone(),
//             tpaw_and_spaw: plan_params.risk.tpaw_and_spaw.clone(),
//             spaw: plan_params.risk.spaw.clone(),
//             spaw_and_swr: Risk_SPAWAndSWR {
//                 allocation: normalize_glide_path(&plan_params.risk.spaw_and_swr.allocation),
//             },
//             swr: plan_params.risk.swr.clone(),
//         };
//         (wealth, adjustments_to_spending, risk)
//     };
//     PlanParamsNormalized {
//         v: plan_params.v.clone(),
//         timestamp: plan_params.timestamp,
//         dialog_position_nominal: plan_params.dialog_position_nominal.clone(),
//         ages,
//         wealth,
//         adjustments_to_spending,
//         risk,
//         advanced: plan_params.advanced.clone(),
//         results: plan_params.results.clone(),
//     }
// }

// fn normalize_ages_for_person(
//     orig: &super::Ages,
//     now_as_calendar_month: &CalendarMonth,
// ) -> NormalizedAges_Person {
//     let (month_of_birth, max_age) = match orig {
//         super::Ages::RetiredWithNoRetirementDateSpecified {
//             month_of_birth,
//             max_age,
//             ..
//         } => (month_of_birth, max_age),
//         super::Ages::RetirementDateSpecified {
//             month_of_birth,
//             max_age,
//             ..
//         } => (month_of_birth, max_age),
//     };
//     let month_of_birth_as_mfn = month_of_birth.minus_in_months(&now_as_calendar_month);
//     NormalizedAges_Person {
//         month_of_birth_as_mfn,
//         max_age_as_mfn: month_of_birth_as_mfn + max_age.in_months,
//         retirement: {
//             match orig {
//                 super::Ages::RetiredWithNoRetirementDateSpecified { .. } => {
//                     NormalizedAges_Person_Retirement {
//                         age_as_mfn_if_in_future_else_null: None,
//                         age_as_mfn_if_specified_else_null: None,
//                         is_retired: true,
//                     }
//                 }
//                 super::Ages::RetirementDateSpecified { retirement_age, .. } => {
//                     let age_as_mfn_if_specified_else_null =
//                         (month_of_birth_as_mfn + retirement_age.in_months);

//                     let age_as_mfn_if_in_future_else_null = if age_as_mfn_if_specified_else_null > 0
//                     {
//                         Some(age_as_mfn_if_specified_else_null)
//                     } else {
//                         None
//                     };
//                     NormalizedAges_Person_Retirement {
//                         age_as_mfn_if_specified_else_null: Some(age_as_mfn_if_specified_else_null),
//                         age_as_mfn_if_in_future_else_null,
//                         is_retired: age_as_mfn_if_in_future_else_null.is_none(),
//                     }
//                 }
//             }
//         },
//     }
// }

// fn get_simulation_months(
//     person1: &NormalizedAges_Person,
//     person2: &Option<NormalizedAges_Person>,
//     withdrawal_start_at_retirement_of: &PersonType,
// ) -> NormalizedAges_SimulationMonths {
//     let last_month_as_mfn = person1
//         .max_age_as_mfn
//         .max(person2.as_ref().map(|x| x.max_age_as_mfn).unwrap_or(0));

//     let withdrawal_start_month_as_mfn = (match withdrawal_start_at_retirement_of {
//         PersonType::Person1 => person1,
//         PersonType::Person2 => &person2.as_ref().unwrap(),
//     })
//     .retirement
//     .age_as_mfn_if_in_future_else_null
//     .unwrap_or(0);
//     NormalizedAges_SimulationMonths {
//         num_months: last_month_as_mfn + 1,
//         last_month_as_mfn,
//         withdrawal_start_month_as_mfn,
//     }
// }

// fn get_valid_month_range_for_future_savings_as_mfn(
//     person1_retirement_age_if_in_future_else_null: Option<i64>,
//     if_person2_person2_retirement_age_if_in_future_else_null: Option<Option<i64>>,
// ) -> Option<SimpleRange<i64>> {
//     if let Some(person2_retirement_age_if_in_future_else_null) =
//         if_person2_person2_retirement_age_if_in_future_else_null
//     {
//         if person1_retirement_age_if_in_future_else_null.is_none()
//             && person2_retirement_age_if_in_future_else_null.is_none()
//         {
//             None
//         } else {
//             let start = 0;
//             let end = person1_retirement_age_if_in_future_else_null
//                 .unwrap_or(0)
//                 .max(person2_retirement_age_if_in_future_else_null.unwrap_or(0))
//                 - 1;
//             Some(SimpleRange { start, end })
//         }
//     } else {
//         if let Some(person1_retirement_age_in_future) =
//             person1_retirement_age_if_in_future_else_null
//         {
//             Some(SimpleRange {
//                 start: 0,
//                 end: person1_retirement_age_in_future - 1,
//             })
//         } else {
//             None
//         }
//     }
// }

// fn get_valid_month_range_for_income_during_retirement_as_mfn(
//     person1_retirement_age_if_in_future_else_null: Option<i64>,
//     if_person2_person2_retirement_age_if_in_future_else_null: Option<Option<i64>>,
//     last_month_as_mfn: i64,
// ) -> SimpleRange<i64> {
//     SimpleRange {
//         start: if let Some(person2_retirement_age_if_in_future_else_null) =
//             if_person2_person2_retirement_age_if_in_future_else_null
//         {
//             person1_retirement_age_if_in_future_else_null
//                 .unwrap_or(0)
//                 .min(person2_retirement_age_if_in_future_else_null.unwrap_or(0))
//         } else {
//             person1_retirement_age_if_in_future_else_null.unwrap_or(0)
//         },
//         end: last_month_as_mfn,
//     }
// }

// fn normalize_ages(orig: &People, now_as_calendar_month: &CalendarMonth) -> NormalizedAges {
//     let (person1, person2, withdrawal_start_at_retirement_of) = match orig {
//         People::WithoutPartner { person1, .. } => (
//             normalize_ages_for_person(&person1.ages, now_as_calendar_month),
//             None,
//             &PersonType::Person1,
//         ),
//         People::WithPartner {
//             person1,
//             person2,
//             withdrawal_start,
//             ..
//         } => (
//             normalize_ages_for_person(&person1.ages, now_as_calendar_month),
//             Some(normalize_ages_for_person(
//                 &person2.ages,
//                 now_as_calendar_month,
//             )),
//             withdrawal_start,
//         ),
//     };

//     let simulation_months =
//         get_simulation_months(&person1, &person2, withdrawal_start_at_retirement_of);
//     let valid_month_ranges = {
//         let if_person2_person2_retirement_age_if_in_future_else_null = person2
//             .as_ref()
//             .map(|x| x.retirement.age_as_mfn_if_in_future_else_null);

//         let future_savings_as_mfn = get_valid_month_range_for_future_savings_as_mfn(
//             person1.retirement.age_as_mfn_if_in_future_else_null,
//             if_person2_person2_retirement_age_if_in_future_else_null,
//         );

//         let income_during_retirement_as_mfn =
//             get_valid_month_range_for_income_during_retirement_as_mfn(
//                 person1.retirement.age_as_mfn_if_in_future_else_null,
//                 if_person2_person2_retirement_age_if_in_future_else_null,
//                 simulation_months.last_month_as_mfn,
//             );

//         let extra_spending = SimpleRange {
//             start: 0,
//             end: simulation_months.last_month_as_mfn,
//         };
//         NormalizedAges_ValidMonthRanges {
//             future_savings_as_mfn,
//             income_during_retirement_as_mfn,
//             extra_spending,
//         }
//     };
//     NormalizedAges {
//         person1,
//         person2,
//         simulation_months,
//         valid_month_ranges,
//     }
// }

// fn get_normalize_month<'a>(
//     person1: &'a NormalizedAges_Person,
//     person2: &'a Option<NormalizedAges_Person>,
//     now_as_calendar_month: &'a CalendarMonth,
// ) -> Box<dyn Fn(&Month) -> i64 + 'a> {
//     Box::new(move |month: &Month| match month {
//         Month::CalendarMonthAsNow { month_of_entry } => {
//             month_of_entry.minus_in_months(now_as_calendar_month)
//         }
//         Month::CalendarMonth { calendar_month } => {
//             calendar_month.minus_in_months(now_as_calendar_month)
//         }
//         Month::NamedAge { person, age } => {
//             let person = match person {
//                 PersonType::Person1 => &person1,
//                 PersonType::Person2 => person2.as_ref().unwrap(),
//             };
//             match age {
//                 super::NamedAgeType::LastWorkingMonth => {
//                     person.retirement.age_as_mfn_if_specified_else_null.unwrap() - 1
//                 }
//                 super::NamedAgeType::Retirement => {
//                     person.retirement.age_as_mfn_if_specified_else_null.unwrap()
//                 }
//                 super::NamedAgeType::Max => person.max_age_as_mfn,
//             }
//         }
//         Month::NumericAge { person, age } => {
//             let person = match person {
//                 PersonType::Person1 => &person1,
//                 PersonType::Person2 => person2.as_ref().unwrap(),
//             };
//             person.month_of_birth_as_mfn + age.in_months
//         }
//     })
// }

// fn get_normalize_month_range<'a>(
//     normalize_month: &'a dyn Fn(&Month) -> i64,
// ) -> Box<dyn Fn(&MonthRange) -> SimpleRange<i64> + 'a> {
//     Box::new(move |month_range: &MonthRange| match month_range {
//         MonthRange::StartAndEnd { start, end } => SimpleRange {
//             start: normalize_month(start),
//             end: normalize_month(end),
//         },
//         MonthRange::StartAndNumMonths { start, num_months } => SimpleRange {
//             start: normalize_month(start),
//             end: normalize_month(start) + num_months - 1,
//         },
//         MonthRange::EndAndNumMonths { end, num_months } => SimpleRange {
//             start: normalize_month(end) - (num_months - 1),
//             end: normalize_month(end),
//         },
//     })
// }

// #[cfg(test)]
// mod test_get_ages_for_person {
//     use super::normalize_ages_for_person;
//     use crate::{calendar_month::CalendarMonth, plan_params::InMonths};
//     use rstest::rstest;

//     const NOW_AS_CALENDAR_MONTH: CalendarMonth = CalendarMonth {
//         year: 2021,
//         month: 1,
//     };

//     #[rstest]
//     fn test_get_ages_for_person_retired_with_no_date() {
//         let ages = normalize_ages_for_person(
//             &super::super::Ages::RetiredWithNoRetirementDateSpecified {
//                 month_of_birth: NOW_AS_CALENDAR_MONTH.add_years(-10),
//                 max_age: InMonths {
//                     in_months: 20 * 12 + 3,
//                 },
//             },
//             &NOW_AS_CALENDAR_MONTH,
//         );
//         assert_eq!(ages.month_of_birth_as_mfn, -10 * 12);
//         assert_eq!(ages.max_age_as_mfn, 20 * 12 + 3 - 10 * 12);
//         assert_eq!(ages.retirement.age_as_mfn_if_in_future_else_null, None);
//         assert_eq!(ages.retirement.age_as_mfn_if_specified_else_null, None);
//         assert_eq!(ages.retirement.is_retired, true);
//     }

//     #[rstest]
//     #[case(10 * 12 + 1, false)]
//     #[case(10 * 12 , true)]
//     #[case(10 * 12 -1, true)]
//     fn test_get_ages_for_person_retired_with_date(
//         #[case] retirement_age: i64,
//         #[case] is_retired: bool,
//     ) {
//         let ages = normalize_ages_for_person(
//             &super::super::Ages::RetirementDateSpecified {
//                 month_of_birth: NOW_AS_CALENDAR_MONTH.add_years(-10),
//                 retirement_age: InMonths {
//                     in_months: retirement_age,
//                 },
//                 max_age: InMonths {
//                     in_months: 20 * 12 + 3,
//                 },
//             },
//             &NOW_AS_CALENDAR_MONTH,
//         );
//         assert_eq!(ages.month_of_birth_as_mfn, -10 * 12);
//         assert_eq!(ages.max_age_as_mfn, 20 * 12 + 3 - 10 * 12);
//         let as_mfn = retirement_age - 10 * 12;
//         assert_eq!(
//             ages.retirement.age_as_mfn_if_specified_else_null,
//             Some(as_mfn)
//         );
//         assert_eq!(
//             ages.retirement.age_as_mfn_if_in_future_else_null,
//             if as_mfn > 0 { Some(as_mfn) } else { None }
//         );
//         assert_eq!(ages.retirement.is_retired, is_retired);
//     }
// }

// #[cfg(test)]
// mod tests {
//     use rstest::rstest;

//     use crate::shared_types::SimpleRange;

//     #[rstest]
//     #[case(None, None)]
//     #[case(Some(1), Some(SimpleRange { start: 0, end: 0 }))]
//     #[case(Some(10), Some(SimpleRange { start: 0, end: 9 }))]
//     fn test_get_valid_month_range_for_future_savings_as_mfn_no_partner(
//         #[case] retirement_age_if_in_future_else_null: Option<i64>,
//         #[case] expected: Option<super::SimpleRange<i64>>,
//     ) {
//         let result = super::get_valid_month_range_for_future_savings_as_mfn(
//             retirement_age_if_in_future_else_null,
//             None,
//         );
//         assert_eq!(result, expected);
//     }

//     #[rstest]
//     #[case(None, None, None)]
//     #[case(None, Some(1), Some(SimpleRange { start: 0, end: 0 }))]
//     #[case(Some(1), None, Some(SimpleRange { start: 0, end: 0 }))]
//     #[case(Some(1), Some(2), Some(SimpleRange { start: 0, end: 1 }))]
//     fn test_get_valid_month_range_for_future_savings_as_mfn_with_partner(
//         #[case] person1_retirement_age_if_in_future_else_null: Option<i64>,
//         #[case] person2_retirement_age_if_in_future_else_null: Option<i64>,
//         #[case] expected: Option<super::SimpleRange<i64>>,
//     ) {
//         let result = super::get_valid_month_range_for_future_savings_as_mfn(
//             person1_retirement_age_if_in_future_else_null,
//             Some(person2_retirement_age_if_in_future_else_null),
//         );
//         assert_eq!(result, expected);
//     }

//     #[rstest]
//     #[case(None, 0)]
//     #[case(Some(1), 1)]
//     fn test_get_valid_month_range_for_income_during_retirement_as_mfn_no_partner(
//         #[case] retirement_age_as_mfn_if_in_future_else_null: Option<i64>,
//         #[case] start: i64,
//     ) {
//         let result = super::get_valid_month_range_for_income_during_retirement_as_mfn(
//             retirement_age_as_mfn_if_in_future_else_null,
//             None,
//             10,
//         );
//         assert_eq!(result, super::SimpleRange { start, end: 10 });
//     }

//     #[rstest]
//     #[case(None, None, 0)]
//     #[case(None, Some(1), 0)]
//     #[case(Some(1), None, 0)]
//     #[case(Some(1), Some(2), 1)]
//     fn test_get_valid_month_range_for_income_during_retirement_as_mfn_with_partner(
//         #[case] person1_retirement_age_as_mfn_if_in_future_else_null: Option<i64>,
//         #[case] person2_retirement_age_as_mfn_if_in_future_else_null: Option<i64>,
//         #[case] start: i64,
//     ) {
//         let result = super::get_valid_month_range_for_income_during_retirement_as_mfn(
//             person1_retirement_age_as_mfn_if_in_future_else_null,
//             Some(person2_retirement_age_as_mfn_if_in_future_else_null),
//             10,
//         );
//         assert_eq!(result, super::SimpleRange { start, end: 10 });
//     }
// }
