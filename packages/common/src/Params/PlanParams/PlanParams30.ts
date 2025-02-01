import {
  boolean,
  bounded,
  chain,
  constant,
  failure,
  gt,
  gte,
  integer,
  JSONGuard,
  JSONGuardResult,
  lte,
  nullable,
  number,
  object,
  string,
  success,
  trimmed,
  union,
} from 'json-guard'
import _ from 'lodash'
import { Guards } from '../../Guards'
import {
  assert,
  block,
  fGet,
  letIn,
  noCase,
  PickType,
  preciseRange,
} from '../../Utils'
import { PlanParams27 as V27 } from './Old/PlanParams27'
import { PlanParams28 as V28 } from './Old/PlanParams28'
import { PlanParams29 as PlanParamsPrev } from './Old/PlanParams29'

export namespace PlanParams30 {
  // Just to re-emphasize that currentVersion has the const type.
  export const currentVersion = 30 as number as 30

  export const CONSTANTS = block(() => {
    const labeledAmountTimedLocations = [
      'futureSavings',
      'incomeDuringRetirement',
      'extraSpendingEssential',
      'extraSpendingDiscretionary',
    ] as const
    const glidePathLocations = ['spawAndSWRStockAllocation'] as const
    const labeledAmountUntimedLocations = ['legacyExternalSources'] as const
    const labeledAmountTimedOrUntimedLocation = [
      ...labeledAmountTimedLocations,
      ...labeledAmountUntimedLocations,
    ]
    const monthLocations = [
      ...labeledAmountTimedLocations,
      ...glidePathLocations,
    ]
    return {
      dialogPositionOrder: V27.CONSTANTS.dialogPositionOrder,
      maxLabelLength: V27.MAX_LABEL_LENGTH,
      maxSizeForGlidePathIntermediateArray:
        V27.MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY,
      maxSizeForLabeledEntriesArr: V27.MAX_SIZE_FOR_MONTH_RANGE_ARR,
      // FEATURE: Organize the rest of the constants like this.
      people: {
        ages: {
          person: {
            currentAgeInfo: {
              datedPlan: {
                earliestYearOfBirth: V28.CONSTANTS.earliestYearOfBirth,
              },
              undatedPlan: {
                maxCurrentAge: 12 * 115,
              },
            },
            // Duplicated in Rust.
            maxAge: V27.MAX_AGE_IN_MONTHS,
          },
        },
      },
      risk: {
        tpaw: {
          riskTolerance: {
            values: block(() => {
              const numSegments = 5
              const countPerSegment = 5
              const numPoints = numSegments * countPerSegment
              const startRRA = 16
              const endRRA = 0.5

              const segmentDef = (segment: number, label: string) => {
                const startIndex = segment * numSegments
                const endIndex = startIndex + countPerSegment - 1 // end is inclusive
                const count = countPerSegment
                const containsIndex = (index: number) =>
                  index >= startIndex && index <= endIndex
                return { startIndex, endIndex, containsIndex, label, count }
              }

              const segments = [
                segmentDef(0, 'Very Conservative'),
                segmentDef(1, 'Conservative'),
                segmentDef(2, 'Moderate'),
                segmentDef(3, 'Aggressive'),
                segmentDef(4, 'Very Aggressive'),
              ]

              return {
                numIntegerValuesStartingFrom0: numPoints,
                startRRA,
                endRRA,
                segments,
                // TODO: Remove
                riskToleranceToRRA: block(() => {
                  const log1OverRRA = (rra: number) => Math.log(1 / rra)

                  const shift = log1OverRRA(startRRA)
                  const scale =
                    (numPoints - 2) /
                    (log1OverRRA(endRRA) - log1OverRRA(startRRA))

                  const withoutInfinityAtZero = (riskTolerance: number) =>
                    1 / Math.exp((riskTolerance - 1) / scale + shift)
                  const withInfinityAtZero = (riskTolerance: number) =>
                    riskTolerance === 0
                      ? Infinity
                      : withoutInfinityAtZero(riskTolerance)
                  withoutInfinityAtZero.inverse = (rra: number) =>
                    (log1OverRRA(rra) - shift) * scale + 1
                  return { withInfinityAtZero, withoutInfinityAtZero }
                }),
              }
            }),
          },
          timePreference: {
            values: V27.TIME_PREFERENCE_VALUES,
          },
          additionalAnnualSpendingTilt: {
            values: V27.ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES,
          },
        },
        spaw: {
          annualSpendingTilt: {
            values: V27.SPAW_ANNUAL_SPENDING_TILT_VALUES,
          },
        },
      },
      advanced: {
        returnsStatsForPlanning: {
          expectedValue: {
            fixedEquityPremium: { values: preciseRange(0, 0.07, 0.001, 3) },
            custom: {
              stocks: {
                base: {
                  values: [
                    'regressionPrediction',
                    'conservativeEstimate',
                    '1/CAPE',
                    'historical',
                  ] as const,
                },
              },
              bonds: {
                base: { values: ['20YearTIPSYield', 'historical'] as const },
              },
              deltaValues: preciseRange(-0.05, 0.05, 0.001, 3),
            },
            fixed: {
              values: V27.MANUAL_STOCKS_BONDS_NON_LOG_ANNUAL_RETURNS_VALUES,
            },
          },
          standardDeviation: {
            stocks: {
              scale: {
                log: { values: V27.STOCK_VOLATILITY_SCALE_VALUES },
              },
            },
          },
        },
        inflation: {
          manual: {
            values: V27.MANUAL_INFLATION_VALUES,
          },
        },
        historicalReturnsAdjustment: {
          standardDeviation: {
            bonds: {
              scale: { log: { values: preciseRange(0, 2, 0.01, 2) } },
            },
          },
        },
      },
      // Duplicated in Rust.
      minPlanParamTime: V27.MIN_PLAN_PARAM_TIME,
      labeledAmountTimedLocations,
      labeledAmountUntimedLocations,
      labeledAmountTimedOrUntimedLocation,
      glidePathLocations,
      monthLocations,
    }
  })

  export type DialogPosition = V27.DialogPosition

  export type LabeledAmountTimedLocation =
    (typeof CONSTANTS.labeledAmountTimedLocations)[number]
  export type LabeledAmountUntimedLocation =
    (typeof CONSTANTS.labeledAmountUntimedLocations)[number]
  export type LabeledAmountTimedOrUntimedLocation =
    | LabeledAmountUntimedLocation
    | LabeledAmountTimedLocation
  export type GlidePathLocation = (typeof CONSTANTS.glidePathLocations)[number]
  export type MonthLocation = LabeledAmountTimedLocation | GlidePathLocation

  export type CalendarMonth = {
    year: number
    month: number // 1 - 12
  }
  export type CalendarDay = {
    year: number
    month: number // 1 - 12
    day: number // 1 - 31
  }

  export type InMonths = { inMonths: number }
  export type CurrentAgeInfo =
    | { isDatedPlan: true; monthOfBirth: CalendarMonth }
    | { isDatedPlan: false; currentAge: InMonths }

  export type PersonId = 'person1' | 'person2'
  export type Person = {
    ages:
      | {
          type: 'retiredWithNoRetirementDateSpecified'
          currentAgeInfo: CurrentAgeInfo
          maxAge: InMonths
        }
      | {
          type: 'retirementDateSpecified'
          currentAgeInfo: CurrentAgeInfo
          retirementAge: InMonths
          maxAge: InMonths
        }
  }

  export type People =
    | { withPartner: false; person1: Person }
    | {
        withPartner: true
        person2: Person
        person1: Person
        withdrawalStart: 'person1' | 'person2'
      }

  export type Month =
    | {
        type: 'now'
        monthOfEntry:
          | { isDatedPlan: true; calendarMonth: CalendarMonth }
          | { isDatedPlan: false }
      }
    | { type: 'calendarMonth'; calendarMonth: CalendarMonth }
    | {
        type: 'namedAge'
        person: 'person1' | 'person2'
        age: 'lastWorkingMonth' | 'retirement' | 'max'
      }
    | {
        type: 'numericAge'
        person: 'person1' | 'person2'
        age: InMonths
      }

  export type MonthRange =
    | { type: 'startAndEnd'; start: Month; end: Month | { type: 'inThePast' } }
    | { type: 'startAndDuration'; start: Month; duration: InMonths }
    | { type: 'endAndDuration'; end: Month; duration: InMonths }

  export type LabeledAmountTimed = {
    label: string | null
    nominal: boolean
    id: string
    sortIndex: number
    colorIndex: number
    amountAndTiming:
      | { type: 'inThePast' }
      | {
          type: 'oneTime'
          amount: number
          month: Month
        }
      | {
          type: 'recurring'
          monthRange: MonthRange
          everyXMonths: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12
          baseAmount: number
          delta: {
            by: { type: 'percent'; percent: number }
            every: 'year' | 'recurrence'
          } | null
        }
  }

  export type LabeledAmountTimedList = Record<string, LabeledAmountTimed>

  export type LabeledAmountUntimed = {
    label: string | null
    amount: number
    nominal: boolean
    id: string
    sortIndex: number
    colorIndex: number
  }
  export type LabeledAmountUntimedList = Record<string, LabeledAmountUntimed>

  export type LabeledAmountTimedOrUntimed =
    | LabeledAmountTimed
    | LabeledAmountUntimed

  export type GlidePath = {
    start: {
      month: PickType<Month, 'now'>
      stocks: number
    }
    intermediate: Record<
      string,
      { id: string; month: Month; indexToSortByAdded: number; stocks: number }
    >
    end: { stocks: number }
  }

  type ExpectedReturnsForPlanning =
    | { type: 'regressionPrediction,20YearTIPSYield' }
    | { type: 'conservativeEstimate,20YearTIPSYield' }
    | { type: '1/CAPE,20YearTIPSYield' }
    | { type: 'historical' }
    | { type: 'fixedEquityPremium'; equityPremium: number }
    | {
        type: 'custom'
        stocks: {
          base: (typeof CONSTANTS.advanced.returnsStatsForPlanning.expectedValue.custom.stocks.base.values)[number]
          delta: number
        }
        bonds: {
          base: (typeof CONSTANTS.advanced.returnsStatsForPlanning.expectedValue.custom.bonds.base.values)[number]
          delta: number
        }
      }
    | { type: 'fixed'; stocks: number; bonds: number }

  type SamplingMonteCarloData = {
    blockSize: InMonths
    staggerRunStarts: boolean
  }

  type SamplingDefaultData = {
    monteCarlo: SamplingMonteCarloData | null
  }

  type Sampling =
    | { type: 'monteCarlo'; data: SamplingMonteCarloData }
    | { type: 'historical'; defaultData: SamplingDefaultData }

  export type PlanParams = {
    v: typeof currentVersion
    // TODO: Record tz info.
    timestamp: number
    datingInfo:
      | { isDated: true }
      | {
          isDated: false
          marketDataAsOfEndOfDayInNY: CalendarDay
        }

    // Note 1. Technically should be in non-plan, but this moves with plan
    // changes, so simpler to keep it here.
    // Note 2. "nominal" because 'future-savings' does not apply if both
    // people are retired. In that case effective will convert 'future-savings'
    // to 'income-during-retirement'. Additional Note: It is not possible
    // to enforce not 'future-savings' if both retired at the guard level,
    // because retired is a property that needs an evaluation time to determine,
    // which is not available here, so we really need to allow 'future-savings'
    // in all cases in the params and deal with converting nominal to effective
    // at evaluation time.
    dialogPositionNominal: DialogPosition
    people: People
    // Wealth
    wealth: {
      portfolioBalance:
        | { isDatedPlan: false; amount: number }
        | { isDatedPlan: true; updatedHere: true; amount: number }
        | {
            isDatedPlan: true
            updatedHere: false
            updatedAtId: string
            updatedTo: number
            updatedAtTimestamp: number
          }
      futureSavings: LabeledAmountTimedList
      incomeDuringRetirement: LabeledAmountTimedList
    }

    // Adjustments to Spending
    adjustmentsToSpending: {
      extraSpending: {
        essential: LabeledAmountTimedList
        discretionary: LabeledAmountTimedList
      }
      tpawAndSPAW: {
        monthlySpendingCeiling: number | null
        monthlySpendingFloor: number | null
        legacy: {
          total: number
          external: LabeledAmountUntimedList
        }
      }
    }

    // Risk
    risk: {
      tpaw: {
        riskTolerance: {
          at20: number
          deltaAtMaxAge: number
          forLegacyAsDeltaFromAt20: number
        }
        timePreference: number
        additionalAnnualSpendingTilt: number
      }
      tpawAndSPAW: {
        lmp: number
      }
      spaw: {
        annualSpendingTilt: number
      }
      spawAndSWR: {
        allocation: GlidePath
      }
      swr: {
        withdrawal:
          | { type: 'asPercentPerYear'; percentPerYear: number }
          | { type: 'asAmountPerMonth'; amountPerMonth: number }
          | { type: 'default' } // Only when strategy !== 'SWR'
      }
    }

    // Advanced.
    advanced: {
      returnsStatsForPlanning: {
        expectedValue: { empiricalAnnualNonLog: ExpectedReturnsForPlanning }
        standardDeviation: {
          stocks: { scale: { log: number } }
          // Emphasising that for planning bonds are assumed to not have
          // volatility.
          bonds: { scale: { log: 0 } }
        }
      }

      // Historical returns are adjusted to match returnsStatsForPlanning except
      // as below:
      historicalReturnsAdjustment: {
        standardDeviation: {
          bonds: {
            scale: { log: number }
          }
        }
        overrideToFixedForTesting:
          | { type: 'none' }
          | { type: 'useExpectedReturnsForPlanning' }
          | { type: 'manual'; stocks: number; bonds: number }
      }

      sampling: Sampling
      annualInflation: { type: 'suggested' } | { type: 'manual'; value: number }
      strategy: 'TPAW' | 'SPAW' | 'SWR'
    }
    results: {
      displayedAssetAllocation: { stocks: number }
    } | null
  }

  // ----------- GUARD  ---------//

  const { uuid, notEmpty } = Guards

  const maxPrecision = (precision: number) => (x: number) => {
    const split = x.toString().split('.')
    return split.length === 1
      ? success(x)
      : fGet(split[1]).length <= precision
        ? success(x)
        : failure(`Max precision is ${precision}.`)
  }

  const among =
    <T>(values: T[]): JSONGuard<T> =>
    (x: unknown) => {
      return values.includes(x as T)
        ? success(x as T)
        : failure('Not among predefined value.')
    }

  const smallId = chain(string, (x) =>
    x.length !== 10 ? failure('Does not have length === 10') : success(x),
  )
  const arrAsObj =
    <T extends { id: string }>(
      valueGuard: JSONGuard<T>,
      maxLen: number,
    ): JSONGuard<Record<string, T>> =>
    (x: unknown) => {
      if (!_.isObject(x)) return failure('Not an object.')
      const keyValues = _.toPairs(x)
      if (keyValues.length > maxLen)
        return failure(`Too many values, max is ${maxLen}.`)

      const asArr = [] as [string, T][]
      for (const [key, value] of keyValues) {
        const check = valueGuard(value)
        if (check.error)
          return failure(`Invalid value for ${key}: ${check.message}`)
        asArr.push([key, check.value])
      }

      const result: Record<string, T> = {}
      for (const [key, value] of asArr) {
        if (key !== value.id)
          return failure(`Key ${key} does not match id ${value.id}`)
        result[key] = value
      }

      return success(result)
    }

  export const componentGuards = block(() => {
    const dialogPositionNominal = union(
      constant('age'),
      constant('current-portfolio-balance'),
      constant('future-savings'),
      constant('income-during-retirement'),
      constant('show-results'),
      constant('show-all-inputs'),
      constant('done'),
    )

    const personId: JSONGuard<PersonId> = union(
      constant('person1'),
      constant('person2'),
    )
    const calendarMonth = (
      planParams: PlanParams | null,
    ): JSONGuard<CalendarMonth> =>
      planParams && !planParams.datingInfo.isDated
        ? () => failure('Cannot use calendar month in undated plan.')
        : object({
            year: chain(number, integer),
            month: chain(number, integer, gte(1), lte(12)),
          })
    const calendarDay: JSONGuard<CalendarDay> = object({
      year: chain(number, integer),
      month: chain(number, integer, gte(1), lte(12)),
      day: chain(number, integer, gte(1), lte(31)),
    })

    const currentAgeInfo = (
      planParams: PlanParams | null,
    ): JSONGuard<CurrentAgeInfo> =>
      chain(
        union(
          object({
            isDatedPlan: constant(true),
            monthOfBirth: calendarMonth(planParams),
          }),
          object({ isDatedPlan: constant(false), currentAge: cg.inMonths }),
        ),
        (x) => {
          if (planParams && planParams.datingInfo.isDated !== x.isDatedPlan)
            return failure(
              'Mismatch between plan dating and current age dating.',
            )
          if (
            x.isDatedPlan &&
            x.monthOfBirth.year <
              CONSTANTS.people.ages.person.currentAgeInfo.datedPlan
                .earliestYearOfBirth
          )
            return failure(`Year of birth is too low.`)
          if (
            !x.isDatedPlan &&
            x.currentAge.inMonths >
              CONSTANTS.people.ages.person.currentAgeInfo.undatedPlan
                .maxCurrentAge
          )
            return failure('Current age is too high.')
          return success(x)
        },
      )
    const inMonths: JSONGuard<InMonths> = object({
      inMonths: chain(number, integer),
    })
    const strategy: JSONGuard<PlanParams['advanced']['strategy']> = union(
      constant('TPAW'),
      constant('SPAW'),
      constant('SWR'),
    )
    const nowMonth = (
      planParams: PlanParams | null,
    ): JSONGuard<PickType<Month, 'now'>> =>
      object({
        type: constant('now'),
        monthOfEntry: chain(
          union(
            object({
              isDatedPlan: constant(true),
              calendarMonth: calendarMonth(planParams),
            }),
            object({ isDatedPlan: constant(false) }),
          ),
          (x) => {
            if (planParams && planParams.datingInfo.isDated !== x.isDatedPlan)
              return failure('Mismatch between plan dating and month dating.')
            return success(x)
          },
        ),
      })

    const month = (planParams: PlanParams | null): JSONGuard<Month> =>
      union(
        nowMonth(planParams),
        object({
          type: constant('calendarMonth'),
          calendarMonth: calendarMonth(planParams),
        }),
        chain(
          object({
            type: constant('namedAge'),
            person: personId,
            age: union(
              constant('lastWorkingMonth'),
              constant('retirement'),
              constant('max'),
            ),
          }),
          (x) => {
            if (!planParams) return success(x)
            const { people } = planParams
            let person: Person
            if (x.person === 'person1') {
              person = people.person1
            } else {
              if (!people.withPartner)
                return failure('In terms of partner, but there is no partner.')
              person = people.person2
            }

            if (
              (x.age === 'retirement' || x.age === 'lastWorkingMonth') &&
              person.ages.type === 'retiredWithNoRetirementDateSpecified'
            ) {
              return failure(
                `In terms retirement age of ${x.person}, but ${x.person} does not have a retirement age specified.`,
              )
            }
            return success(x)
          },
        ),
        chain(
          object({
            type: constant('numericAge'),
            person: personId,
            age: inMonths,
          }),
          (x) => {
            if (!planParams) return success(x)
            const { people } = planParams
            if (x.person === 'person2' && !people.withPartner)
              return failure('In terms of partner, but there is no partner.')
            return success(x)
          },
        ),
      )

    const monthRangeDuration = object({
      inMonths: chain(number, integer, gt(0)),
    })

    const monthRange = (
      planParams: PlanParams | null,
    ): JSONGuard<MonthRange> => {
      const _month = month(planParams)
      return chain(
        union(
          object({
            type: constant('startAndEnd'),
            start: _month,
            end: union(_month, object({ type: constant('inThePast') })),
          }),
          object({
            type: constant('startAndDuration'),
            start: _month,
            duration: monthRangeDuration,
          }),
          object({
            type: constant('endAndDuration'),
            end: _month,
            duration: monthRangeDuration,
          }),
        ),
        (x): JSONGuardResult<MonthRange> => {
          switch (x.type) {
            case 'startAndEnd':
            case 'startAndDuration':
              if (
                x.start.type === 'namedAge' &&
                x.start.age === 'lastWorkingMonth'
              ) {
                // Range cannot start with lastWorkingMonth because if person
                // switches to retired, then lastWorkingMonth will be in the
                // past, which is a problem for "startAndNumMonths" because we
                // can't determine the end of the range. But we enforce this for
                // all types of ranges for simplicity. Note, this is not a
                // problem for the end of the range, because "endAndNumMonths",
                // will then be entirely in the past and we don't not have to
                // determine the range.
                return failure('Cannot use lastWorkingMonth as start of range.')
              }
              return success(x)
            case 'endAndDuration':
              return success(x)
            default:
              noCase(x)
          }
        },
      )
    }

    const labeledAmountUntimed: JSONGuard<LabeledAmountUntimed> = object({
      label: nullable(
        chain(string, trimmed, notEmpty, bounded(CONSTANTS.maxLabelLength)),
      ),
      amount: chain(number, gte(0)),
      nominal: boolean,
      id: smallId,
      sortIndex: chain(number, integer),
      colorIndex: chain(number, integer, gte(0)),
    })

    const amountAndTiming = (
      planParams: PlanParams | null,
    ): JSONGuard<LabeledAmountTimed['amountAndTiming']> =>
      union(
        object({
          type: constant('inThePast'),
        }),
        object({
          type: constant('oneTime'),
          month: month(planParams),
          amount: chain(number, integer, gte(0)),
        }),
        object({
          type: constant('recurring'),
          monthRange: monthRange(planParams),
          everyXMonths: among([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const),
          baseAmount: chain(number, integer, gte(0)),
          delta: nullable(
            object({
              by: object({
                type: constant('percent'),
                // TODO
                percent: (x: unknown): JSONGuardResult<number> =>
                  failure('Not Implemented'),
              }),
              every: union(constant('year'), constant('recurrence')),
            }),
          ),
        }),
      )

    const labeledAmountTimed = (
      planParams: PlanParams | null,
    ): JSONGuard<LabeledAmountTimed> =>
      object({
        label: nullable(
          chain(string, trimmed, notEmpty, bounded(CONSTANTS.maxLabelLength)),
        ),
        nominal: boolean,
        id: smallId,
        sortIndex: chain(number, integer),
        colorIndex: chain(number, integer, gte(0)),
        amountAndTiming: amountAndTiming(planParams),
      })

    const labeledAmountTimedList = (
      params: PlanParams | null = null,
    ): JSONGuard<LabeledAmountTimedList> =>
      arrAsObj(
        labeledAmountTimed(params),
        CONSTANTS.maxSizeForLabeledEntriesArr,
      )

    const labeledAmountUntimedList: JSONGuard<LabeledAmountUntimedList> =
      arrAsObj(labeledAmountUntimed, CONSTANTS.maxSizeForLabeledEntriesArr)

    const glidePathStocks = chain(number, gte(0), lte(1), maxPrecision(2))
    const glidePath = (planParams: PlanParams | null): JSONGuard<GlidePath> =>
      object({
        start: object({
          month: nowMonth(planParams),
          stocks: glidePathStocks,
        }),
        intermediate: arrAsObj(
          object({
            id: smallId,
            month: month(planParams),
            stocks: glidePathStocks,
            indexToSortByAdded: chain(number, integer),
          }),
          CONSTANTS.maxSizeForGlidePathIntermediateArray,
        ),
        end: object({ stocks: glidePathStocks }),
      })

    const samplingType = union(constant('monteCarlo'), constant('historical'))

    const expectedReturnsForPlanning: JSONGuard<
      PlanParams['advanced']['returnsStatsForPlanning']['expectedValue']['empiricalAnnualNonLog']
    > = union(
      object({
        type: constant('regressionPrediction,20YearTIPSYield'),
      }),
      object({
        type: constant('conservativeEstimate,20YearTIPSYield'),
      }),
      object({ type: constant('1/CAPE,20YearTIPSYield') }),
      object({ type: constant('historical') }),
      object({
        type: constant('fixedEquityPremium'),
        equityPremium: among(
          CONSTANTS.advanced.returnsStatsForPlanning.expectedValue
            .fixedEquityPremium.values,
        ),
      }),
      object({
        type: constant('custom'),
        stocks: object({
          base: union(
            constant('regressionPrediction'),
            constant('conservativeEstimate'),
            constant('1/CAPE'),
            constant('historical'),
          ),
          delta: among(
            CONSTANTS.advanced.returnsStatsForPlanning.expectedValue.custom
              .deltaValues,
          ),
        }),
        bonds: object({
          base: union(constant('20YearTIPSYield'), constant('historical')),
          delta: among(
            CONSTANTS.advanced.returnsStatsForPlanning.expectedValue.custom
              .deltaValues,
          ),
        }),
      }),
      object({
        type: constant('fixed'),
        stocks: among(
          CONSTANTS.advanced.returnsStatsForPlanning.expectedValue.fixed.values,
        ),
        bonds: among(
          CONSTANTS.advanced.returnsStatsForPlanning.expectedValue.fixed.values,
        ),
      }),
    )

    const annualInflation: JSONGuard<
      PlanParams['advanced']['annualInflation']
    > = union(
      object({ type: constant('suggested') }),
      object({
        type: constant('manual'),
        value: among(CONSTANTS.advanced.inflation.manual.values),
      }),
    )
    const labeledAmountTimedLocation: JSONGuard<LabeledAmountTimedLocation> =
      union(
        constant('futureSavings'),
        constant('incomeDuringRetirement'),
        constant('extraSpendingEssential'),
        constant('extraSpendingDiscretionary'),
      )
    const labeledAmountUntimedLocation: JSONGuard<LabeledAmountUntimedLocation> =
      constant('legacyExternalSources')
    const labeledAmountTimedOrUntimedLocation: JSONGuard<
      LabeledAmountUntimedLocation | LabeledAmountTimedLocation
    > = union(labeledAmountTimedLocation, constant('legacyExternalSources'))
    const glidePathLocation: JSONGuard<GlidePathLocation> = constant(
      'spawAndSWRStockAllocation',
    )
    const historicalReturnsAdjustment = {
      overrideToFixedForTesting: union(
        object({ type: constant('none') }),
        object({ type: constant('useExpectedReturnsForPlanning') }),
        object({ type: constant('manual'), stocks: number, bonds: number }),
      ),
    }

    return {
      calendarMonth,
      calendarDay,
      currentAgeInfo,
      personId,
      strategy,
      inMonths,
      month,
      monthRange,
      monthRangeDuration,
      dialogPositionNominal,
      glidePath,
      samplingType,
      expectedReturnsForPlanning,
      annualInflation,
      labeledAmountUntimed,
      labeledAmountUntimedList,
      amountAndTiming,
      labeledAmountTimed,
      labeledAmountTimedList,
      labeledAmountTimedLocation,
      labeledAmountTimedOrUntimedLocation,
      labeledAmountUntimedLocation,
      glidePathLocation,
      historicalReturnsAdjustment,
    }
  })
  const cg = componentGuards

  const _ages = (planParams: PlanParams | null): JSONGuard<Person['ages']> =>
    chain(
      union(
        object({
          type: constant('retiredWithNoRetirementDateSpecified'),
          currentAgeInfo: cg.currentAgeInfo(planParams),
          maxAge: cg.inMonths,
        }),
        object({
          type: constant('retirementDateSpecified'),
          currentAgeInfo: cg.currentAgeInfo(planParams),
          retirementAge: cg.inMonths,
          maxAge: cg.inMonths,
        }),
      ),
      (ages: Person['ages']): JSONGuardResult<Person['ages']> => {
        const { maxAge } = ages

        if (maxAge.inMonths > CONSTANTS.people.ages.person.maxAge)
          return failure(`Max age is too large.`)

        // Two months because asset allocation graph may not show last month,
        // and needs at least 2 other months, so total of 3 (note the range is
        // inclusive, so +2 allows for 3 months.)
        if (maxAge.inMonths <= 3)
          return failure(
            'Max age should be at least two months after birth month.',
          )

        if (ages.type === 'retirementDateSpecified') {
          const { retirementAge } = ages
          if (retirementAge.inMonths < 1) {
            return failure(
              'Retirement age should be at least one month after birth month.',
            )
          }
          if (maxAge.inMonths < retirementAge.inMonths + 1) {
            return failure(
              'Max age should be at least one month after retirement age.',
            )
          }
        }
        return success(ages)
      },
    )

  const person = (planParams: PlanParams | null): JSONGuard<Person> =>
    object({
      ages: _ages(planParams),
    })

  const people = (
    planParams: PlanParams | null,
  ): JSONGuard<PlanParams['people']> =>
    union(
      object({
        withPartner: constant(false),
        person1: person(planParams),
      }),
      object({
        withPartner: constant(true),
        person2: person(planParams),
        person1: person(planParams),
        withdrawalStart: cg.personId,
      }),
    )

  const wealth = (params: PlanParams | null): JSONGuard<PlanParams['wealth']> =>
    chain(
      object({
        portfolioBalance: union(
          object({
            isDatedPlan: constant(false),
            amount: chain(number, integer, gte(0)),
          }),
          object({
            isDatedPlan: constant(true),
            updatedHere: constant(true),
            amount: chain(number, integer, gte(0)),
          }),
          object({
            isDatedPlan: constant(true),
            updatedHere: constant(false),
            updatedAtId: uuid,
            updatedTo: chain(number, gte(0)),
            updatedAtTimestamp: chain(number, integer, gte(0)),
          }),
        ),
        futureSavings: cg.labeledAmountTimedList(params),
        incomeDuringRetirement: cg.labeledAmountTimedList(params),
      }),
      (x) => {
        const colorIndexes = [
          ..._.values(x.futureSavings),
          ..._.values(x.incomeDuringRetirement),
        ].map((x) => x.colorIndex)
        const uniqueColorIndexes = _.uniq(colorIndexes)
        if (uniqueColorIndexes.length !== colorIndexes.length)
          return failure('Duplicate color indexes.')
        return success(x)
      },
    )

  const adjustmentsToSpending = (
    planParams: PlanParams | null,
  ): JSONGuard<PlanParams['adjustmentsToSpending']> =>
    object({
      tpawAndSPAW: object({
        monthlySpendingCeiling: chain(
          nullable(chain(number, gte(0))),
          chain(nullable(chain(number, gte(0))), (x) => {
            if (!planParams) return success(x)
            if (x === null) return success(x)
            if (
              planParams.adjustmentsToSpending.tpawAndSPAW
                .monthlySpendingFloor !== null &&
              x <
                planParams.adjustmentsToSpending.tpawAndSPAW
                  .monthlySpendingFloor
            ) {
              failure('Spending floor is greater than spending ceiling.')
            }
            return success(x)
          }),
        ),
        monthlySpendingFloor: nullable(chain(number, gte(0))),
        legacy: object({
          total: chain(number, gte(0)),
          external: cg.labeledAmountUntimedList,
        }),
      }),
      extraSpending: chain(
        object({
          essential: cg.labeledAmountTimedList(planParams),
          discretionary: cg.labeledAmountTimedList(planParams),
        }),
        (x) => {
          const colorIndexes = [
            ..._.values(x.essential),
            ..._.values(x.discretionary),
          ].map((x) => x.colorIndex)
          const uniqueColorIndexes = _.uniq(colorIndexes)
          if (uniqueColorIndexes.length !== colorIndexes.length)
            return failure('Duplicate color indexes.')
          return success(x)
        },
      ),
    })

  const risk = (
    planParams: PlanParams | null,
  ): JSONGuard<PlanParams['risk']> => {
    const riskToleranceValues = _.range(
      0,
      CONSTANTS.risk.tpaw.riskTolerance.values.numIntegerValuesStartingFrom0,
    )
    return object({
      tpaw: object({
        riskTolerance: object({
          at20: among(riskToleranceValues),
          deltaAtMaxAge: among(riskToleranceValues.map((x) => x * -1)),
          forLegacyAsDeltaFromAt20: among(riskToleranceValues),
        }),
        timePreference: among(CONSTANTS.risk.tpaw.timePreference.values),
        additionalAnnualSpendingTilt: among(
          CONSTANTS.risk.tpaw.additionalAnnualSpendingTilt.values,
        ),
      }),
      tpawAndSPAW: object({
        lmp: chain(number, gte(0)),
      }),
      spaw: object({
        annualSpendingTilt: among(
          CONSTANTS.risk.spaw.annualSpendingTilt.values,
        ),
      }),
      swr: object({
        withdrawal: chain(
          union(
            object({
              type: constant('asPercentPerYear'),
              percentPerYear: chain(number, gte(0), lte(1)),
            }),
            object({
              type: constant('asAmountPerMonth'),
              amountPerMonth: chain(number, integer, gte(0)),
            }),
            object({ type: constant('default') }),
          ),
          (x) => {
            if (planParams === null) return success(x)
            if (planParams.advanced.strategy === 'SWR' && x.type === 'default')
              return failure(
                'SWR withdrawal strategy cannot be "default" when in SWR mode.',
              )
            return success(x)
          },
        ),
      }),
      spawAndSWR: object({
        allocation: cg.glidePath(planParams),
      }),
    })
  }

  const samplingMonteCarloData: JSONGuard<SamplingMonteCarloData> = object({
    blockSize: object({
      inMonths: chain(
        number,
        integer,
        gte(1),
        lte(CONSTANTS.people.ages.person.maxAge),
      ),
    }),
    staggerRunStarts: boolean,
  })
  const samplingDefaultData: JSONGuard<SamplingDefaultData> = object({
    monteCarlo: nullable(samplingMonteCarloData),
  })
  const sampling: JSONGuard<PlanParams['advanced']['sampling']> = union(
    object({
      type: constant('monteCarlo'),
      data: samplingMonteCarloData,
    }),
    object({
      type: constant('historical'),
      defaultData: samplingDefaultData,
    }),
  )

  const advanced: JSONGuard<PlanParams['advanced']> = block(() => {
    const returnsStatsForPlanning: JSONGuard<
      PlanParams['advanced']['returnsStatsForPlanning']
    > = object({
      expectedValue: object({
        empiricalAnnualNonLog: cg.expectedReturnsForPlanning,
      }),
      standardDeviation: object({
        stocks: object({
          scale: object({
            log: among(
              CONSTANTS.advanced.returnsStatsForPlanning.standardDeviation
                .stocks.scale.log.values,
            ),
          }),
        }),
        bonds: object({ scale: object({ log: constant(0) }) }),
      }),
    })
    const historicalReturnsAdjustment: JSONGuard<
      PlanParams['advanced']['historicalReturnsAdjustment']
    > = object({
      standardDeviation: object({
        bonds: object({
          scale: object({
            log: among(
              CONSTANTS.advanced.historicalReturnsAdjustment.standardDeviation
                .bonds.scale.log.values,
            ),
          }),
        }),
      }),
      overrideToFixedForTesting:
        cg.historicalReturnsAdjustment.overrideToFixedForTesting,
    })
    return object({
      returnsStatsForPlanning,
      historicalReturnsAdjustment,
      sampling,
      annualInflation: cg.annualInflation,
      strategy: cg.strategy,
    })
  })

  const oneOrTwoPassGuard = (
    planParams: PlanParams | null,
  ): JSONGuard<PlanParams> =>
    object({
      v: constant(currentVersion),
      timestamp: number,
      datingInfo: union(
        object({ isDated: constant(true) }),
        object({
          isDated: constant(false),
          marketDataAsOfEndOfDayInNY: cg.calendarDay,
        }),
      ),
      dialogPositionNominal: cg.dialogPositionNominal,
      people: people(planParams),
      wealth: wealth(planParams),
      adjustmentsToSpending: adjustmentsToSpending(planParams),
      risk: risk(planParams),
      advanced,
      results: nullable(
        object({
          displayedAssetAllocation: object({
            stocks: chain(number, gte(0), lte(1)),
          }),
        }),
      ),
    })

  export const guard: JSONGuard<PlanParams> = chain(
    oneOrTwoPassGuard(null),
    (x) => oneOrTwoPassGuard(x)(x),
  )

  export type SomePlanParams = PlanParamsPrev.SomePlanParams | PlanParams
  export type TimestampedPlanParams =
    | PlanParamsPrev.TimestampedPlanParams
    | PlanParams

  export const backwardsCompatibleGuard: JSONGuard<SomePlanParams> = (
    x: unknown,
  ) => {
    const result = union(PlanParamsPrev.backwardsCompatibleGuard, guard)(x)
    return result.error ? result : success(x as SomePlanParams)
  }
  export const backwardsCompatibleToTimestampGuard: JSONGuard<
    TimestampedPlanParams
  > = (x: unknown) => {
    const result = union(PlanParamsPrev.backwardsCompatibleGuard, guard)(x)
    return result.error ? result : success(x as TimestampedPlanParams)
  }

  export const migrate = (x: SomePlanParams): PlanParams => {
    if ('v' in x && x.v === currentVersion) return x
    const prev = PlanParamsPrev.migrate(x)

    const result: PlanParams = {
      ...prev,
      v: currentVersion,
      wealth: {
        ...prev.wealth,
        portfolioBalance: block(() => {
          const p = prev.wealth.portfolioBalance
          return !p.isDatedPlan
            ? { ...p, amount: Math.round(p.amount) }
            : p.updatedHere
              ? {
                  ...p,
                  // This should have been rounded to start with, but some floating point has leaked into
                  // existing parameters. Rounding to deal with that.
                  amount: Math.round(p.amount),
                }
              : {
                ...p,
                updatedTo:Math.round(p.updatedTo)
              }
        }),
      },
      advanced: {
        ...prev.advanced,
        historicalReturnsAdjustment: block(() => {
          const p = prev.advanced.historicalReturnsAdjustment.standardDeviation
          return {
            standardDeviation: { bonds: p.bonds },
            overrideToFixedForTesting: p.overrideToFixedForTesting
              ? { type: 'useExpectedReturnsForPlanning' }
              : { type: 'none' },
          }
        }),
      },
    }
    const check = guard(result)
    if (check.error) throw new Error(check.message)
    return result
  }
}
