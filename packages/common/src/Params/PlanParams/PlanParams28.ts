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
import { assert, block, fGet, letIn, noCase } from '../../Utils'
import {
  PlanParams27 as PlanParamsPrev,
  PlanParams27 as V27,
} from './Old/PlanParams27'

export namespace PlanParams28 {
  export const currentVersion = 28 as const

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
      // Duplicated in Rust.
      maxAgeInMonths: V27.MAX_AGE_IN_MONTHS,
      earliestYearOfBirth: 1915,
      maxSizeForGlidePathIntermediateArray:
        V27.MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY,
      maxSizeForLabeledEntriesArr: V27.MAX_SIZE_FOR_MONTH_RANGE_ARR,
      stockVolatilityScaleValues: V27.STOCK_VOLATILITY_SCALE_VALUES,
      timePreferenceValues: V27.TIME_PREFERENCE_VALUES,
      additionalAnnualSpendingTiltValues:
        V27.ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES,
      manualInflationValues: V27.MANUAL_INFLATION_VALUES,
      manualStocksBondsNonLogAnnualReturnsValues:
        V27.MANUAL_STOCKS_BONDS_NON_LOG_ANNUAL_RETURNS_VALUES,
      spawAnnualSpendingTiltValues: V27.SPAW_ANNUAL_SPENDING_TILT_VALUES,
      riskToleranceValues: V27.RISK_TOLERANCE_VALUES,
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
  export type InMonths = { inMonths: number }

  export type Person = {
    ages:
      | {
          type: 'retiredWithNoRetirementDateSpecified'
          monthOfBirth: CalendarMonth
          maxAge: InMonths
        }
      | {
          type: 'retirementDateSpecified'
          monthOfBirth: CalendarMonth
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
    | { type: 'calendarMonthAsNow'; monthOfEntry: CalendarMonth }
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
          everyXMonths:
            | 1
            | 2
            | 3
            | 4
            | 5
            | 6
            | 7
            | 8
            | 9
            | 10
            | 11
            | 12
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
    start: { month: CalendarMonth; stocks: number }
    intermediate: Record<
      string,
      { id: string; month: Month; indexToSortByAdded: number; stocks: number }
    >
    end: { stocks: number }
  }

  export type PlanParams = {
    v: typeof currentVersion
    timestamp: number
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
        | { updatedHere: true; amount: number }
        | {
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
      expectedReturnsForPlanning:
        | { type: 'regressionPrediction,20YearTIPSYield' }
        | { type: 'conservativeEstimate,20YearTIPSYield' }
        | { type: '1/CAPE,20YearTIPSYield' }
        | { type: 'historical' }
        | { type: 'manual'; stocks: number; bonds: number }

      historicalMonthlyLogReturnsAdjustment: {
        standardDeviation: {
          stocks: { scale: number }
          bonds: { enableVolatility: boolean }
        }
        overrideToFixedForTesting: boolean
      }
      sampling: {
        type: 'monteCarlo' | 'historical'
        forMonteCarlo: {
          blockSize: InMonths
          staggerRunStarts: boolean
        }
      }
      annualInflation: { type: 'suggested' } | { type: 'manual'; value: number }
      strategy: 'TPAW' | 'SPAW' | 'SWR'
    }
    results: {
      displayedAssetAllocation: { stocks: number }
    } | null
  }

  // ----------- GUARD  ---------//

  const { uuid } = Guards
  const notEmpty = (x: string) =>
    x === '' ? failure('Empty string.') : success(x)

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

    const personType: JSONGuard<'person1' | 'person2'> = union(
      constant('person1'),
      constant('person2'),
    )
    const calendarMonth: JSONGuard<CalendarMonth> = object({
      year: chain(number, integer),
      month: chain(number, integer),
    })
    const monthOfBirth = chain(calendarMonth, (x) => {
      if (x.year < CONSTANTS.earliestYearOfBirth)
        return failure(
          `Earliest year of birth is ${CONSTANTS.earliestYearOfBirth}.`,
        )
      return success(x)
    })
    const inMonths: JSONGuard<InMonths> = object({
      inMonths: chain(number, integer),
    })
    const strategy: JSONGuard<PlanParams['advanced']['strategy']> = union(
      constant('TPAW'),
      constant('SPAW'),
      constant('SWR'),
    )
    const month = (planParams: PlanParams | null) =>
      union(
        object({
          type: constant('calendarMonthAsNow'),
          monthOfEntry: calendarMonth,
        }),
        object({
          type: constant('calendarMonth'),
          calendarMonth: calendarMonth,
        }),
        chain(
          object({
            type: constant('namedAge'),
            person: personType,
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
            person: personType,
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
          everyXMonths: among([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          ] as const),
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
          month: calendarMonth,
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
      PlanParams['advanced']['expectedReturnsForPlanning']
    > = union(
      object({ type: constant('regressionPrediction,20YearTIPSYield') }),
      object({ type: constant('conservativeEstimate,20YearTIPSYield') }),
      object({ type: constant('1/CAPE,20YearTIPSYield') }),
      object({ type: constant('historical') }),
      object({
        type: constant('manual'),
        stocks: among(CONSTANTS.manualStocksBondsNonLogAnnualReturnsValues),
        bonds: among(CONSTANTS.manualStocksBondsNonLogAnnualReturnsValues),
      }),
    )

    const historicalMonthlyLogReturnsAdjustment: JSONGuard<
      PlanParams['advanced']['historicalMonthlyLogReturnsAdjustment']
    > = object({
      standardDeviation: object({
        stocks: object({ scale: among(CONSTANTS.stockVolatilityScaleValues) }),
        bonds: object({ enableVolatility: boolean }),
      }),
      overrideToFixedForTesting: boolean,
    })

    const annualInflation: JSONGuard<
      PlanParams['advanced']['annualInflation']
    > = union(
      object({ type: constant('suggested') }),
      object({
        type: constant('manual'),
        value: among(CONSTANTS.manualInflationValues),
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

    return {
      calendarMonth,
      monthOfBirth,
      personType,
      strategy,
      inMonths,
      month,
      monthRange,
      monthRangeDuration,
      dialogPositionNominal,
      glidePath,
      samplingType,
      expectedReturnsForPlanning,
      historicalMonthlyLogReturnsAdjustment,
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
    }
  })
  const cg = componentGuards

  const _ages: JSONGuard<Person['ages']> = chain(
    union(
      object({
        type: constant('retiredWithNoRetirementDateSpecified'),
        monthOfBirth: cg.monthOfBirth,
        maxAge: cg.inMonths,
      }),
      object({
        type: constant('retirementDateSpecified'),
        monthOfBirth: cg.monthOfBirth,
        retirementAge: cg.inMonths,
        maxAge: cg.inMonths,
      }),
    ),
    (ages: Person['ages']): JSONGuardResult<Person['ages']> => {
      const { maxAge } = ages

      if (maxAge.inMonths > CONSTANTS.maxAgeInMonths)
        return failure(`Max age is greater than ${CONSTANTS.maxAgeInMonths}.`)

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

  const person: JSONGuard<Person> = object({
    ages: _ages,
  })

  const people: JSONGuard<PlanParams['people']> = union(
    object({
      withPartner: constant(false),
      person1: person,
    }),
    object({
      withPartner: constant(true),
      person2: person,
      person1: person,
      withdrawalStart: cg.personType,
    }),
  )

  const wealth = (params: PlanParams | null): JSONGuard<PlanParams['wealth']> =>
    chain(
      object({
        portfolioBalance: union(
          object({
            updatedHere: constant(true),
            amount: chain(number, gte(0)),
          }),
          object({
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

  const risk = (planParams: PlanParams | null): JSONGuard<PlanParams['risk']> =>
    object({
      tpaw: object({
        riskTolerance: object({
          at20: among(CONSTANTS.riskToleranceValues.DATA),
          deltaAtMaxAge: among(
            CONSTANTS.riskToleranceValues.DATA.map((x) => x * -1),
          ),
          forLegacyAsDeltaFromAt20: among(CONSTANTS.riskToleranceValues.DATA),
        }),
        timePreference: among(CONSTANTS.timePreferenceValues),
        additionalAnnualSpendingTilt: among(
          CONSTANTS.additionalAnnualSpendingTiltValues,
        ),
      }),
      tpawAndSPAW: object({
        lmp: chain(number, gte(0)),
      }),
      spaw: object({
        annualSpendingTilt: among(CONSTANTS.spawAnnualSpendingTiltValues),
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

  const advanced: JSONGuard<PlanParams['advanced']> = object({
    expectedReturnsForPlanning: cg.expectedReturnsForPlanning,
    historicalMonthlyLogReturnsAdjustment:
      cg.historicalMonthlyLogReturnsAdjustment,
    annualInflation: cg.annualInflation,
    sampling: object({
      type: cg.samplingType,
      forMonteCarlo: object({
        blockSize: object({
          inMonths: chain(
            number,
            integer,
            gte(1),
            lte(CONSTANTS.maxAgeInMonths),
          ),
        }),
        staggerRunStarts: boolean,
      }),
    }),
    strategy: cg.strategy,
  })

  const oneOrTwoPassGuard = (x: PlanParams | null): JSONGuard<PlanParams> =>
    object({
      v: constant(currentVersion),
      timestamp: number,
      dialogPositionNominal: cg.dialogPositionNominal,
      people: people,
      wealth: wealth(x),
      adjustmentsToSpending: adjustmentsToSpending(x),
      risk: risk(x),
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

    const _migrateLabel = (x: string | null): string | null =>
      x
        ? letIn(x.trim(), (trimmed) => (trimmed.length === 0 ? null : trimmed))
        : null

    const _migrateLabeledAmount = (
      prev: PlanParamsPrev.LabeledAmount,
    ): LabeledAmountUntimed => ({
      ...prev,
      label: _migrateLabel(prev.label),
      amount: prev.value,
    })

    const _migrateLabeledAmounts = (
      x: PlanParamsPrev.LabeledAmounts,
    ): LabeledAmountUntimedList => _.mapValues(x, _migrateLabeledAmount)

    const _migrateMonthRange = (
      prev: PlanParamsPrev.MonthRange,
    ): MonthRange => {
      switch (prev.type) {
        case 'startAndEnd':
          return prev
        case 'startAndNumMonths':
          return {
            type: 'startAndDuration',
            start: prev.start,
            duration: { inMonths: prev.numMonths },
          }
        case 'endAndNumMonths':
          return {
            type: 'endAndDuration',
            end: prev.end,
            duration: { inMonths: prev.numMonths },
          }
        default:
          noCase(prev)
      }
    }

    const _migrateAmountForMonthRange = (
      prev: PlanParamsPrev.ValueForMonthRange,
    ): LabeledAmountTimed => ({
      label: _migrateLabel(prev.label),
      nominal: prev.nominal,
      id: prev.id,
      sortIndex: prev.sortIndex,
      colorIndex: prev.colorIndex,
      amountAndTiming: {
        type: 'recurring',
        monthRange: _migrateMonthRange(prev.monthRange),
        everyXMonths: 1,
        baseAmount: prev.value,
        delta: null,
      },
    })

    const _migrateValueForMonthRanges = (
      x: PlanParamsPrev.ValueForMonthRanges,
    ): LabeledAmountTimedList => _.mapValues(x, _migrateAmountForMonthRange)

    const result: PlanParams = {
      ...prev,
      v: currentVersion,
      wealth: {
        ...prev.wealth,
        futureSavings: _migrateValueForMonthRanges(prev.wealth.futureSavings),
        incomeDuringRetirement: _migrateValueForMonthRanges(
          prev.wealth.incomeDuringRetirement,
        ),
      },
      adjustmentsToSpending: {
        ...prev.adjustmentsToSpending,
        extraSpending: {
          essential: _migrateValueForMonthRanges(
            prev.adjustmentsToSpending.extraSpending.essential,
          ),
          discretionary: _migrateValueForMonthRanges(
            prev.adjustmentsToSpending.extraSpending.discretionary,
          ),
        },
        tpawAndSPAW: {
          ...prev.adjustmentsToSpending.tpawAndSPAW,
          legacy: {
            ...prev.adjustmentsToSpending.tpawAndSPAW.legacy,
            external: _migrateLabeledAmounts(
              prev.adjustmentsToSpending.tpawAndSPAW.legacy.external,
            ),
          },
        },
      },
      advanced: {
        ...prev.advanced,
        sampling: {
          ...prev.advanced.sampling,
          forMonteCarlo: {
            ...prev.advanced.sampling.forMonteCarlo,
            blockSize: {
              inMonths: prev.advanced.sampling.forMonteCarlo.blockSize,
            },
          },
        },
      },
    }
    assert(!guard(result).error)
    return result
  }
}
