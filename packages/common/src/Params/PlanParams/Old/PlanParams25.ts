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
  union,
} from 'json-guard'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { Guards } from '../../../Guards'
import { assert, block, fGet, preciseRange } from '../../../Utils'
import { PlanParams24 as PlanParamsPrev } from './PlanParams24'

export namespace PlanParams25 {
  export const currentVersion = 25 as const

  export const MAX_LABEL_LENGTH = 150
  export const MAX_AGE_IN_MONTHS = 120 * 12
  export const MIN_AGE_IN_MONTHS = 120 * 10
  export const MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY = 1000
  export const MAX_SIZE_FOR_MONTH_RANGE_ARR = 100
  export const MAX_EXTERNAL_LEGACY_SOURCES = 100
  export const STOCK_VOLATILITY_SCALE_VALUES = preciseRange(0.5, 1.5, 0.01, 2)
  export const TIME_PREFERENCE_VALUES = preciseRange(-0.05, 0.05, 0.001, 3)
  export const ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES = preciseRange(
    -0.05,
    0.05,
    0.001,
    3,
  )
  export const MANUAL_INFLATION_VALUES = preciseRange(-0.01, 0.1, 0.001, 3)
  export const MANUAL_STOCKS_BONDS_RETURNS_VALUES = preciseRange(
    -0.01,
    0.1,
    0.001,
    3,
  )

  export const SPAW_ANNUAL_SPENDING_TILT_VALUES = preciseRange(
    -0.03,
    0.03,
    0.001,
    3,
  )
  export const RISK_TOLERANCE_VALUES = (() => {
    const numSegments = 5
    const countPerSegment = 5
    const numPoints = numSegments * countPerSegment

    const startRRA = 16
    const endRRA = 0.5
    const log1OverRRA = (rra: number) => Math.log(1 / rra)
    const shift = log1OverRRA(startRRA)
    const scale =
      (numPoints - 2) / (log1OverRRA(endRRA) - log1OverRRA(startRRA))

    const riskToleranceToRRA = block(() => {
      const withoutInfinityAtZero = (riskTolerance: number) =>
        1 / Math.exp((riskTolerance - 1) / scale + shift)
      const withInfinityAtZero = (riskTolerance: number) =>
        riskTolerance === 0 ? Infinity : withoutInfinityAtZero(riskTolerance)
      withoutInfinityAtZero.inverse = (rra: number) =>
        (log1OverRRA(rra) - shift) * scale + 1
      return { withInfinityAtZero, withoutInfinityAtZero }
    })

    const DATA = _.times(numPoints, (i) => i)

    const segmentDef = (segment: number, label: string) => {
      const startIndex = segment * numSegments
      const endIndex = startIndex + countPerSegment - 1 // end is inclusive
      const count = countPerSegment
      const containsIndex = (index: number) =>
        index >= startIndex && index <= endIndex
      return { startIndex, endIndex, containsIndex, label, count }
    }

    const SEGMENTS = [
      segmentDef(0, 'Very Conservative'),
      segmentDef(1, 'Conservative'),
      segmentDef(2, 'Moderate'),
      segmentDef(3, 'Aggressive'),
      segmentDef(4, 'Very Aggressive'),
    ]
    return {
      DATA,
      SEGMENTS,
      riskToleranceToRRA,
    }
  })()

  export const MIN_PLAN_PARAM_TIME = 1680481336120

  export const CONSTANTS = {
    dialogPositionOrder: [
      'age',
      'current-portfolio-balance',
      'future-savings',
      'income-during-retirement',
      'show-results',
      'show-all-inputs',
      'done',
    ] as const,
  }

  export type DialogPosition = (typeof CONSTANTS)['dialogPositionOrder'][number]
  export const fns = block(() => {
    const getNextDialogPosition = (
      x: Exclude<DialogPosition, 'done'>,
    ): DialogPosition =>
      fGet(
        CONSTANTS.dialogPositionOrder[
          CONSTANTS.dialogPositionOrder.indexOf(x) + 1
        ],
      )

    return {
      // Intentionally, future savings is allowed even if the couple is jointly
      // retired (withdrawals have started), but one of them are not individually
      // retired. This does not matter either way because future savings and income
      // during retirement are really the same thing under the hood.
      getIsFutureSavingsAllowed: (
        person1Retired: boolean,
        person2Retired: boolean | undefined,
      ) =>
        person2Retired === undefined
          ? !person1Retired
          : !(person1Retired && person2Retired),

      getNextDialogPosition,
      getDialogPositionEffective: (
        dialogPositionNominal: DialogPosition,
        isFutureSavingsAllowed: boolean,
      ) =>
        dialogPositionNominal === 'future-savings' && !isFutureSavingsAllowed
          ? getNextDialogPosition('future-savings')
          : dialogPositionNominal,
    }
  })

  // -----------------
  // MasterListOfMonths
  // ------------------
  // (always include "MasterListOfMonths" where this data is used, so we can
  // find and keep everything updated.)
  //
  // Value for Month Range
  // ---------------------
  // 1. Future Savings
  // 2. Income During Retirement
  // 3. Extra Spending - Essential
  // 3. Extra Spending - Discretionary
  //
  // Glide Path
  // ----------
  // 1. SPAW and SWR Allocation

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
    | { type: 'startAndEnd'; start: Month; end: Month }
    | { type: 'startAndNumMonths'; start: Month; numMonths: number }
    | { type: 'endAndNumMonths'; end: Month; numMonths: number }

  export type ValueForMonthRange = {
    label: string | null
    monthRange: MonthRange
    value: number
    nominal: boolean
    id: string
    sortIndex: number
    colorIndex: number
  }

  export type ValueForMonthRanges = Record<string, ValueForMonthRange>

  export type LabeledAmount = {
    label: string | null
    value: number
    nominal: boolean
    id: string
    sortIndex: number
    colorIndex: number
  }
  export type LabeledAmounts = Record<string, LabeledAmount>

  export type GlidePath = {
    start: { month: CalendarMonth; stocks: number }
    intermediate: Record<
      string,
      { id: string; month: Month; indexToSortByAdded: number; stocks: number }
    >
    end: { stocks: number }
  }

  type AdjustExpectedReturn =
    | { type: 'none' }
    | { type: 'toExpectedUsedForPlanning'; correctForBlockSampling: boolean }
    | {
        type: 'toAnnualExpectedReturn'
        annualExpectedReturn: number
        correctForBlockSampling: boolean
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
      futureSavings: ValueForMonthRanges
      incomeDuringRetirement: ValueForMonthRanges
    }

    // Adjustments to Spending
    adjustmentsToSpending: {
      extraSpending: {
        essential: ValueForMonthRanges
        discretionary: ValueForMonthRanges
      }
      tpawAndSPAW: {
        monthlySpendingCeiling: number | null
        monthlySpendingFloor: number | null
        legacy: {
          total: number
          external: LabeledAmounts
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
      expectedAnnualReturnForPlanning:
        | { type: 'suggested' }
        | { type: 'oneOverCAPE' }
        | { type: 'regressionPrediction' }
        | { type: 'historical' }
        | { type: 'manual'; stocks: number; bonds: number }
      historicalReturnsAdjustment: {
        stocks: {
          adjustExpectedReturn: AdjustExpectedReturn
          volatilityScale: number
        }
        bonds: {
          adjustExpectedReturn: AdjustExpectedReturn
          enableVolatility: boolean
        }
      }
      sampling: {
        type: 'monteCarlo' | 'historical'
        blockSizeForMonteCarloSampling: number
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

  const among =
    <T>(values: T[]): JSONGuard<T> =>
    (x: unknown) => {
      return values.includes(x as T)
        ? success(x as T)
        : failure('Not among predefined value.')
    }

  export const componentGuards = (() => {
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
    const inMonths: JSONGuard<InMonths> = object({
      inMonths: chain(number, integer),
    })
    const strategy: JSONGuard<PlanParams['advanced']['strategy']> = union(
      constant('TPAW'),
      constant('SPAW'),
      constant('SWR'),
    )
    const month = (planParams: PlanParams | null): JSONGuard<Month> =>
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

    const monthRange = (
      planParams: PlanParams | null,
    ): JSONGuard<MonthRange> => {
      const _month = month(planParams)
      return union(
        object({ type: constant('startAndEnd'), start: _month, end: _month }),
        object({
          type: constant('startAndNumMonths'),
          start: _month,
          numMonths: chain(number, integer, gt(0)),
        }),
        object({
          type: constant('endAndNumMonths'),
          end: _month,
          numMonths: chain(number, integer, gt(0)),
        }),
      )
    }

    const glidePath = (planParams: PlanParams | null): JSONGuard<GlidePath> =>
      object({
        start: object({
          month: calendarMonth,
          stocks: chain(number, gte(0), lte(1)),
        }),
        intermediate: arrAsObj(
          object({
            id: smallId,
            month: cg.month(planParams),
            stocks: chain(number, gte(0), lte(1)),
            indexToSortByAdded: chain(number, integer),
          }),
          MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY,
        ),
        end: object({ stocks: chain(number, gte(0), lte(1)) }),
      })

    const samplingType = union(constant('monteCarlo'), constant('historical'))

    const adjustExpectedReturn = union(
      object({ type: constant('none') }),
      object({
        type: constant('toExpectedUsedForPlanning'),
        correctForBlockSampling: boolean,
      }),
      object({
        type: constant('toAnnualExpectedReturn'),
        annualExpectedReturn: number,
        correctForBlockSampling: boolean,
      }),
    )

    const expectedAnnualReturnForPlanning = union(
      object({ type: constant('suggested') }),
      object({ type: constant('oneOverCAPE') }),
      object({ type: constant('regressionPrediction') }),
      object({ type: constant('historical') }),
      object({
        type: constant('manual'),
        stocks: among(MANUAL_STOCKS_BONDS_RETURNS_VALUES),
        bonds: among(MANUAL_STOCKS_BONDS_RETURNS_VALUES),
      }),
    )

    const annualInflation: JSONGuard<
      PlanParams['advanced']['annualInflation']
    > = union(
      object({ type: constant('suggested') }),
      object({
        type: constant('manual'),
        value: among(MANUAL_INFLATION_VALUES),
      }),
    )

    return {
      calendarMonth,
      personType,
      strategy,
      inMonths,
      month,
      monthRange,
      dialogPositionNominal,
      glidePath,
      samplingType,
      adjustExpectedReturn,
      expectedAnnualReturnForPlanning,
      annualInflation,
    }
  })()
  const cg = componentGuards

  const { calendarMonth, personType, strategy } = componentGuards

  const _ages: JSONGuard<Person['ages']> = chain(
    union(
      object({
        type: constant('retiredWithNoRetirementDateSpecified'),
        monthOfBirth: calendarMonth,
        maxAge: cg.inMonths,
      }),
      object({
        type: constant('retirementDateSpecified'),
        monthOfBirth: calendarMonth,
        retirementAge: cg.inMonths,
        maxAge: cg.inMonths,
      }),
    ),
    (ages: Person['ages']): JSONGuardResult<Person['ages']> => {
      const { maxAge } = ages

      if (maxAge.inMonths > MAX_AGE_IN_MONTHS)
        return failure(`Max age is greater than ${MAX_AGE_IN_MONTHS}.`)

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
      withdrawalStart: personType,
    }),
  )

  const smallId = chain(string, (x) =>
    x.length !== 10 ? failure('Does not have length === 10') : success(x),
  )

  const labeledAmount: JSONGuard<LabeledAmount> = object({
    label: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
    value: chain(number, gte(0)),
    nominal: boolean,
    id: smallId,
    sortIndex: chain(number, integer),
    colorIndex: chain(number, integer, gte(0)),
  })

  const valueForMonthRange = (
    planParams: PlanParams | null,
  ): JSONGuard<ValueForMonthRange> =>
    object({
      // Not trimmed because it won't allow space even temporarily.
      // TODO: non longer need not trimmed, add back to validation.
      label: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
      monthRange: cg.monthRange(planParams),
      value: chain(number, gte(0)),
      nominal: boolean,
      id: smallId,
      sortIndex: chain(number, integer),
      colorIndex: chain(number, integer, gte(0)),
    })

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

  const valueForMonthRanges = (
    params: PlanParams | null = null,
  ): JSONGuard<ValueForMonthRanges> =>
    arrAsObj(valueForMonthRange(params), MAX_SIZE_FOR_MONTH_RANGE_ARR)

  const labeledAmounts: JSONGuard<LabeledAmounts> = arrAsObj(
    labeledAmount,
    MAX_SIZE_FOR_MONTH_RANGE_ARR,
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
        futureSavings: valueForMonthRanges(params),
        incomeDuringRetirement: valueForMonthRanges(params),
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
          external: labeledAmounts,
        }),
      }),
      extraSpending: chain(
        object({
          essential: valueForMonthRanges(planParams),
          discretionary: valueForMonthRanges(planParams),
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
          at20: among(RISK_TOLERANCE_VALUES.DATA),
          deltaAtMaxAge: among(RISK_TOLERANCE_VALUES.DATA.map((x) => x * -1)),
          forLegacyAsDeltaFromAt20: among(RISK_TOLERANCE_VALUES.DATA),
        }),
        timePreference: among(TIME_PREFERENCE_VALUES),
        additionalAnnualSpendingTilt: among(
          ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES,
        ),
      }),
      tpawAndSPAW: object({
        lmp: chain(number, gte(0)),
      }),
      spaw: object({
        annualSpendingTilt: among(SPAW_ANNUAL_SPENDING_TILT_VALUES),
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
    expectedAnnualReturnForPlanning: cg.expectedAnnualReturnForPlanning,
    historicalReturnsAdjustment: object({
      stocks: object({
        adjustExpectedReturn: cg.adjustExpectedReturn,
        volatilityScale: among(STOCK_VOLATILITY_SCALE_VALUES),
      }),
      bonds: object({
        adjustExpectedReturn: cg.adjustExpectedReturn,
        enableVolatility: boolean,
      }),
    }),
    annualInflation: cg.annualInflation,
    sampling: object({
      type: cg.samplingType,
      blockSizeForMonteCarloSampling: chain(
        number,
        integer,
        gte(1),
        lte(MAX_AGE_IN_MONTHS),
      ),
    }),
    strategy,
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

    const result: PlanParams = {
      v: currentVersion,
      timestamp: prev.timestamp,
      dialogPositionNominal: prev.dialogPosition,
      people: prev.people,
      wealth: prev.wealth,
      adjustmentsToSpending: prev.adjustmentsToSpending,
      risk: prev.risk,
      advanced: prev.advanced,
      results: prev.results,
    }
    assert(!guard(result).error)
    return result
  }

  // ---------------------------------------
  // ---------------- Helpers --------------
  // ---------------------------------------

  export const calendarMonthFromTime = (time: DateTime): CalendarMonth => {
    return { year: time.year, month: time.month }
  }
}
