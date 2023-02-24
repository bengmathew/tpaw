import {
  array,
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
import { noCase, preciseRange } from '../Utils'
import { PlanParams18 } from './Old/PlanParams18'

export namespace PlanParams19 {
  export const MAX_LABEL_LENGTH = 150
  export const MAX_AGE_IN_MONTHS = 120 * 12
  export const MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY = 1000
  export const MAX_SIZE_FOR_MONTH_RANGE_ARR = 100
  export const MAX_EXTERNAL_LEGACY_SOURCES = 100
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

    const riskToleranceToRRA = (riskTolerance: number) =>
      1 / Math.exp((riskTolerance - 1) / scale + shift)

    riskToleranceToRRA.inverse = (rra: number) =>
      (log1OverRRA(rra) - shift) * scale + 1

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
    return { DATA, SEGMENTS, riskToleranceToRRA }
  })()

  export type Person = {
    ages:
      | { type: 'retired'; currentMonth: number; maxMonth: number }
      | {
          type: 'notRetired'
          currentMonth: number
          retirementMonth: number
          maxMonth: number
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
    | { type: 'now' }
    | {
        type: 'namedAge'
        person: 'person1' | 'person2'
        age: 'lastWorkingMonth' | 'retirement' | 'max'
      }
    | {
        type: 'numericAge'
        person: 'person1' | 'person2'
        ageInMonths: number
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
    id: number
  }

  export type LabeledAmount = {
    label: string | null
    value: number
    nominal: boolean
    id: number
  }

  export type GlidePath = {
    start: { stocks: number }
    intermediate: { month: Month; stocks: number }[]
    end: { stocks: number }
  }

  export type Params = {
    v: 19
    warnedAbout14to15Converstion: boolean
    warnedAbout16to17Converstion: boolean
    dialogPosition:
      | 'age'
      | 'current-portfolio-balance'
      | 'future-savings'
      | 'income-during-retirement'
      | 'show-results'
      | 'show-all-inputs'
      | 'done'

    people: People

    // Wealth
    wealth: {
      currentPortfolioBalance: number
      futureSavings: ValueForMonthRange[]
      retirementIncome: ValueForMonthRange[]
    }

    // Adjustments to Spending
    adjustmentsToSpending: {
      extraSpending: {
        essential: ValueForMonthRange[]
        discretionary: ValueForMonthRange[]
      }
      tpawAndSPAW: {
        monthlySpendingCeiling: number | null
        monthlySpendingFloor: number | null
        legacy: {
          total: number
          external: LabeledAmount[]
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
      annualReturns: {
        expected:
          | { type: 'suggested' }
          | { type: 'oneOverCAPE' }
          | { type: 'regressionPrediction' }
          | { type: 'historical' }
          | { type: 'manual'; stocks: number; bonds: number }
        historical:
          | {
              type: 'adjusted'
              adjustment:
                | { type: 'by'; stocks: number; bonds: number }
                | { type: 'to'; stocks: number; bonds: number }
                | { type: 'toExpected' }
              correctForBlockSampling: boolean
            }
          | { type: 'fixed'; stocks: number; bonds: number }
          | { type: 'unadjusted' }
      }
      annualInflation: { type: 'suggested' } | { type: 'manual'; value: number }
      sampling: 'monteCarlo' | 'historical'
      samplingBlockSizeForMonteCarlo: number
      strategy: 'TPAW' | 'SPAW' | 'SWR'
    }
    dev: {
      alwaysShowAllMonths: boolean
    }
  }

  // ----------- GUARD  ---------//

  const among =
    <T>(values: T[]): JSONGuard<T> =>
    (x: unknown) => {
      return values.includes(x as T)
        ? success(x as T)
        : failure('Not among predefined value.')
    }

  const _ageRange = chain(number, gte(0), lte(MAX_AGE_IN_MONTHS))

  const _ages: JSONGuard<Person['ages']> = chain(
    union(
      object({
        type: constant('retired'),
        currentMonth: _ageRange,
        maxMonth: _ageRange,
      }),
      object({
        type: constant('notRetired'),
        currentMonth: _ageRange,
        retirementMonth: _ageRange,
        maxMonth: _ageRange,
      }),
    ),
    (ages: Person['ages']): JSONGuardResult<Person['ages']> => {
      const { currentMonth, maxMonth } = ages
      // Two months because asset allocation graph may not show last month,
      // and needs at least 2 other months, so total of 3 (note the range is
      // inclusive, so +2 allows for 3 months.)
      if (maxMonth < currentMonth + 2) {
        return failure(
          'Max age should be at least two months after current age.',
        )
      }
      if (ages.type === 'notRetired') {
        const { retirementMonth } = ages
        if (retirementMonth < currentMonth + 1) {
          return failure(
            'Retirement age should be at least one month after current age.',
          )
        }
        if (maxMonth < retirementMonth + 1) {
          return failure(
            'Max age should be at least one month after retirement age.',
          )
        }
      }
      return success(ages)
    },
  )
  const dialogPosition = (
    params: Params | null,
  ): JSONGuard<Params['dialogPosition']> =>
    chain(
      union(
        constant('age'),
        constant('current-portfolio-balance'),
        constant('future-savings'),
        constant('income-during-retirement'),
        constant('show-results'),
        constant('show-all-inputs'),
        constant('done'),
      ),
      (x): JSONGuardResult<Params['dialogPosition']> => {
        if (!params) return success(x)
        if (params.dialogPosition !== 'future-savings') return success(x)
        const targetPerson = !params.people.withPartner
          ? params.people.person1
          : params.people[params.people.withdrawalStart]
        if (targetPerson.ages.type !== 'retired') return success(x)
        return failure(
          'dialogPosition is future-savings, but withdrawals have already started.',
        )
      },
    )

  const person: JSONGuard<Person> = object({
    ages: _ages,
  })

  const people: JSONGuard<Params['people']> = union(
    object({
      withPartner: constant(false),
      person1: person,
    }),
    object({
      withPartner: constant(true),
      person2: person,
      person1: person,
      withdrawalStart: union(constant('person1'), constant('person2')),
    }),
  )

  const month = (params: Params | null): JSONGuard<Month> =>
    union(
      object({ type: constant('now') }),
      chain(
        object({
          type: constant('namedAge'),
          person: union(constant('person1'), constant('person2')),
          age: union(
            constant('lastWorkingMonth'),
            constant('retirement'),
            constant('max'),
          ),
        }),
        (x) => {
          if (!params) return success(x)
          const { people } = params
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
            person.ages.type === 'retired'
          ) {
            return failure(
              `In terms retirement age of ${x.person}, but ${x.person} is already retired.`,
            )
          }
          return success(x)
        },
      ),
      chain(
        object({
          type: constant('numericAge'),
          person: union(constant('person1'), constant('person2')),
          ageInMonths: chain(number, integer),
        }),
        (x) => {
          if (!params) return success(x)
          const { people } = params
          if (x.person === 'person2' && !people.withPartner)
            return failure('In terms of partner, but there is no partner.')
          return success(x)
        },
      ),
    )

  const monthRange = (params: Params | null): JSONGuard<MonthRange> => {
    const _month = month(params)
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

  const valueForMonthRange = (
    params: Params | null,
  ): JSONGuard<ValueForMonthRange> =>
    object({
      // Not trimmed because it won't allow space even temporarily.
      label: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
      monthRange: monthRange(params),
      value: chain(number, gte(0)),
      nominal: boolean,
      id: chain(number, integer, gte(0)),
    })

  const valueForMonthRangeArr = (
    params: Params | null = null,
  ): JSONGuard<ValueForMonthRange[]> =>
    array(valueForMonthRange(params), MAX_SIZE_FOR_MONTH_RANGE_ARR)

  const wealth = (params: Params | null): JSONGuard<Params['wealth']> =>
    object({
      currentPortfolioBalance: chain(number, gte(0)),
      futureSavings: valueForMonthRangeArr(params),
      retirementIncome: valueForMonthRangeArr(params),
    })

  const adjustmentsToSpending = (
    params: Params | null,
  ): JSONGuard<Params['adjustmentsToSpending']> =>
    object({
      tpawAndSPAW: object({
        monthlySpendingCeiling: chain(
          nullable(chain(number, gte(0))),
          chain(nullable(chain(number, gte(0))), (x) => {
            if (!params) return success(x)
            if (x === null) return success(x)
            if (
              params.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor !==
                null &&
              x < params.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor
            ) {
              failure('Spending floor is greater than spending ceiling.')
            }
            return success(x)
          }),
        ),
        monthlySpendingFloor: nullable(chain(number, gte(0))),
        legacy: object({
          total: chain(number, gte(0)),
          external: array(
            object({
              label: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
              value: chain(number, gte(0)),
              nominal: boolean,
              id: number,
            }),
            MAX_EXTERNAL_LEGACY_SOURCES,
          ),
        }),
      }),
      extraSpending: object({
        essential: array(
          valueForMonthRange(params),
          MAX_SIZE_FOR_MONTH_RANGE_ARR,
        ),
        discretionary: array(
          valueForMonthRange(params),
          MAX_SIZE_FOR_MONTH_RANGE_ARR,
        ),
      }),
    })

  const glidePath = (params: Params | null): JSONGuard<GlidePath> =>
    object({
      start: object({ stocks: chain(number, gte(0), lte(1)) }),
      intermediate: array(
        object({
          month: month(params),
          stocks: chain(number, gte(0), lte(1)),
        }),
        MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY,
      ),
      end: object({ stocks: chain(number, gte(0), lte(1)) }),
    })

  const risk = (params: Params | null): JSONGuard<Params['risk']> =>
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
            if (params === null) return success(x)
            if (params.advanced.strategy === 'SWR' && x.type === 'default')
              return failure(
                'SWR withdrawal strategy cannot be "default" when in SWR mode.',
              )
            return success(x)
          },
        ),
      }),
      spawAndSWR: object({
        allocation: glidePath(params),
      }),
    })

  const annualReturns: JSONGuard<Params['advanced']['annualReturns']> = object({
    expected: union(
      object({ type: constant('suggested') }),
      object({ type: constant('oneOverCAPE') }),
      object({ type: constant('regressionPrediction') }),
      object({ type: constant('historical') }),
      object({
        type: constant('manual'),
        stocks: among(MANUAL_STOCKS_BONDS_RETURNS_VALUES),
        bonds: among(MANUAL_STOCKS_BONDS_RETURNS_VALUES),
      }),
    ),
    historical: union(
      object({
        type: constant('adjusted'),
        adjustment: union(
          object({
            type: constant('to'),
            stocks: number,
            bonds: number,
          }),
          object({
            type: constant('by'),
            stocks: number,
            bonds: number,
          }),
          object({ type: constant('toExpected') }),
        ),
        correctForBlockSampling: boolean,
      }),
      object({
        type: constant('fixed'),
        stocks: number,
        bonds: number,
      }),
      object({ type: constant('unadjusted') }),
    ),
  })

  const annualInflation: JSONGuard<Params['advanced']['annualInflation']> =
    union(
      object({ type: constant('suggested') }),
      object({
        type: constant('manual'),
        value: among(MANUAL_INFLATION_VALUES),
      }),
    )

  const advanced: JSONGuard<Params['advanced']> = object({
    annualReturns,
    annualInflation,
    sampling: union(constant('monteCarlo'), constant('historical')),
    samplingBlockSizeForMonteCarlo: chain(
      number,
      integer,
      gte(1),
      lte(MAX_AGE_IN_MONTHS),
    ),
    strategy: union(constant('TPAW'), constant('SPAW'), constant('SWR')),
  })

  const params = (x: Params | null): JSONGuard<Params> =>
    object({
      v: constant(19),
      warnedAbout14to15Converstion: boolean,
      warnedAbout16to17Converstion: boolean,
      dialogPosition: dialogPosition(x),
      people,
      wealth: wealth(x),
      adjustmentsToSpending: adjustmentsToSpending(x),
      risk: risk(x),
      advanced,
      dev: object({
        alwaysShowAllMonths: boolean,
      }),
    })

  const currGuard: JSONGuard<Params> = chain(params(null), (x) => params(x)(x))

  export const guard: JSONGuard<Params> = (x: unknown) => {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
    if ((x as any).v === 19) {
      return currGuard(x)
    } else {
      const prevResult = PlanParams18.guard(x)
      if (prevResult.error) return prevResult
      const { ...prev } = prevResult.value

      const updateAges = (ages: PlanParams18.Person['ages']): Person['ages'] =>
        ages.type === 'retired'
          ? {
              type: 'retired',
              currentMonth: ages.current * 12,
              maxMonth: ages.max * 12,
            }
          : {
              type: 'notRetired',
              currentMonth: ages.current * 12,
              retirementMonth: ages.retirement * 12,
              maxMonth: ages.max * 12,
            }

      const updateYear = (year: PlanParams18.Year): Month => {
        switch (year.type) {
          case 'now':
            return { type: 'now' }
          case 'namedAge': {
            return {
              type: 'namedAge',
              person: year.person,
              age:
                year.age === 'lastWorkingYear' ? 'lastWorkingMonth' : year.age,
            }
          }
          case 'numericAge': {
            return {
              type: 'numericAge',
              person: year.person,
              ageInMonths: year.age * 12,
            }
          }
          default:
            noCase(year)
        }
      }

      const updateYearRange = (x: PlanParams18.YearRange): MonthRange => {
        switch (x.type) {
          case 'startAndEnd':
            return {
              type: 'startAndEnd',
              start: updateYear(x.start),
              end: updateYear(x.end),
            }
          case 'startAndNumYears':
            return {
              type: 'startAndNumMonths',
              start: updateYear(x.start),
              numMonths: x.numYears * 12,
            }
          case 'endAndNumYears':
            return {
              type: 'endAndNumMonths',
              end: updateYear(x.end),
              numMonths: x.numYears * 12,
            }
          default:
            noCase(x)
        }
      }

      const updateValueForYearRange = (
        x: PlanParams18.ValueForYearRange,
      ): ValueForMonthRange => ({
        label: x.label,
        monthRange: updateYearRange(x.yearRange),
        value: Math.round(x.value / 12),
        nominal: x.nominal,
        id: x.id,
      })
      const curr: Params = {
        v: 19,
        warnedAbout14to15Converstion: prev.warnedAbout14to15Converstion,
        warnedAbout16to17Converstion: prev.warnedAbout16to17Converstion,
        dialogPosition: prev.dialogPosition,
        people: prev.people.withPartner
          ? {
              withPartner: true,
              person1: { ages: updateAges(prev.people.person1.ages) },
              person2: { ages: updateAges(prev.people.person2.ages) },
              withdrawalStart: prev.people.withdrawalStart,
            }
          : {
              withPartner: false,
              person1: { ages: updateAges(prev.people.person1.ages) },
            },
        wealth: {
          currentPortfolioBalance: prev.wealth.currentPortfolioBalance,
          futureSavings: prev.wealth.futureSavings.map(updateValueForYearRange),
          retirementIncome: prev.wealth.retirementIncome.map(
            updateValueForYearRange,
          ),
        },
        adjustmentsToSpending: {
          extraSpending: {
            essential: prev.adjustmentsToSpending.extraSpending.essential.map(
              updateValueForYearRange,
            ),
            discretionary:
              prev.adjustmentsToSpending.extraSpending.discretionary.map(
                updateValueForYearRange,
              ),
          },
          tpawAndSPAW: {
            legacy: {
              total: prev.adjustmentsToSpending.tpawAndSPAW.legacy.total,
              external:
                prev.adjustmentsToSpending.tpawAndSPAW.legacy.external.map(
                  (e, i) => ({ ...e, id: i }),
                ),
            },
            monthlySpendingCeiling: prev.adjustmentsToSpending.tpawAndSPAW
              .spendingCeiling
              ? Math.round(
                  prev.adjustmentsToSpending.tpawAndSPAW.spendingCeiling / 12,
                )
              : null,
            monthlySpendingFloor: prev.adjustmentsToSpending.tpawAndSPAW
              .spendingFloor
              ? Math.round(
                  prev.adjustmentsToSpending.tpawAndSPAW.spendingFloor / 12,
                )
              : null,
          },
        },
        risk: {
          tpaw: {
            riskTolerance: prev.risk.tpaw.riskTolerance,
            timePreference: prev.risk.tpaw.timePreference,
            additionalAnnualSpendingTilt: prev.risk.tpaw.additionalSpendingTilt,
          },
          tpawAndSPAW: {
            lmp: prev.risk.tpawAndSPAW.lmp,
          },
          spaw: {
            annualSpendingTilt: prev.risk.spaw.spendingTilt,
          },
          spawAndSWR: {
            allocation: {
              start: prev.risk.spawAndSWR.allocation.start,
              intermediate: prev.risk.spawAndSWR.allocation.intermediate.map(
                (x) => ({ stocks: x.stocks, month: updateYear(x.year) }),
              ),
              end: prev.risk.spawAndSWR.allocation.end,
            },
          },
          swr: {
            withdrawal: (() => {
              switch (prev.risk.swr.withdrawal.type) {
                case 'asAmount':
                  return {
                    type: 'asAmountPerMonth',
                    amountPerMonth: Math.round(
                      prev.risk.swr.withdrawal.amount / 12,
                    ),
                  }
                case 'asPercent':
                  return {
                    type: 'asPercentPerYear',
                    percentPerYear: prev.risk.swr.withdrawal.percent,
                  }
                case 'default':
                  return { type: 'default' }
                default:
                  noCase(prev.risk.swr.withdrawal)
              }
            })(),
          },
        },
        advanced: {
          annualReturns: {
            expected: prev.returns.expected,
            historical: (() => {
              const { historical } = prev.returns
              if (historical.type === 'fixed') return historical
              const adjust = historical.adjust
              if (adjust.type === 'none') return { type: 'unadjusted' }
              return {
                type: 'adjusted',
                adjustment: adjust,
                correctForBlockSampling: true,
              }
            })(),
          },
          annualInflation: prev.inflation,
          sampling: prev.sampling,
          samplingBlockSizeForMonteCarlo: 12 * 5,
          strategy: prev.strategy,
        },
        dev: {
          alwaysShowAllMonths: prev.display.alwaysShowAllYears,
        },
      }
      return success(currGuard(curr).force())
    }
  }
}
