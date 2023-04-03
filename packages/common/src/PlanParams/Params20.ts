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
  lt,
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
import { fGet, noCase, preciseRange } from '../Utils'
import { PlanParams19 as PlanParamsPrev } from './Old/PlanParams19'

export namespace Params20 {
  export const MAX_LABEL_LENGTH = 150
  export const MAX_AGE_IN_MONTHS = 120 * 12
  export const MIN_AGE_IN_MONTHS = 120 * 10
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

  export const MIN_PLAN_PARAM_TIME = 1680481336120

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
    id: number
  }

  export type LabeledAmount = {
    label: string | null
    value: number
    nominal: boolean
    id: number
  }

  export type GlidePath = {
    start: { month: CalendarMonth; stocks: number }
    intermediate: { month: Month; stocks: number }[]
    end: { stocks: number }
  }

  export type Params = {
    v: 20

    plan: {
      timestamp: number
      // Technically should be in non-plan, but this moves with plan changes,
      // so simpler to keep it here.
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
        // Should be estimate except if there nave not been any full market days
        // between portfolio update and plan update.
        portfolioBalance:
          | {
              isLastPlanChange: false
              history: {
                monthBoundaryDetails: {
                  // Not using timestamp to float this to the evaluation timezone.
                  startOfMonth: CalendarMonth
                  netContributionOrWithdrawal:
                    | { type: 'contribution'; contribution: number }
                    | { type: 'withdrawal'; withdrawal: number }
                  allocation: { stocks: number }
                }[]
                planChangeStockAllocations: {
                  effectiveAtMarketCloseTime: number
                  allocation: { stocks: number }
                }[]
              }

              original: {
                amount: number
                timestamp: number
              }
            }
          | {
              isLastPlanChange: true
              amount: number
              timestamp: number
            }
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
        annualInflation:
          | { type: 'suggested' }
          | { type: 'manual'; value: number }
        sampling: 'monteCarlo' | 'historical'
        monteCarloSampling: {
          blockSize: number
          numOfSimulations: number
        }
        strategy: 'TPAW' | 'SPAW' | 'SWR'
      }
    }

    nonPlan: {
      migrationWarnings: {
        v14tov15: boolean
        v16tov17: boolean
        v19tov20: boolean
      }

      percentileRange: { start: number; end: number }
      defaultTimezone: {
        type: 'auto' | 'manual'
        ianaTimezoneName: string
      }
      dev: {
        alwaysShowAllMonths: boolean
        currentTimeFastForward:
          | { shouldFastForward: false }
          | {
              shouldFastForward: true
              years: number
              months: number
              days: number
              hours: number
              restoreTo: string
              marketDataExtensionStrategy: {
                dailyStockMarketPerformance: // Probably want expected to be 'manual'.
                | 'latestExpected'
                  | 'roundRobinPastValues'
                  | 'repeatGrowShrinkZero'
              }
            }
      }
    }
  }

  export type PlanParams = Params['plan']
  export type NonPlanParams = Params['nonPlan']

  // ----------- GUARD  ---------//

  const among =
    <T>(values: T[]): JSONGuard<T> =>
    (x: unknown) => {
      return values.includes(x as T)
        ? success(x as T)
        : failure('Not among predefined value.')
    }

  const _calendarMonth: JSONGuard<CalendarMonth> = object({
    year: chain(number, integer),
    month: chain(number, integer),
  })

  const _inMonths: JSONGuard<InMonths> = object({
    inMonths: chain(number, integer),
  })

  const _ages: JSONGuard<Person['ages']> = chain(
    union(
      object({
        type: constant('retiredWithNoRetirementDateSpecified'),
        monthOfBirth: _calendarMonth,
        maxAge: _inMonths,
      }),
      object({
        type: constant('retirementDateSpecified'),
        monthOfBirth: _calendarMonth,
        retirementAge: _inMonths,
        maxAge: _inMonths,
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

  const dialogPosition = (
    params: Params | null,
  ): JSONGuard<PlanParams['dialogPosition']> =>
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
      (x): JSONGuardResult<PlanParams['dialogPosition']> => {
        if (!params) return success(x)
        if (params.plan.dialogPosition !== 'future-savings') return success(x)
        const targetPerson = !params.plan.people.withPartner
          ? params.plan.people.person1
          : params.plan.people[params.plan.people.withdrawalStart]

        if (targetPerson.ages.type !== 'retiredWithNoRetirementDateSpecified')
          return success(x)
        return failure(
          'dialogPosition is future-savings, but withdrawals have already started.',
        )
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
      withdrawalStart: union(constant('person1'), constant('person2')),
    }),
  )

  const month = (params: Params | null): JSONGuard<Month> =>
    union(
      object({
        type: constant('calendarMonthAsNow'),
        monthOfEntry: _calendarMonth,
      }),
      object({
        type: constant('calendarMonth'),
        calendarMonth: _calendarMonth,
      }),
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
          const { people } = params.plan
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
          person: union(constant('person1'), constant('person2')),
          age: _inMonths,
        }),
        (x) => {
          if (!params) return success(x)
          const { people } = params.plan
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

  const _portfolioBalanceHistoryPlanChangeStockAllocations: JSONGuard<
    Extract<
      PlanParams['wealth']['portfolioBalance'],
      { isLastPlanChange: false }
    >['history']['planChangeStockAllocations']
  > = chain(
    array(
      object({
        effectiveAtMarketCloseTime: number,
        allocation: object({ stocks: chain(number, gte(0), lte(1)) }),
      }),
      365 * 10,
    ),
    (x) => {
      if (x.length === 0) return failure('Should have at least one entry.')
      if (
        !x.every(
          (_, i) =>
            i === 0 ||
            fGet(x[i - 1]).effectiveAtMarketCloseTime <
              fGet(x[i]).effectiveAtMarketCloseTime,
        )
      )
        return failure('Not sorted or has duplicates.')
      return success(x)
    },
  )

  const _portfolioBalance: JSONGuard<PlanParams['wealth']['portfolioBalance']> =
    union(
      object({
        isLastPlanChange: constant(true),
        amount: chain(number, gte(0)),
        timestamp: number,
      }),
      object({
        isLastPlanChange: constant(false),
        history: object({
          monthBoundaryDetails: array(
            object({
              startOfMonth: _calendarMonth,
              netContributionOrWithdrawal: union(
                object({
                  type: constant('contribution'),
                  contribution: chain(number, gte(0)),
                }),
                object({
                  type: constant('withdrawal'),
                  withdrawal: chain(number, gte(0)),
                }),
              ),
              allocation: object({ stocks: chain(number, gte(0), lte(1)) }),
            }),
            12 * 100,
          ),
          planChangeStockAllocations:
            _portfolioBalanceHistoryPlanChangeStockAllocations,
        }),
        original: object({
          amount: chain(number, gte(0)),
          timestamp: number,
        }),
      }),
    )
  const wealth = (params: Params | null): JSONGuard<PlanParams['wealth']> =>
    object({
      portfolioBalance: _portfolioBalance,
      futureSavings: valueForMonthRangeArr(params),
      retirementIncome: valueForMonthRangeArr(params),
    })

  const adjustmentsToSpending = (
    params: Params | null,
  ): JSONGuard<PlanParams['adjustmentsToSpending']> =>
    object({
      tpawAndSPAW: object({
        monthlySpendingCeiling: chain(
          nullable(chain(number, gte(0))),
          chain(nullable(chain(number, gte(0))), (x) => {
            if (!params) return success(x)
            if (x === null) return success(x)
            if (
              params.plan.adjustmentsToSpending.tpawAndSPAW
                .monthlySpendingFloor !== null &&
              x <
                params.plan.adjustmentsToSpending.tpawAndSPAW
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
      start: object({
        month: _calendarMonth,
        stocks: chain(number, gte(0), lte(1)),
      }),
      intermediate: array(
        object({
          month: month(params),
          stocks: chain(number, gte(0), lte(1)),
        }),
        MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY,
      ),
      end: object({ stocks: chain(number, gte(0), lte(1)) }),
    })

  const risk = (params: Params | null): JSONGuard<PlanParams['risk']> =>
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
            if (params.plan.advanced.strategy === 'SWR' && x.type === 'default')
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

  const annualReturns: JSONGuard<PlanParams['advanced']['annualReturns']> =
    object({
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

  const annualInflation: JSONGuard<PlanParams['advanced']['annualInflation']> =
    union(
      object({ type: constant('suggested') }),
      object({
        type: constant('manual'),
        value: among(MANUAL_INFLATION_VALUES),
      }),
    )

  const advanced: JSONGuard<PlanParams['advanced']> = object({
    annualReturns,
    annualInflation,
    sampling: union(constant('monteCarlo'), constant('historical')),
    monteCarloSampling: object({
      blockSize: chain(number, integer, gte(1), lte(MAX_AGE_IN_MONTHS)),
      numOfSimulations: chain(number, integer, gte(1)),
    }),
    strategy: union(constant('TPAW'), constant('SPAW'), constant('SWR')),
  })

  const params = (x: Params | null): JSONGuard<Params> =>
    object({
      v: constant(20),
      plan: object({
        timestamp: number,
        dialogPosition: dialogPosition(x),
        people: people,
        wealth: wealth(x),
        adjustmentsToSpending: adjustmentsToSpending(x),
        risk: risk(x),
        advanced,
      }),
      nonPlan: object({
        migrationWarnings: object({
          v14tov15: boolean,
          v16tov17: boolean,
          v19tov20: boolean,
        }),
        percentileRange: object({
          start: chain(number, integer, gte(1), lt(50)),
          end: chain(number, integer, gt(50), lte(99)),
        }),
        defaultTimezone: object({
          type: union(constant('auto'), constant('manual')),
          ianaTimezoneName: string,
        }),
        dev: object({
          alwaysShowAllMonths: boolean,
          currentTimeFastForward: union(
            object({
              shouldFastForward: constant(false),
            }),
            object({
              shouldFastForward: constant(true),
              restoreTo: string,
              years: chain(number, integer, gte(0)),
              months: chain(number, integer, gte(0)),
              days: chain(number, integer, gte(0)),
              hours: chain(number, integer, gte(0)),
              marketDataExtensionStrategy: object({
                dailyStockMarketPerformance: union(
                  constant('latestExpected'),
                  constant('roundRobinPastValues'),
                  constant('repeatGrowShrinkZero'),
                ),
              }),
            }),
          ),
        }),
      }),
    })

  const currGuard: JSONGuard<Params> = chain(params(null), (x) => params(x)(x))

  export const guard: JSONGuard<Params> = (x: unknown) => {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
    if ((x as any).v === 20) {
      return currGuard(x)
    } else {
      const prevResult = PlanParamsPrev.guard(x)
      if (prevResult.error) return prevResult
      const {
        warnedAbout14to15Converstion,
        warnedAbout16to17Converstion,
        advanced: prevAdvanced,
        people: prevPeople,
        wealth: prevWealth,
        adjustmentsToSpending: prevAdjustmentsToSpending,
        risk: prevRisk,
        ...prev
      } = prevResult.value

      // Migrate People
      const currentTime = DateTime.fromMillis(MIN_PLAN_PARAM_TIME)
      const currentMonth = calendarMonthFromTime(currentTime)
      const people = ((): People => {
        const migratePerson = ({ ages }: PlanParamsPrev.Person): Person => {
          const monthOfBirth = calendarMonthFromTime(
            currentTime.minus({ month: ages.currentMonth }),
          )
          switch (ages.type) {
            case 'retired':
              return {
                ages: {
                  type: 'retiredWithNoRetirementDateSpecified',
                  monthOfBirth,
                  maxAge: { inMonths: ages.maxMonth },
                },
              }
            case 'notRetired':
              return {
                ages: {
                  type: 'retirementDateSpecified',
                  monthOfBirth,
                  retirementAge: { inMonths: ages.retirementMonth },
                  maxAge: { inMonths: ages.maxMonth },
                },
              }
            default:
              noCase(ages)
          }
        }
        if (prevPeople.withPartner) {
          return {
            withPartner: true,
            person1: migratePerson(prevPeople.person1),
            person2: migratePerson(prevPeople.person2),
            withdrawalStart: prevPeople.withdrawalStart,
          }
        } else {
          return {
            withPartner: false,
            person1: migratePerson(prevPeople.person1),
          }
        }
      })()

      const migrateMonth = (month: PlanParamsPrev.Month): Month => {
        switch (month.type) {
          case 'now':
            return {
              type: 'calendarMonthAsNow',
              monthOfEntry: currentMonth,
            }
          case 'namedAge':
            return month
          case 'numericAge':
            return {
              type: 'numericAge',
              person: month.person,
              age: { inMonths: month.ageInMonths },
            }
          default:
            noCase(month)
        }
      }
      const migrateMonthRange = (
        monthRange: PlanParamsPrev.MonthRange,
      ): MonthRange => {
        switch (monthRange.type) {
          case 'endAndNumMonths':
            return { ...monthRange, end: migrateMonth(monthRange.end) }
          case 'startAndNumMonths':
            return { ...monthRange, start: migrateMonth(monthRange.start) }
          case 'startAndEnd':
            return {
              ...monthRange,
              start: migrateMonth(monthRange.start),
              end: migrateMonth(monthRange.end),
            }
          default:
            noCase(monthRange)
        }
      }
      const migrateValueForMonthRange = (
        x: PlanParamsPrev.ValueForMonthRange,
      ): ValueForMonthRange => {
        return { ...x, monthRange: migrateMonthRange(x.monthRange) }
      }
      const migrateGlidePath = (x: PlanParamsPrev.GlidePath): GlidePath => {
        return {
          start: { month: currentMonth, stocks: x.start.stocks },
          intermediate: x.intermediate.map(({ month, stocks }) => ({
            month: migrateMonth(month),
            stocks,
          })),
          end: x.end,
        }
      }
      const curr: Params = {
        v: 20,
        plan: {
          timestamp: MIN_PLAN_PARAM_TIME,
          dialogPosition: prev.dialogPosition,
          people,
          wealth: {
            portfolioBalance: {
              isLastPlanChange: true,
              amount: prevWealth.currentPortfolioBalance,
              timestamp: MIN_PLAN_PARAM_TIME,
            },
            futureSavings: prevWealth.futureSavings.map(
              migrateValueForMonthRange,
            ),
            retirementIncome: prevWealth.retirementIncome.map(
              migrateValueForMonthRange,
            ),
          },
          adjustmentsToSpending: {
            tpawAndSPAW: prevAdjustmentsToSpending.tpawAndSPAW,
            extraSpending: {
              essential: prevAdjustmentsToSpending.extraSpending.essential.map(
                migrateValueForMonthRange,
              ),
              discretionary:
                prevAdjustmentsToSpending.extraSpending.discretionary.map(
                  migrateValueForMonthRange,
                ),
            },
          },
          risk: {
            tpaw: prevRisk.tpaw,
            tpawAndSPAW: prevRisk.tpawAndSPAW,
            spaw: prevRisk.spaw,
            spawAndSWR: {
              allocation: migrateGlidePath(prevRisk.spawAndSWR.allocation),
            },
            swr: prevRisk.swr,
          },
          advanced: {
            annualReturns: prevAdvanced.annualReturns,
            annualInflation: prevAdvanced.annualInflation,
            sampling: prevAdvanced.sampling,
            strategy: prevAdvanced.strategy,
            monteCarloSampling: {
              blockSize: prevAdvanced.samplingBlockSizeForMonteCarlo,
              numOfSimulations: 500,
            },
          },
        },
        nonPlan: {
          migrationWarnings: {
            v14tov15: warnedAbout14to15Converstion,
            v16tov17: warnedAbout16to17Converstion,
            v19tov20: false,
          },
          percentileRange: { start: 5, end: 95 },
          defaultTimezone: {
            type: 'auto',
            ianaTimezoneName: currentTime.zoneName,
          },
          dev: {
            alwaysShowAllMonths: prev.dev.alwaysShowAllMonths,
            currentTimeFastForward: {
              shouldFastForward: false,
            },
          },
        },
      }
      return success(currGuard(curr).force())
    }
  }

  // ---------------------------------------
  // ---------------- Helpers --------------
  // ---------------------------------------

  export const calendarMonthFromTime = (time: DateTime): CalendarMonth => {
    return { year: time.year, month: time.month }
  }
}
