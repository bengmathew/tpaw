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
import { preciseRange } from '../../Utils'
import { PlanParams15 } from './PlanParams15'

export namespace PlanParams16 {
  export const MAX_LABEL_LENGTH = 150
  export const MAX_AGE = 120
  export const MAX_NUM_YEARS_IN_GLIDE_PATH = 1000
  export const MAX_VALUE_FOR_YEAR_RANGE = 100
  export const MAX_EXTERNAL_LEGACY_SOURCES = 100
  export const TIME_PREFERENCE_VALUES = preciseRange(-0.05, 0.05, 0.001, 3)
  export const MANUAL_INFLATION_VALUES = preciseRange(-0.01, 0.1, 0.001, 3)
  export const MANUAL_STOCKS_BONDS_RETURNS_VALUES = preciseRange(
    -0.01,
    0.1,
    0.001,
    3,
  )

  export const SPAW_SPENDING_TILT_VALUES = preciseRange(-0.03, 0.03, 0.001, 3)
  export const RISK_TOLERANCE_VALUES = (() => {
    const numSegments = 5
    const countPerSegment = 5
    const numPoints = numSegments * countPerSegment
    const startRRA = 8
    const endRRA = 0.25

    const { rraToRiskTolerance, riskToleranceToRRA } = (() => {
      const log1OverRRA = (rra: number) => Math.log(1 / rra)
      const shift = log1OverRRA(startRRA)
      const scale =
        (numPoints - 2) / (log1OverRRA(endRRA) - log1OverRRA(startRRA))

      const rraToRiskTolerance = (rra: number) =>
        (log1OverRRA(rra) - shift) * scale + 1

      const riskToleranceToRRA = (riskTolerance: number) =>
        1 / Math.exp((riskTolerance - 1) / scale + shift)

      return { rraToRiskTolerance, riskToleranceToRRA }
    })()

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
    return { DATA, SEGMENTS, riskToleranceToRRA, rraToRiskTolerance }
  })()

  export type Person = {
    ages:
      | { type: 'retired'; current: number; max: number }
      | { type: 'notRetired'; current: number; retirement: number; max: number }
    displayName: string | null
  }

  export type People =
    | { withPartner: false; person1: Person }
    | {
        withPartner: true
        person2: Person
        person1: Person
        withdrawalStart: 'person1' | 'person2'
        xAxis: 'person1' | 'person2'
      }

  export type Year =
    | { type: 'now' }
    | {
        type: 'namedAge'
        person: 'person1' | 'person2'
        age: 'lastWorkingYear' | 'retirement' | 'max'
      }
    | {
        type: 'numericAge'
        person: 'person1' | 'person2'
        age: number
      }

  export type YearRange =
    | { type: 'startAndEnd'; start: Year; end: Year }
    | { type: 'startAndNumYears'; start: Year; numYears: number }
    | { type: 'endAndNumYears'; end: Year; numYears: number }

  export type ValueForYearRange = {
    label: string | null
    yearRange: YearRange
    value: number
    nominal: boolean
    id: number
  }

  export type LabeledAmount = {
    label: string | null
    value: number
    nominal: boolean
  }

  export type GlidePath = {
    start: { stocks: number }
    intermediate: { year: Year; stocks: number }[]
    end: { stocks: number }
  }

  export type Params = {
    v: 16
    warnedAbout14to15Converstion: boolean
    strategy: 'TPAW' | 'SPAW' | 'SWR'
    dialogPosition:
      | 'age'
      | 'current-portfolio-balance'
      | 'future-savings'
      | 'income-during-retirement'
      | 'show-results'
      | 'show-all-inputs'
      | 'done'

    // Basic Inputs
    people: People
    currentPortfolioBalance: number
    futureSavings: ValueForYearRange[]
    retirementIncome: ValueForYearRange[]

    // Adjustments to Spending
    adjustmentsToSpending: {
      tpawAndSPAW: {
        spendingCeiling: number | null
        spendingFloor: number | null
        legacy: {
          total: number
          external: LabeledAmount[]
        }
      }
      extraSpending: {
        essential: ValueForYearRange[]
        discretionary: ValueForYearRange[]
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
      }
      tpawAndSPAW: {
        lmp: number
      }
      spaw: {
        spendingTilt: number
      }
      spawAndSWR: {
        allocation: GlidePath
      }
      swr: {
        withdrawal:
          | { type: 'asPercent'; percent: number }
          | { type: 'asAmount'; amount: number }
          | { type: 'default' }
      }
    }

    // Advanced.
    returns: {
      expected:
        | { type: 'suggested' }
        | { type: 'oneOverCAPE' }
        | { type: 'regressionPrediction' }
        | { type: 'historical' }
        | { type: 'manual'; stocks: number; bonds: number }
      historical:
        | {
            type: 'default'
            adjust:
              | { type: 'by'; stocks: number; bonds: number }
              | { type: 'to'; stocks: number; bonds: number }
              | { type: 'toExpected' }
              | { type: 'none' }
          }
        | { type: 'fixed'; stocks: number; bonds: number }
    }
    inflation: { type: 'suggested' } | { type: 'manual'; value: number }
    sampling: 'monteCarlo' | 'historical'

    // Other.
    display: {
      alwaysShowAllYears: boolean
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

  const _ageRange = chain(number, gte(0), lte(MAX_AGE))

  const _ages: JSONGuard<Person['ages']> = chain(
    union(
      object({
        type: constant('retired'),
        current: _ageRange,
        max: _ageRange,
      }),
      object({
        type: constant('notRetired'),
        current: _ageRange,
        retirement: _ageRange,
        max: _ageRange,
      }),
    ),
    (ages: Person['ages']): JSONGuardResult<Person['ages']> => {
      const { current, max } = ages
      if (max < current + 1) {
        return failure('Max age should be at least one year after current age.')
      }
      if (ages.type === 'notRetired') {
        const { retirement } = ages
        if (retirement < current + 1) {
          return failure(
            'Retirement age should be at least one year after current age.',
          )
        }
        if (max < retirement + 1) {
          return failure(
            'Max age should be at least one year after retirement age.',
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
    displayName: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
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
      xAxis: union(constant('person1'), constant('person2')),
    }),
  )

  const year = (people: Params['people'] | null) =>
    union(
      object({ type: constant('now') }),
      chain(
        object({
          type: constant('namedAge'),
          person: union(constant('person1'), constant('person2')),
          age: union(
            constant('lastWorkingYear'),
            constant('retirement'),
            constant('max'),
          ),
        }),
        (x) => {
          if (!people) return success(x)
          let person: Person
          if (x.person === 'person1') {
            person = people.person1
          } else {
            if (!people.withPartner)
              return failure('In terms of partner, but there is no partner.')
            person = people.person2
          }
          if (
            (x.age === 'retirement' || x.age === 'lastWorkingYear') &&
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
          age: chain(number, integer),
        }),
        (x) => {
          if (!people) return success(x)
          if (x.person === 'person2' && !people.withPartner)
            return failure('In terms of partner, but there is no partner.')
          return success(x)
        },
      ),
    )

  const yearRange = (people: Params['people'] | null): JSONGuard<YearRange> => {
    const yr = year(people)
    return union(
      object({ type: constant('startAndEnd'), start: yr, end: yr }),
      object({
        type: constant('startAndNumYears'),
        start: yr,
        numYears: chain(number, integer, gt(0)),
      }),
      object({
        type: constant('endAndNumYears'),
        end: yr,
        numYears: chain(number, integer, gt(0)),
      }),
    )
  }

  const valueForYearRange = (
    people: Params['people'] | null,
  ): JSONGuard<ValueForYearRange> =>
    object({
      // Not trimmed because it won't allow space even temporarily.
      label: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
      yearRange: yearRange(people),
      value: chain(number, gte(0)),
      nominal: boolean,
      id: chain(number, integer, gte(0)),
    })
  const valueForYearRangeArr = (
    people: Params['people'] | null = null,
  ): JSONGuard<ValueForYearRange[]> =>
    array(valueForYearRange(people), MAX_VALUE_FOR_YEAR_RANGE)

  const adjustmentsToSpending = (
    params: Params | null,
  ): JSONGuard<Params['adjustmentsToSpending']> =>
    object({
      tpawAndSPAW: object({
        spendingCeiling: chain(
          nullable(chain(number, gte(0))),
          chain(nullable(chain(number, gte(0))), (x) => {
            if (!params) return success(x)
            if (x === null) return success(x)
            if (
              params.adjustmentsToSpending.tpawAndSPAW.spendingFloor !== null &&
              x < params.adjustmentsToSpending.tpawAndSPAW.spendingFloor
            ) {
              failure('Spending Floor is greater than spending ceiling.')
            }
            return success(x)
          }),
        ),
        spendingFloor: nullable(chain(number, gte(0))),
        legacy: object({
          total: chain(number, gte(0)),
          external: array(
            object({
              label: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
              value: chain(number, gte(0)),
              nominal: boolean,
            }),
            MAX_EXTERNAL_LEGACY_SOURCES,
          ),
        }),
      }),
      extraSpending: object({
        essential: array(
          valueForYearRange(params?.people ?? null),
          MAX_VALUE_FOR_YEAR_RANGE,
        ),
        discretionary: array(
          valueForYearRange(params?.people ?? null),
          MAX_VALUE_FOR_YEAR_RANGE,
        ),
      }),
    })

  const glidePath = (params: Params | null): JSONGuard<GlidePath> =>
    object({
      start: object({ stocks: chain(number, gte(0), lte(1)) }),
      intermediate: array(
        object({
          year: year(params?.people ?? null),
          stocks: chain(number, gte(0), lte(1)),
        }),
        MAX_NUM_YEARS_IN_GLIDE_PATH - 2,
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
      }),
      tpawAndSPAW: object({
        lmp: chain(number, gte(0)),
      }),
      spaw: object({ spendingTilt: among(SPAW_SPENDING_TILT_VALUES) }),
      swr: object({
        withdrawal: union(
          object({
            type: constant('asPercent'),
            percent: chain(number, gte(0), lte(1)),
          }),
          object({
            type: constant('asAmount'),
            amount: chain(number, integer, gte(0)),
          }),
          object({ type: constant('default') }),
        ),
      }),
      spawAndSWR: object({
        allocation: glidePath(params),
      }),
    })

  const returns: JSONGuard<Params['returns']> = object({
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
        type: constant('default'),
        adjust: union(
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
          object({ type: constant('none') }),
        ),
      }),
      object({
        type: constant('fixed'),
        stocks: number,
        bonds: number,
      }),
    ),
  })

  const inflation: JSONGuard<Params['inflation']> = union(
    object({ type: constant('suggested') }),
    object({
      type: constant('manual'),
      value: among(MANUAL_INFLATION_VALUES),
    }),
  )

  const params = (x: Params | null): JSONGuard<Params> =>
    object({
      v: constant(16),
      warnedAbout14to15Converstion: boolean,
      strategy: union(constant('TPAW'), constant('SPAW'), constant('SWR')),
      dialogPosition: dialogPosition(x),
      people,
      currentPortfolioBalance: chain(number, gte(0)),
      futureSavings: valueForYearRangeArr(x?.people ?? null),
      retirementIncome: valueForYearRangeArr(x?.people ?? null),
      adjustmentsToSpending: adjustmentsToSpending(x),
      risk: risk(x),
      returns,
      inflation,
      sampling: union(constant('monteCarlo'), constant('historical')),
      display: object({ alwaysShowAllYears: boolean }),
    })

  const v16Guard: JSONGuard<Params> = chain(params(null), (x) => params(x)(x))

  export const guard: JSONGuard<Params> = (x: unknown) => {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
    if ((x as any).v === 16) {
      return v16Guard(x)
    } else {
      const v15Result = PlanParams15.guard(x)
      if (v15Result.error) return v15Result
      const { dialogMode, ...v15 } = v15Result.value

      const v16: Params = {
        ...v15,
        v: 16,
        dialogPosition: dialogMode ? 'age' : 'done',
      }
      return success(v16Guard(v16).force())
    }
  }
}
