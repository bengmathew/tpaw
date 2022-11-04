// import {getDefaultParams} from './DefaultPlanParams'
// import {TPAWParamsV13} from './TPAWParamsOld/TPAWParamsV13'
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
  intersection,
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
import { fGet } from '../Utils'
import { getDefaultPlanParams } from './DefaultPlanParams'
import { TPAWParamsV1WithoutHistorical } from './Old/TPAWParamsV1'
import { TPAWParamsV10 } from './Old/TPAWParamsV10'
import { TPAWParamsV11 } from './Old/TPAWParamsV11'
import { TPAWParamsV12 } from './Old/TPAWParamsV12'
import { TPAWParamsV13 } from './Old/TPAWParamsV13'
import { tpawParamsV1Validator } from './Old/TPAWParamsV1Validator'
import { TPAWParamsV2WithoutHistorical } from './Old/TPAWParamsV2'
import { tpawParamsV2Validator } from './Old/TPAWParamsV2Validator'
import { TPAWParamsV3WithoutHistorical } from './Old/TPAWParamsV3'
import { tpawParamsV3Validator } from './Old/TPAWParamsV3Validator'
import { TPAWParamsV4 } from './Old/TPAWParamsV4'
import { TPAWParamsV5 } from './Old/TPAWParamsV5'
import { TPAWParamsV6 } from './Old/TPAWParamsV6'
import { TPAWParamsV7 } from './Old/TPAWParamsV7'
import { TPAWParamsV8 } from './Old/TPAWParamsV8'
import { TPAWParamsV9 } from './Old/TPAWParamsV9'
import { Validator } from './Old/Validator'

export namespace PlanParams14 {
  export const MAX_LABEL_LENGTH = 150
  export const MAX_AGE = 120
  export const MAX_NUM_YEARS_IN_GLIDE_PATH = 1000
  export const MAX_VALUE_FOR_YEAR_RANGE = 100
  export const MAX_EXTERNAL_LEGACY_SOURCES = 100

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

  export type TPAWRiskLevel =
    | 'riskLevel-1'
    | 'riskLevel-2'
    | 'riskLevel-3'
    | 'riskLevel-4'

  export type TPAWRisk = {
    tpaw: {
      allocation: GlidePath
      allocationForLegacy: { stocks: number }
    }
    tpawAndSPAW: {
      spendingCeiling: number | null
      spendingFloor: number | null
      spendingTilt: number
      lmp: number
    }
  }

  export type Params = {
    v: 14
    strategy: 'TPAW' | 'SPAW' | 'SWR'
    dialogMode: boolean

    // Basic Inputs
    people: People
    currentPortfolioBalance: number
    futureSavings: ValueForYearRange[]
    retirementIncome: ValueForYearRange[]

    // Spending Goals
    extraSpending: {
      essential: ValueForYearRange[]
      discretionary: ValueForYearRange[]
    }
    legacy: {
      tpawAndSPAW: {
        total: number
        external: LabeledAmount[]
      }
    }

    // Risk
    // useTPAWPreset should be true only if strategy === 'TPAW'
    risk: ({ useTPAWPreset: true } | ({ useTPAWPreset: false } & TPAWRisk)) & {
      tpawPreset: TPAWRiskLevel
      customTPAWPreset: TPAWRisk | null
      savedTPAWPreset: TPAWRisk | null
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

  // ----------- VALIDATOR  ---------//

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

  const extraSpending = (
    people: Params['people'] | null = null,
  ): JSONGuard<Params['extraSpending']> =>
    object({
      essential: array(valueForYearRange(people), MAX_VALUE_FOR_YEAR_RANGE),
      discretionary: array(valueForYearRange(people), MAX_VALUE_FOR_YEAR_RANGE),
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

  const tpawRisk = (params: Params | null): JSONGuard<TPAWRisk> =>
    object(
      {
        tpaw: object({
          allocation: glidePath(params),
          allocationForLegacy: object({
            stocks: chain(number, gte(0), lte(1)),
          }),
        }),
        tpawAndSPAW: chain(
          object({
            spendingCeiling: nullable(chain(number, gte(0))),
            spendingFloor: nullable(chain(number, gte(0))),
            spendingTilt: chain(number, gte(-0.03), lte(0.03)),
            lmp: chain(number, gte(0)),
          }),
          (x) =>
            x.spendingCeiling !== null &&
            (x.spendingFloor ?? 0) > x.spendingCeiling
              ? failure('Spending Floor is greater than spending ceiling.')
              : success(x),
        ),
      },
      'extraKeysOk',
    )

  const risk = (params: Params | null): JSONGuard<Params['risk']> =>
    intersection(
      union(
        object(
          {
            useTPAWPreset: chain(constant(true), (x) =>
              !params || params.strategy === 'TPAW'
                ? success(x)
                : failure('useTPAWPreset is true but strategy is not TPAW.'),
            ),
          },
          'extraKeysOk',
        ),
        intersection(
          object({ useTPAWPreset: constant(false) }, 'extraKeysOk'),
          tpawRisk(params),
        ),
      ),
      object(
        {
          tpawPreset: union(
            constant('riskLevel-1'),
            constant('riskLevel-2'),
            constant('riskLevel-3'),
            constant('riskLevel-4'),
          ),
          customTPAWPreset: nullable(tpawRisk(params)),
          savedTPAWPreset: nullable(tpawRisk(params)),
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
        },
        'extraKeysOk',
      ),
    )

  const returns: JSONGuard<Params['returns']> = object({
    expected: union(
      object({ type: constant('suggested') }),
      object({ type: constant('oneOverCAPE') }),
      object({ type: constant('regressionPrediction') }),
      object({ type: constant('historical') }),
      object({
        type: constant('manual'),
        stocks: chain(number, gte(-0.01), lte(0.1)),
        bonds: chain(number, gte(-0.01), lte(0.1)),
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

  const legacy: JSONGuard<Params['legacy']> = object({
    tpawAndSPAW: object({
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
  })

  const inflation: JSONGuard<Params['inflation']> = union(
    object({ type: constant('suggested') }),
    object({
      type: constant('manual'),
      value: chain(number, gte(-0.01), lte(0.1)),
    }),
  )

  const params = (x: Params | null): JSONGuard<Params> =>
    object({
      v: constant(14),
      strategy: union(constant('TPAW'), constant('SPAW'), constant('SWR')),
      dialogMode: boolean,
      people,
      currentPortfolioBalance: chain(number, gte(0)),
      futureSavings: valueForYearRangeArr(x?.people ?? null),
      retirementIncome: valueForYearRangeArr(x?.people ?? null),
      extraSpending: extraSpending(x?.people ?? null),
      legacy,
      risk: risk(x),
      returns,
      inflation,
      sampling: union(constant('monteCarlo'), constant('historical')),
      display: object({ alwaysShowAllYears: boolean }),
    })

  const v14Guard: JSONGuard<Params> = chain(params(null), (x) => params(x)(x))

  export const guard: JSONGuard<Params> = (x: unknown) => {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
    if ((x as any).v === 14) {
      return v14Guard(x)
    } else {
      const v13Result = v13Guard(x)
      if (v13Result.error) return v13Result
      const v13 = v13Result.value
      const v14: Params = {
        ...v13,
        v: 14,
        risk:
          v13.risk.tpawPreset === 'custom'
            ? {
                useTPAWPreset: false,
                tpawPreset: getDefaultPlanParams().risk.tpawPreset,
                customTPAWPreset: v13.risk.customTPAWPreset,
                savedTPAWPreset: null,
                tpaw: _.cloneDeep(fGet(v13.risk.customTPAWPreset).tpaw),
                tpawAndSPAW: _.cloneDeep(
                  fGet(v13.risk.customTPAWPreset).tpawAndSPAW,
                ),
                swr: v13.risk.swr,
                spawAndSWR: v13.risk.spawAndSWR,
              }
            : {
                ...v13.risk,
                tpawPreset: v13.risk.tpawPreset,
                savedTPAWPreset: null,
              },
      }
      return success(v14Guard(v14).force())
    }
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const v13Guard: JSONGuard<TPAWParamsV13.Params> = (parsed: any) => {
  try {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
    const version = 'v' in parsed ? parsed.v : 1
    if (typeof version !== 'number') return failure('Version is not a number.')
    if (version > 13 || version < 0) return failure('Invalid version number.')

    const v1 = version === 1 ? tpawParamsV1Validator(parsed) : null
    const v2 =
      version === 2 ? tpawParamsV2Validator(parsed) : v1 ? _v1ToV2(v1) : null

    const v3 =
      version === 3 ? tpawParamsV3Validator(parsed) : v2 ? _v2ToV3(v2) : null
    const v4 =
      version === 4 ? TPAWParamsV4.validator(parsed) : v3 ? _v3ToV4(v3) : null
    const v5 =
      version === 5 ? TPAWParamsV5.validator(parsed) : v4 ? _v4ToV5(v4) : null

    const v6 =
      version === 6
        ? TPAWParamsV6.validator(parsed)
        : v5
        ? TPAWParamsV6.fromV5(v5)
        : null

    const v7 =
      version === 7
        ? TPAWParamsV7.validator(parsed)
        : v6
        ? TPAWParamsV7.fromV6(v6)
        : null

    const v8 =
      version === 8
        ? TPAWParamsV8.validator(parsed)
        : v7
        ? TPAWParamsV8.fromV7(v7)
        : null

    const v9 =
      version === 9
        ? TPAWParamsV9.validator(parsed)
        : v8
        ? TPAWParamsV9.fromV8(v8)
        : null

    const v10 =
      version === 10
        ? TPAWParamsV10.validator(parsed)
        : v9
        ? TPAWParamsV10.fromV9(v9)
        : null

    const v11 =
      version === 11
        ? TPAWParamsV11.validator(parsed)
        : v10
        ? TPAWParamsV11.fromV10(v10)
        : null

    const v12 =
      version === 12
        ? TPAWParamsV12.validator(parsed)
        : v11
        ? TPAWParamsV12.fromV11(v11)
        : null

    return success(
      version === 13
        ? TPAWParamsV13.validator(parsed)
        : TPAWParamsV13.fromV12(fGet(v12)),
    )
  } catch (e) {
    if (e instanceof Validator.Failed) {
      return failure(e.fullMessage)
    } else {
      throw e
    }
  }
}

const _v1ToV2 = (
  v1: TPAWParamsV1WithoutHistorical,
): TPAWParamsV2WithoutHistorical => {
  type ValueForYearRange = TPAWParamsV2WithoutHistorical['savings'][number]
  const savings: ValueForYearRange[] = []
  const retirementIncome: ValueForYearRange[] = []
  v1.savings.forEach((x) => {
    const start = _numericYear(v1, x.yearRange.start)
    const end = _numericYear(v1, x.yearRange.end)
    if (start < v1.age.retirement && end >= v1.age.retirement) {
      savings.push({
        ...x,
        yearRange: { ...x.yearRange, end: 'lastWorkingYear' as const },
      })
      retirementIncome.push({
        ...x,
        yearRange: { ...x.yearRange, start: 'retirement' as const },
      })
    } else {
      start < v1.age.retirement ? savings.push(x) : retirementIncome.push(x)
    }
  })

  return {
    v: 2,
    ...v1,
    savings,
    retirementIncome,
  }
}

const _v2ToV3 = (
  v2: TPAWParamsV2WithoutHistorical,
): TPAWParamsV3WithoutHistorical => {
  return {
    ..._.cloneDeep(v2),
    v: 3,
    spendingFloor: null,
  }
}
const _v3ToV4 = (
  v3: TPAWParamsV3WithoutHistorical,
): TPAWParamsV4.ParamsWithoutHistorical => {
  const { retirementIncome, withdrawals, ...rest } = _.cloneDeep(v3)
  const addId = (
    x: TPAWParamsV3WithoutHistorical['savings'][number],
    id: number,
  ): TPAWParamsV4.ValueForYearRange => ({ ...x, id })
  retirementIncome
  return {
    ...rest,
    v: 4,
    retirementIncome: retirementIncome.map(addId),
    savings: retirementIncome.map(addId),
    withdrawals: {
      fundedByBonds: withdrawals.fundedByBonds.map(addId),
      fundedByRiskPortfolio: withdrawals.fundedByRiskPortfolio.map(addId),
    },
  }
}
const _v4ToV5 = (
  v4: TPAWParamsV4.ParamsWithoutHistorical,
): TPAWParamsV5.ParamsWithoutHistorical => {
  const { age, savings, retirementIncome, withdrawals, ...rest } = v4

  const year = (year: TPAWParamsV4.YearRangeEdge): TPAWParamsV5.Year =>
    year === 'start'
      ? { type: 'now' }
      : typeof year === 'number'
      ? { type: 'numericAge', person: 'person1', age: year }
      : {
          type: 'namedAge',
          person: 'person1',
          age: year === 'end' ? 'max' : year,
        }

  const valueForYearRange = ({
    yearRange,
    ...rest
  }: TPAWParamsV4.ValueForYearRange): TPAWParamsV5.ValueForYearRange => ({
    yearRange: {
      type: 'startAndEnd',
      start: year(yearRange.start),
      end: year(yearRange.end),
    },
    ...rest,
  })

  const result: TPAWParamsV5.ParamsWithoutHistorical = {
    ...rest,
    v: 5,
    people: {
      withPartner: false,
      person1: {
        ages:
          age.start === age.retirement
            ? {
                type: 'retired',
                current: age.start,
                max: age.end,
              }
            : {
                type: 'notRetired',
                current: age.start,
                retirement: age.retirement,
                max: age.end,
              },
        displayName: null,
      },
    },
    savings: savings.map(valueForYearRange),
    retirementIncome: retirementIncome.map(valueForYearRange),
    withdrawals: {
      fundedByBonds: withdrawals.fundedByBonds.flatMap(valueForYearRange),
      fundedByRiskPortfolio:
        withdrawals.fundedByRiskPortfolio.flatMap(valueForYearRange),
    },
  }

  TPAWParamsV5.validator(result)
  return result
}

const _numericYear = (
  { age }: { age: { start: number; retirement: number; end: number } },
  x: TPAWParamsV4.YearRangeEdge,
) =>
  x === 'start'
    ? age.start
    : x === 'lastWorkingYear'
    ? age.retirement - 1
    : x === 'retirement'
    ? age.retirement
    : x === 'end'
    ? age.end
    : x
