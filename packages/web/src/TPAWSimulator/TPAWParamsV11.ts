import _ from 'lodash'
import {Validator} from '../Utils/Validator'
import {TPAWParamsV10} from './TPAWParamsOld/TPAWParamsV10'

export namespace TPAWParamsV11 {
  export const MAX_LABEL_LENGTH = 150
  export const MAX_AGE = 120
  export const MAX_NUM_YEARS_IN_GLIDE_PATH = 1000

  export type Person = {
    ages:
      | {type: 'retired'; current: number; max: number}
      | {type: 'notRetired'; current: number; retirement: number; max: number}
    displayName: string | null
  }

  export type People =
    | {withPartner: false; person1: Person}
    | {
        withPartner: true
        person2: Person
        person1: Person
        withdrawalStart: 'person1' | 'person2'
        xAxis: 'person1' | 'person2'
      }

  export type Year =
    | {type: 'now'}
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
    | {type: 'startAndEnd'; start: Year; end: Year}
    | {type: 'startAndNumYears'; start: Year; numYears: number}
    | {type: 'endAndNumYears'; end: Year; numYears: number}
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
    start: {stocks: number}
    intermediate: {year: Year; stocks: number}[]
    end: {stocks: number}
  }

  export type Params = {
    v: 10
    strategy: 'TPAW' | 'SPAW' | 'SWR'
    people: People
    returns: {
      expected:
        | {type: 'suggested'}
        | {type: 'oneOverCAPE'}
        | {type: 'regressionPrediction'}
        | {type: 'historical'}
        | {type: 'manual'; stocks: number; bonds: number}
      historical:
        | {
            type: 'default'
            adjust:
              | {type: 'by'; stocks: number; bonds: number}
              | {type: 'to'; stocks: number; bonds: number}
              | {type: 'toExpected'}
              | {type: 'none'}
          }
        | {type: 'fixed'; stocks: number; bonds: number}
    }
    inflation: {type: 'suggested'} | {type: 'manual'; value: number}
    targetAllocation: {
      regularPortfolio: {
        forTPAW: {stocks: number}
        forSPAWAndSWR: GlidePath
      }
      legacyPortfolio: {stocks: number}
    }
    spendingCeiling: number | null
    spendingFloor: number | null
    scheduledWithdrawalGrowthRate: number
    savingsAtStartOfStartYear: number
    savings: ValueForYearRange[]
    retirementIncome: ValueForYearRange[]
    withdrawals: {
      lmp: number
      essential: ValueForYearRange[]
      discretionary: ValueForYearRange[]
    }
    swrWithdrawal:
      | {type: 'asPercent'; percent: number}
      | {type: 'asAmount'; amount: number}
      | {type: 'default'}
    legacy: {
      total: number
      external: LabeledAmount[]
    }
    sampling: 'monteCarlo' | 'historical'
    display: {
      alwaysShowAllYears: boolean
    }
  }

  const {number, string, constant, chain, object, union, array, boolean} =
    Validator

  const numRangeInclusive = (min: number, max: number) =>
    chain(number(), _geTest(min), _leTest(max))
  const numGE = (min: number) => chain(number(), _geTest(min))
  const numInt = () => chain(number(), _intTest)
  const numIntNonNeg = () => chain(numInt(), _geTest(0))
  const numIntPos = () => chain(numInt(), _geTest(1))

  const strBounded = (maxLength: number) =>
    chain(string(), _boundedStringTest(maxLength))
  const strBoundedTrimmed = (maxLength: number) =>
    chain(strBounded(maxLength), _trimmedTest)

  const sizedArray = <T>(test: Validator<T>, min: number, max: number) =>
    chain(array(test), x => {
      if (x.length < min)
        throw new Validator.Failed(`Less than ${min} elements.`)
      if (x.length > max)
        throw new Validator.Failed(`Greater than ${max} elements.`)
      return x
    })

  const _intTest = (x: number) => {
    if (!_.isInteger(x)) throw new Validator.Failed('Not an integer.')
    return x
  }
  const _boundedStringTest = (bound: number) => (x: string) => {
    if (x.length > bound) throw new Validator.Failed('Too large.')
    return x
  }
  const _trimmedTest = (x: string) => {
    if (x.trim().length !== x.length) throw new Validator.Failed('Not trimmed.')
    return x
  }

  const _geTest = (y: number) => (x: number) => {
    if (x < y) throw new Validator.Failed(`Less than ${y}.`)
    return x
  }
  const _leTest = (y: number) => (x: number) => {
    if (x > y) throw new Validator.Failed(`Greater than ${y}.`)
    return x
  }

  // ---------- FROM V10 ------------//
  export const fromV10 = (params: TPAWParamsV10.Params): Params => {
    const result: Params = {
      ...params,
      inflation: {type: 'manual', value: params.inflation},
      v: 10,
    }
    validator(result)
    return result
  }

  // ----------- VALIDATOR  ---------//

  const _ageRange = chain(number(), _geTest(0), _leTest(MAX_AGE))

  const _ages = chain(
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
      })
    ),
    (ages): Person['ages'] => {
      const {current, max} = ages
      if (max < current + 1) {
        throw new Validator.Failed(
          'Max age should be at least one year after current age.'
        )
      }
      if (ages.type === 'notRetired') {
        const {retirement} = ages
        if (retirement < current + 1) {
          throw new Validator.Failed(
            'Retirement age should be at least one year after current age.'
          )
        }
        if (max < retirement + 1) {
          throw new Validator.Failed(
            'Max age should be at least one year after retirement age.'
          )
        }
      }
      return ages
    }
  )

  const _person: Validator<Person> = object({
    ages: _ages,
    displayName: union(
      strBoundedTrimmed(MAX_LABEL_LENGTH),
      constant(null)
    ),
  })

  const _year = union(
    object({type: constant('now')}),
    object({
      type: constant('namedAge'),
      person: union(constant('person1'), constant('person2')),
      age: union(
        constant('lastWorkingYear'),
        constant('retirement'),
        constant('max')
      ),
    }),
    object({
      type: constant('numericAge'),
      person: union(constant('person1'), constant('person2')),
      age: numInt(),
    })
  )

  const _yearRange = union(
    object({type: constant('startAndEnd'), start: _year, end: _year}),
    object({
      type: constant('startAndNumYears'),
      start: _year,
      numYears: numIntPos(),
    }),
    object({
      type: constant('endAndNumYears'),
      end: _year,
      numYears: numIntPos(),
    })
  )

  const _valueForYearRange = object({
    // Not trimmed because it won't allow space even temporarily.
    label: union(strBounded(MAX_LABEL_LENGTH), constant(null)),
    yearRange: _yearRange,
    value: numGE(0),
    nominal: boolean(),
    id: numIntNonNeg(),
  })

  const _glidePath: Validator<GlidePath> = object({
    start: object({stocks: numRangeInclusive(0, 1)}),
    intermediate: sizedArray(
      object({year: _year, stocks: numRangeInclusive(0, 1)}),
      0,
      MAX_NUM_YEARS_IN_GLIDE_PATH - 2
    ),
    end: object({stocks: numRangeInclusive(0, 1)}),
  })

  export const validator: Validator<Params> = chain(
    object({
      v: constant(10),
      strategy: union(constant('TPAW'), constant('SPAW'), constant('SWR')),
      people: union(
        object({
          withPartner: constant(false),
          person1: _person,
        }),
        object({
          withPartner: constant(true),
          person2: _person,
          person1: _person,
          withdrawalStart: union(constant('person1'), constant('person2')),
          xAxis: union(constant('person1'), constant('person2')),
        })
      ),
      returns: object({
        expected: union(
          object({type: constant('suggested')}),
          object({type: constant('oneOverCAPE')}),
          object({type: constant('regressionPrediction')}),
          object({type: constant('historical')}),
          object({
            type: constant('manual'),
            stocks: numRangeInclusive(-0.01, 0.1),
            bonds: numRangeInclusive(-0.01, 0.1),
          })
        ),
        historical: union(
          object({
            type: constant('default'),
            adjust: union(
              object({type: constant('to'), stocks: number(), bonds: number()}),
              object({type: constant('by'), stocks: number(), bonds: number()}),
              object({type: constant('toExpected')}),
              object({type: constant('none')})
            ),
          }),
          object({type: constant('fixed'), stocks: number(), bonds: number()})
        ),
      }),

      inflation: union(
        object({type: constant('suggested')}),
        object({type: constant('manual'), value: numRangeInclusive(-0.01, 0.1)})
      ),
      targetAllocation: object({
        regularPortfolio: object({
          forTPAW: object({
            stocks: numRangeInclusive(0, 1),
          }),
          forSPAWAndSWR: _glidePath,
        }),
        legacyPortfolio: object({
          stocks: numRangeInclusive(0, 1),
        }),
      }),
      swrWithdrawal: union(
        object({type: constant('asPercent'), percent: numRangeInclusive(0, 1)}),
        object({type: constant('asAmount'), amount: numIntNonNeg()}),
        object({type: constant('default')})
      ),
      scheduledWithdrawalGrowthRate: numRangeInclusive(-0.03, 0.03),
      savingsAtStartOfStartYear: numGE(0),
      savings: array(_valueForYearRange),
      retirementIncome: array(_valueForYearRange),
      spendingCeiling: union(constant(null), numGE(0)),
      spendingFloor: union(constant(null), numGE(0)),
      withdrawals: object({
        lmp: numGE(0),
        essential: array(_valueForYearRange),
        discretionary: array(_valueForYearRange),
      }),
      legacy: object({
        total: numGE(0),
        external: array(
          object({
            label: union(string(), constant(null)),
            value: numGE(0),
            nominal: boolean(),
          })
        ),
      }),
      sampling: union(constant('monteCarlo'), constant('historical')),
      display: object({alwaysShowAllYears: boolean()}),
    }),
    x => {
      if (
        x.spendingCeiling !== null &&
        (x.spendingFloor ?? 0) > x.spendingCeiling
      ) {
        throw new Validator.Failed(
          'Spending Floor is greater than spending ceiling.'
        )
      }

      const checkYear = (year: Year, prefix: string) => {
        if (year.type === 'namedAge' || year.type === 'numericAge') {
          let person: Person
          if (year.person === 'person1') {
            person = x.people.person1
          } else {
            if (!x.people.withPartner) {
              throw new Validator.Failed(
                `${prefix} is in terms of the age of an unspecified person.`
              )
            }
            person = x.people.person2
          }
          if (
            (year.age === 'retirement' || year.age === 'lastWorkingYear') &&
            person.ages.type === 'retired'
          ) {
            throw new Validator.Failed(
              `${prefix} is in terms retirement age of ${year.person}, but ${year.person} is already retired.`
            )
          }
        }
      }

      const checkYearRange =
        (desc: string) =>
        ({yearRange, label}: ValueForYearRange, i: number) => {
          if (
            yearRange.type === 'startAndEnd' ||
            yearRange.type === 'startAndNumYears'
          ) {
            checkYear(
              yearRange.start,
              `Starting year of ${desc} entry ${label ?? `at index ${i}`}`
            )
          }
          if (
            yearRange.type === 'startAndEnd' ||
            yearRange.type === 'endAndNumYears'
          ) {
            checkYear(
              yearRange.end,
              `Ending year of ${desc} ${label ?? `at index ${i}`}`
            )
          }
        }
      x.savings.forEach(checkYearRange('savings'))
      x.retirementIncome.forEach(checkYearRange('retirement income'))
      x.withdrawals.essential.forEach(
        checkYearRange('withdrawals funded by bonds')
      )
      x.withdrawals.discretionary.forEach(
        checkYearRange('withdrawals funded by risk portfolio')
      )

      const checkGlidePathYears = (glidePath: GlidePath['intermediate']) => {
        glidePath.forEach((x, i) =>
          checkYear(x.year, `Year at entry ${i} in the static glide path`)
        )
      }
      checkGlidePathYears(
        x.targetAllocation.regularPortfolio.forSPAWAndSWR.intermediate
      )
      return x
    }
  )
}
