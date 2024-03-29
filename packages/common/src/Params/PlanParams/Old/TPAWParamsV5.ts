import _ from 'lodash'
import { Validator } from './Validator'

export namespace TPAWParamsV5 {
  export const MAX_LABEL_LENGTH = 150
  export const MAX_AGE = 120

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

  export type Params = {
    v: 5
    people: People
    returns: {
      expected: {
        stocks: number
        bonds: number
      }
      historical: {
        adjust:
          | {type: 'by'; stocks: number; bonds: number}
          | {type: 'to'; stocks: number; bonds: number}
      }
    }
    inflation: number
    targetAllocation: {
      regularPortfolio: {stocks: number}
      legacyPortfolio: {stocks: number}
    }
    spendingCeiling: number | null
    spendingFloor: number | null
    scheduledWithdrawalGrowthRate: number
    savingsAtStartOfStartYear: number
    savings: ValueForYearRange[]
    retirementIncome: ValueForYearRange[]
    withdrawals: {
      fundedByBonds: ValueForYearRange[]
      fundedByRiskPortfolio: ValueForYearRange[]
    }
    legacy: {
      total: number
      external: LabeledAmount[]
    }
  }
  export type ParamsWithoutHistorical = Omit<Params, 'returns'> & {
    returns: Omit<Params['returns'], 'historical'>
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

  // ----------- VALIDATOR  ---------//

  const _ageRange = chain(number(), _geTest(0), _leTest(TPAWParamsV5.MAX_AGE))

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
    (ages): TPAWParamsV5.Person['ages'] => {
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

  const _person: Validator<TPAWParamsV5.Person> = object({
    ages: _ages,
    displayName: union(
      strBoundedTrimmed(TPAWParamsV5.MAX_LABEL_LENGTH),
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
    label: union(strBounded(TPAWParamsV5.MAX_LABEL_LENGTH), constant(null)),
    yearRange: _yearRange,
    value: numGE(0),
    nominal: boolean(),
    id: numIntNonNeg(),
  })

  export const validator: Validator<TPAWParamsV5.ParamsWithoutHistorical> = chain(
    object({
      v: constant(5),
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
        expected: object({
          stocks: numRangeInclusive(-0.01, 0.1),
          bonds: numRangeInclusive(-0.01, 0.1),
        }),
      }),

      inflation: numRangeInclusive(-0.01, 0.1),
      targetAllocation: object({
        regularPortfolio: object({
          stocks: numRangeInclusive(0, 1),
        }),
        legacyPortfolio: object({
          stocks: numRangeInclusive(0, 1),
        }),
      }),
      scheduledWithdrawalGrowthRate: numRangeInclusive(-0.03, 0.03),
      savingsAtStartOfStartYear: numGE(0),
      savings: array(_valueForYearRange),
      retirementIncome: array(_valueForYearRange),
      spendingCeiling: union(constant(null), numGE(0)),
      spendingFloor: union(constant(null), numGE(0)),
      withdrawals: object({
        fundedByBonds: array(_valueForYearRange),
        fundedByRiskPortfolio: array(_valueForYearRange),
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

      const checkYear = (year: TPAWParamsV5.Year, prefix: string) => {
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
          if (year.age === 'retirement' && person.ages.type === 'retired') {
            throw new Validator.Failed(
              `${prefix} is in terms retirement age of ${year.person}, but ${year.person} is already retired.`
            )
          }
        }
      }

      const checkYearRange =
        (desc: string) =>
        ({yearRange, label}: TPAWParamsV5.ValueForYearRange, i: number) => {
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
      x.withdrawals.fundedByBonds.forEach(
        checkYearRange('withdrawals funded by bonds')
      )
      x.withdrawals.fundedByRiskPortfolio.forEach(
        checkYearRange('withdrawals funded by risk portfolio')
      )
      return x
    }
  )
}
