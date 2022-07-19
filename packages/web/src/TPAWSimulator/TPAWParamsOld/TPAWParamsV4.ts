import _ from 'lodash'
import { Validator } from '../../Utils/Validator'

export namespace TPAWParamsV4 {
  export type YearRangeFixedEdge =
    | 'start'
    | 'lastWorkingYear'
    | 'retirement'
    | 'end'

  export type YearRangeEdge = number | YearRangeFixedEdge

  export type YearRange = {start: YearRangeEdge; end: YearRangeEdge}
  export type ValueForYearRange = {
    label: string | null
    yearRange: YearRange
    value: number
    nominal: boolean
    id: number
  }

  export type Params = {
    v: 4
    age: {
      start: number
      retirement: number
      end: number
    }
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
      external: {label: string | null; value: number; nominal: boolean}[]
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

  const numNonNegInteger = () => chain(number(), _intTest, _geTest(0))

  const _intTest = (x: number) => {
    if (!_.isInteger(x)) throw new Validator.Failed('Not an integer.')
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

  // ----------- VALIDATOR ---------//

  export const MAX_AGE_V4 = 120
  const _ageRange = chain(number(), _geTest(0), _leTest(MAX_AGE_V4))

  const _ageValidator = chain(
    object({
      start: _ageRange,
      retirement: _ageRange,
      end: _ageRange,
    }),
    (age): Params['age'] => {
      const {start, retirement, end} = age
      if (retirement < start) {
        throw new Validator.Failed(
          'Retirement age is earlier than current age.'
        )
      }
      if (end < start + 1) {
        throw new Validator.Failed(
          'Max age should be at least one year after current age.'
        )
      }
      if (retirement >= end) {
        throw new Validator.Failed(
          'Max age should be at least one year after retirement age.'
        )
      }
      return age
    }
  )
  const _yearRangeEdge = union(
    number(),
    constant('start'),
    constant('lastWorkingYear'),
    constant('retirement'),
    constant('end')
  )
  const _valueForYearRangeValidator = object({
    label: union(string(), constant(null)),
    yearRange: object({start: _yearRangeEdge, end: _yearRangeEdge}),
    value: numGE(0),
    nominal: boolean(),
    id: numNonNegInteger(),
  })

  export const validator: Validator<ParamsWithoutHistorical> =
    chain(
      object({
        v: constant(4),
        age: _ageValidator,
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
        savings: array(_valueForYearRangeValidator),
        retirementIncome: array(_valueForYearRangeValidator),
        spendingCeiling: union(constant(null), numGE(0)),
        spendingFloor: union(constant(null), numGE(0)),
        withdrawals: object({
          fundedByBonds: array(_valueForYearRangeValidator),
          fundedByRiskPortfolio: array(_valueForYearRangeValidator),
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
            'Spending Floor is greater than spending ceiling'
          )
        }
        return x
      }
    )
}
