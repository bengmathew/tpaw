import { Validator } from '../Utils/Validator'
import { TPAWParams } from './TPAWParams'
import { TPAWParamsV1, TPAWParamsV1WithoutHistorical } from './TPAWParamsV1'

// ----------- UTILS ---------//
const {number, string, constant, chain, object, union, array, boolean} =
  Validator

const numRangeInclusive = (min: number, max: number) =>
  chain(number(), _geTest(min), _leTest(max))

const numGE = (min: number) => chain(number(), _geTest(min))

const _geTest = (y: number) => (x: number) => {
  if (x < y) throw new Validator.Failed(`Less than ${y}.`)
  return x
}
const _leTest = (y: number) => (x: number) => {
  if (x > y) throw new Validator.Failed(`Greater than ${y}.`)
  return x
}

// ----------- MAIN ---------//

const MAX_AGE = 120
const _ageRange = chain(number(), _geTest(0), _leTest(MAX_AGE))

const _ageValidator = chain(
  object({
    start: _ageRange,
    retirement: _ageRange,
    end: _ageRange,
  }),
  (age): TPAWParams['age'] => {
    const {start, retirement, end} = age
    if (retirement < start) {
      throw new Validator.Failed('Retirement age is earlier than current age.')
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
})

export const tpawParamsV1Validator:Validator<TPAWParamsV1WithoutHistorical> = object({
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
  spendingCeiling: union(constant(null), numGE(0)),
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
})
