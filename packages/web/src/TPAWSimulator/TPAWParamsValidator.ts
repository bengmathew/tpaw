import {Validator} from '../Utils/Validator'
import {TPAWParams} from './TPAWParams'

// ----------- UTILS ---------//
// const strAsNum = () =>
//   Validator.chain(Validator.string(), (x: string) => {
//     if (
//       !x.match(/[^0123456789.-]/) &&
//       x.indexOf('.') === x.lastIndexOf('.') &&
//       [0, -1].includes(x.indexOf('-')) &&
//       [0, -1].includes(x.lastIndexOf('-')) &&
//       x.length > 0 &&
//       !isNaN(parseFloat(x))
//     ) {
//       return parseFloat(x)
//     }
//     throw new Validator.Failed('Not a number.')
//   })

// const strAsOrNum = () => Validator.union(Validator.number(), strAsNum())

const numRangeInclusive = (min: number, max: number) =>
  Validator.chain(Validator.number(), _geTest(min), _leTest(max))

const numGE = (min: number) => Validator.chain(Validator.number(), _geTest(min))

const _geTest = (y: number) => (x: number) => {
  if (x < y) throw new Validator.Failed(`Less than ${y}.`)
  return x
}
const _leTest = (y: number) => (x: number) => {
  if (x > y) throw new Validator.Failed(`Greater than ${y}.`)
  return x
}

// ----------- MAIN ---------//

export const MAX_AGE = 120
const _ageRange = Validator.chain(Validator.number(), _geTest(0), _leTest(MAX_AGE))

const _ageValidator = Validator.chain(
  Validator.object({
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
const _yearRangeEdge = Validator.union(
  Validator.number(),
  Validator.constant('start'),
  Validator.constant('lastWorkingYear'),
  Validator.constant('retirement'),
  Validator.constant('end')
)
const _valueForYearRangeValidator = Validator.object({
  label: Validator.union(Validator.string(), Validator.constant(null)),
  yearRange: Validator.object({start: _yearRangeEdge, end: _yearRangeEdge}),
  value: numGE(0),
  nominal: Validator.boolean(),
})

export const tpawParamsValidator = Validator.object({
  age: _ageValidator,
  returns: Validator.object({
    expected: Validator.object({
      stocks: numRangeInclusive(-0.01, 0.1),
      bonds: numRangeInclusive(-0.01, 0.1),
    }),
  }),

  inflation: numRangeInclusive(-0.01, 0.1),
  targetAllocation: Validator.object({
    regularPortfolio: Validator.object({
      stocks: numRangeInclusive(0, 1),
    }),
    legacyPortfolio: Validator.object({
      stocks: numRangeInclusive(0, 1),
    }),
  }),
  scheduledWithdrawalGrowthRate: numRangeInclusive(-0.03, 0.03),
  savingsAtStartOfStartYear: numGE(0),
  savings: Validator.array(_valueForYearRangeValidator),
  spendingCeiling: Validator.union(Validator.constant(null), numGE(0)),
  withdrawals: Validator.object({
    fundedByBonds: Validator.array(_valueForYearRangeValidator),
    fundedByRiskPortfolio: Validator.array(_valueForYearRangeValidator),
  }),
  legacy: Validator.object({
    total: numGE(0),
    external: Validator.array(
      Validator.object({
        label: Validator.union(Validator.string(), Validator.constant(null)),
        value: numGE(0),
        nominal: Validator.boolean(),
      })
    ),
  }),
})
