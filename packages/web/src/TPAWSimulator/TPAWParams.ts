import _ from 'lodash'
import {numOfYears} from '../Utils/NumOfYears'
import {historicalReturns} from './HistoricalReturns'

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
}

export type TPAWParams = {
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
  scheduledWithdrawalGrowthRate: number
  savingsAtStartOfStartYear: number
  savings: ValueForYearRange[]
  withdrawals: {
    fundedByBonds: ValueForYearRange[]
    fundedByRiskPortfolio: ValueForYearRange[]
  }
  legacy: {
    total: number
    external: {label: string | null; value: number; nominal: boolean}[]
  }
}

export type TPAWParamsProcessed2 = ReturnType<typeof processTPAWParams>
export function processTPAWParams(
  params: TPAWParams,
  randomIndexesIntoHistoricalReturnsByYear?: (year: number) => number
) {
  const {inflation, ...paramsWithoutInflation} = params
  const _completeAllocation = ({stocks}: {stocks: number}) => ({
    stocks,
    bonds: 1 - stocks,
  })
  return {
    ...paramsWithoutInflation,
    returns: _processReturnsParams(
      params,
      randomIndexesIntoHistoricalReturnsByYear
    ),
    targetAllocation: {
      legacyPortfolio: _completeAllocation(
        params.targetAllocation.legacyPortfolio
      ),
      regularPortfolio: _completeAllocation(
        params.targetAllocation.regularPortfolio
      ),
    },
    byYear: _processByYearParams(params),
    legacy: (() => {
      const {total} = params.legacy
      const external = _.sum(
        params.legacy.external.map(x =>
          _realValue(x, params, numOfYears(params.age))
        )
      )
      const target = Math.max(total - external, 0)
      return {totall: total, external, target}
    })(),
  }
}

function _processByYearParams(params: TPAWParams) {
  const {age, savings, withdrawals} = params
  const byYear = _.times(numOfYears(age), () => ({
    savings: 0,
    withdrawals: {fundedByBonds: 0, fundedByRiskPortfolio: 0},
  }))

  const exec = (
    values: ValueForYearRange[],
    updater: (target: typeof byYear[0], value: number) => void
  ) => {
    values.forEach(({yearRange, value, nominal}) => {
      const start = Math.max(numericYear(params, yearRange.start), age.start)
      const end = Math.min(numericYear(params, yearRange.end), age.end)
      const offset = start - age.start
      _.times(Math.max(0, end + 1 - start), () => value).forEach((v, i) => {
        const yearsFromNow = i + offset
        const value = nominal
          ? v / Math.pow(1 + params.inflation, yearsFromNow)
          : v
        updater(byYear[i + offset], value)
      })
    })
  }

  exec(savings, (t, v) => (t.savings += v))
  exec(withdrawals.fundedByBonds, (t, v) => (t.withdrawals.fundedByBonds += v))
  exec(
    withdrawals.fundedByRiskPortfolio,
    (t, v) => (t.withdrawals.fundedByRiskPortfolio += v)
  )
  return byYear
}

function _processReturnsParams(
  {returns, age}: TPAWParams,
  randomIndexesIntoHistoricalReturnsByYear: (index: number) => number = () =>
    _.random(historicalReturns.length - 1)
) {
  const adjustFn = (rate: number, adjustment: number) =>
    Math.exp(Math.log(1 + rate) - adjustment) - 1

  const n = historicalReturns.length
  const actualReturns = {
    stocks: _.sumBy(historicalReturns, x => x.stocks) / n,
    bonds: _.sumBy(historicalReturns, x => x.bonds) / n,
  }
  const adjustment =
    returns.historical.adjust.type === 'by'
      ? returns.historical.adjust
      : {
          stocks: actualReturns.stocks - returns.historical.adjust.stocks,
          bonds: actualReturns.bonds - returns.historical.adjust.bonds,
        }
  const historicalAdjusted = historicalReturns.map(x => ({
    stocks: adjustFn(x.stocks, adjustment.stocks),
    bonds: adjustFn(x.bonds, adjustment.bonds),
  }))

  const realized = _.times(numOfYears(age), i =>
    randomIndexesIntoHistoricalReturnsByYear(i)
  ).map(i => historicalAdjusted[i])
  return {...returns, historicalAdjusted, realized}
}

const _realValue = (
  {value, nominal}: {value: number; nominal: boolean},
  params: TPAWParams,
  yearsFromNow: number
) => (nominal ? value / Math.pow(1 + params.inflation, yearsFromNow) : value)

export const numericYear = (
  {age}: {age: {start: number; retirement: number; end: number}},
  x: YearRangeEdge
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

export const checkYearRange = (
  params: TPAWParams,
  yearRange: ValueForYearRange['yearRange']
) => {
  const start = numericYear(params, yearRange.start)
  const end = numericYear(params, yearRange.end)
  return start < params.age.start || end > params.age.end
    ? ('outOfBounds' as const)
    : start > end
    ? ('startGreaterThanEnd' as const)
    : ('ok' as const)
}
