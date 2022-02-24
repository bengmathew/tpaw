import _ from 'lodash'
import {numOfYears} from '../Utils/NumOfYears'
import {nominalToReal} from '../Utils/NominalToReal'
import {historicalReturns} from './HistoricalReturns'
import {
  TPAWParams,
  tpawParamsValidator,
  ValueForYearRange,
  YearRange,
  YearRangeEdge,
} from './TPAWParams'

export type TPAWParamsProcessed = ReturnType<typeof processTPAWParams>
export function processTPAWParams(
  params: TPAWParams,
  randomIndexesIntoHistoricalReturnsByYear?: (year: number) => number
) {
  tpawParamsValidator(params)
  const {inflation, ...paramsWithoutInflation} = params
  const _completeAllocation = ({stocks}: {stocks: number}) => ({
    stocks,
    bonds: 1 - stocks,
  })
  const result = {
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
          nominalToReal(x, params.inflation, numOfYears(params.age))
        )
      )
      const target = Math.max(total - external, 0)
      return {totall: total, external, target}
    })(),
  }
  return result
}

function _processByYearParams(params: TPAWParams) {
  const {age, savings, retirementIncome, withdrawals} = params
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
        updater(
          byYear[yearsFromNow],
          nominalToReal({value: v, nominal}, params.inflation, yearsFromNow)
        )
      })
    })
  }

  exec(savings, (t, v) => (t.savings += v))
  exec(retirementIncome, (t, v) => (t.savings += v))
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

export const numericYearRange = (
  params: {age: {start: number; retirement: number; end: number}},
  x: YearRange
) => ({start: numericYear(params, x.start), end: numericYear(params, x.end)})
