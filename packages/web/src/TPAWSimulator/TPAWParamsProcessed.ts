import _ from 'lodash'
import {nominalToReal} from '../Utils/NominalToReal'
import {extendTPAWParams} from './TPAWParamsExt'
import {historicalReturns} from './HistoricalReturns'
import {TPAWParams, tpawParamsValidator, ValueForYearRange} from './TPAWParams'

export type TPAWParamsProcessed = ReturnType<typeof processTPAWParams>
export function processTPAWParams(
  params: TPAWParams,
  randomIndexesIntoHistoricalReturnsByYear?: (year: number) => number
) {
  const {numYears} = extendTPAWParams(params)
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
          nominalToReal(x, params.inflation, numYears)
        )
      )
      const target = Math.max(total - external, 0)
      return {totall: total, external, target}
    })(),
    original: params,
  }
  return result
}

function _processByYearParams(params: TPAWParams) {
  const {people, savings, retirementIncome, withdrawals} = params
  const {asYFN, withdrawalStartYear, numYears} = extendTPAWParams(params)
  const withdrawalStart = asYFN(withdrawalStartYear)
  const lastWorkingYear = withdrawalStart > 0 ? withdrawalStart - 1 : 0
  const endYear = numYears - 1
  const byYear = _.times(numYears, () => ({
    savings: 0,
    withdrawals: {fundedByBonds: 0, fundedByRiskPortfolio: 0},
  }))

  const exec = (
    values: ValueForYearRange[],
    minYear: number,
    maxYear: number,
    updater: (target: typeof byYear[0], value: number) => void
  ) => {
    values.forEach(({yearRange, value, nominal}) => {
      const normYearRange = extendTPAWParams(params).asYFN(yearRange)
      const start = _.clamp(normYearRange.start, minYear, maxYear)
      const end = _.clamp(normYearRange.end, start, maxYear)
      _.range(start, end + 1).forEach((x, yearsFromNow) => {
        updater(
          byYear[yearsFromNow],
          nominalToReal({value, nominal}, params.inflation, yearsFromNow)
        )
      })
    })
  }

  exec(savings, 0, lastWorkingYear, (t, v) => (t.savings += v))
  exec(retirementIncome, withdrawalStart, endYear, (t, v) => (t.savings += v))
  exec(
    withdrawals.fundedByBonds,
    0,
    endYear,
    (t, v) => (t.withdrawals.fundedByBonds += v)
  )
  exec(
    withdrawals.fundedByRiskPortfolio,
    0,
    endYear,
    (t, v) => (t.withdrawals.fundedByRiskPortfolio += v)
  )
  return byYear
}

function _processReturnsParams(
  params: TPAWParams,
  randomIndexesIntoHistoricalReturnsByYear: (index: number) => number = () =>
    _.random(historicalReturns.length - 1)
) {
  const {returns, people} = params
  const {numYears} = extendTPAWParams(params)
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

  const realized = _.times(numYears, i =>
    randomIndexesIntoHistoricalReturnsByYear(i)
  ).map(i => historicalAdjusted[i])
  return {...returns, historicalAdjusted, realized}
}
