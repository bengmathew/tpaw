import _ from 'lodash'
import {MarketData} from '../Pages/Common/GetMarketData'
import {linearFnFomPoints} from '../Utils/LinearFn'
import {nominalToReal} from '../Utils/NominalToReal'
import {assert, noCase} from '../Utils/Utils'
import {
  defaultSWRWithdrawalRate,
  EXPECTED_RETURN_PRESETS,
  SUGGESTED_INFLATION,
} from './DefaultParams'
import {historicalReturns} from './HistoricalReturns'
import {
  GlidePath,
  TPAWParams,
  tpawParamsValidator,
  ValueForYearRange,
} from './TPAWParams'
import {extendTPAWParams, TPAWParamsExt} from './TPAWParamsExt'

export type TPAWParamsProcessed = ReturnType<typeof processTPAWParams>
export function processTPAWParams(
  paramsExt: TPAWParamsExt,
  marketData: MarketData
) {
  const {numYears, asYFN, params, numRetirementYears} = paramsExt
  tpawParamsValidator(params)
  const {inflation: inflationIn, ...paramsWithoutInflation} = params

  const _normalizeGlidePath = (glidePath: GlidePath) => {
    const stage1 = [
      {year: 0, stocks: glidePath.start.stocks},
      // Keeps only the first year of duplicates, and removes now and end.
      ..._.uniqBy(
        _.sortBy(
          glidePath.intermediate.map(({year, stocks}) => ({
            year: asYFN(year),
            stocks,
          })),
          x => x.year
        ).filter(x => x.year !== 0 && x.year !== numYears - 1),
        x => x.year
      ),
      {year: numYears - 1, stocks: glidePath.end.stocks},
    ]

    let result = _.times(numYears)
    _.times(stage1.length - 1, i => {
      const curr = stage1[i]
      const next = stage1[i + 1]
      const fn = linearFnFomPoints(
        curr.year,
        curr.stocks,
        next.year,
        next.stocks
      )
      _.range(curr.year, next.year + 1).forEach(x => (result[x] = fn(x)))
    })
    assert(result.length === numYears)
    return result
  }

  const byYear = _processByYearParams(paramsExt, marketData)

  const targetAllocation = {
    ...params.targetAllocation,
    regularPortfolio: {
      forTPAW: _normalizeGlidePath(
        params.targetAllocation.regularPortfolio.forTPAW
      ),
      forSPAWAndSWR: _normalizeGlidePath(
        params.targetAllocation.regularPortfolio.forSPAWAndSWR
      ),
    },
  }

  const legacy = (() => {
    const {total} = params.legacy
    const external = _.sum(
      params.legacy.external.map(x =>
        nominalToReal(
          x,
          processInflation(params.inflation, marketData),
          numYears
        )
      )
    )
    const target = Math.max(total - external, 0)
    return {total, external, target}
  })()

  const result = {
    ...paramsWithoutInflation,
    swrWithdrawal:
      params.swrWithdrawal.type === 'default'
        ? {
            type: 'asPercent' as const,
            percent: defaultSWRWithdrawalRate(numRetirementYears),
          }
        : params.swrWithdrawal,
    spendingCeiling:
      params.spendingCeiling === null
        ? null
        : Math.max(params.spendingCeiling, params.withdrawals.lmp),
    returns: _processReturnsParams(params, marketData),
    targetAllocation,
    byYear,
    legacy,
    original: params,
  }
  return result
}

function _processByYearParams(
  paramsExt: TPAWParamsExt,
  marketData: MarketData
) {
  const {asYFN, withdrawalStartYear, numYears, params} = paramsExt
  const {savings, retirementIncome, withdrawals} = params
  const withdrawalStart = asYFN(withdrawalStartYear)
  const lastWorkingYear = withdrawalStart > 0 ? withdrawalStart - 1 : 0
  const endYear = numYears - 1
  const byYear = _.times(numYears, year => ({
    savings: 0,
    withdrawals: {
      essential: 0,
      discretionary: 0,
      lmp: year < withdrawalStart ? 0 : params.withdrawals.lmp,
    },
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
      _.range(start, end + 1).forEach(yearsFromNow => {
        updater(
          byYear[yearsFromNow],
          nominalToReal(
            {value, nominal},
            processInflation(params.inflation, marketData),
            yearsFromNow
          )
        )
      })
    })
  }

  exec(savings, 0, lastWorkingYear, (t, v) => (t.savings += v))
  exec(retirementIncome, withdrawalStart, endYear, (t, v) => (t.savings += v))
  exec(
    withdrawals.essential,
    0,
    endYear,
    (t, v) => (t.withdrawals.essential += v)
  )
  exec(
    withdrawals.discretionary,
    0,
    endYear,
    (t, v) => (t.withdrawals.discretionary += v)
  )
  return byYear
}

export function processExpectedReturns(
  expected: TPAWParams['returns']['expected'],
  marketData: MarketData
) {
  switch (expected.type) {
    case 'manual':
      return {stocks: expected.stocks, bonds: expected.bonds}
    default:
      return EXPECTED_RETURN_PRESETS(expected.type, marketData)
  }
}

export function processInflation(
  inflation: TPAWParams['inflation'],
  marketData: MarketData
) {
  return inflation.type === 'suggested'
    ? SUGGESTED_INFLATION(marketData)
    : inflation.type === 'manual'
    ? inflation.value
    : noCase(inflation)
}

function _processReturnsParams(params: TPAWParams, marketData: MarketData) {
  const {returns} = params
  const expected = processExpectedReturns(params.returns.expected, marketData)
  const n = historicalReturns.length

  const historicalAdjusted = (() => {
    switch (returns.historical.type) {
      case 'default': {
        const adjustment = returns.historical.adjust
        const adjust = (type: 'stocks' | 'bonds') => {
          const n = historicalReturns.length
          const historical = historicalReturns.map(x => x[type])
          const historicalExpected = _.mean(historical)

          const targetExpected =
            adjustment.type === 'to'
              ? adjustment[type]
              : adjustment.type === 'toExpected'
              ? expected[type]
              : adjustment.type === 'none'
              ? historicalExpected
              : adjustment.type === 'by'
              ? historicalExpected - adjustment[type]
              : noCase(adjustment)

          const historicalLog = historical.map(x => Math.log(1 + x))
          const historicalLogExpected = _.mean(historicalLog)
          const historicalLogVariance =
            _.sumBy(historicalLog, x =>
              Math.pow(x - historicalLogExpected, 2)
            ) /
            (n - 1)

          // Empirically determined to be more accurate.
          const delta = {stocks: 0.0006162, bonds: 0.0000005}
          const targetLogExpected =
            Math.log(1 + targetExpected) -
            historicalLogVariance / 2 +
            delta[type]

          const adjustmentLogExpected =
            historicalLogExpected - targetLogExpected

          const adjustedLog = historicalLog.map(
            log => log - adjustmentLogExpected
          )
          const adjusted = adjustedLog.map(x => Math.exp(x) - 1)
          // console.dir('-------------')
          // console.dir(type)
          // console.dir(`historicalLogVariance/2: ${historicalLogVariance / 2}`)
          // console.dir(`additionalDelta: ${delta[type]}`)
          // console.dir(`total: ${historicalLogVariance / 2 + delta[type]}`)
          // console.dir(`targetExpected: ${targetExpected}`)
          // console.dir(`adjustedExpected: ${_.mean(adjusted)}`)
          // console.dir(`diff: ${Math.abs(targetExpected - _.mean(adjusted))}`)
          return adjusted
        }

        const stocks = adjust('stocks')
        const bonds = adjust('bonds')
        return stocks.map((stocks, i) => ({stocks, bonds: bonds[i]}))
      }
      case 'fixed': {
        const {stocks, bonds} = returns.historical
        return _.times(n, () => ({stocks, bonds}))
      }
      default:
        noCase(returns.historical)
    }
  })()

  return {historical: returns.historical, expected, historicalAdjusted}
}
