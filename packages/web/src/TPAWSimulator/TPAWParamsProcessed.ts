import _ from 'lodash'
import {blendReturns} from '../Utils/BlendReturns'
import {linearFnFomPoints} from '../Utils/LinearFn'
import {nominalToReal} from '../Utils/NominalToReal'
import {assert, noCase} from '../Utils/Utils'
import {historicalReturns} from './HistoricalReturns'
import {
  GlidePath,
  TPAWParams,
  tpawParamsValidator,
  ValueForYearRange,
} from './TPAWParams'
import {extendTPAWParams} from './TPAWParamsExt'

export type TPAWParamsProcessed = ReturnType<typeof processTPAWParams>
export function processTPAWParams(params: TPAWParams) {
  const {numYears, asYFN, withdrawalStartYear} = extendTPAWParams(params)
  tpawParamsValidator(params)
  const {inflation, ...paramsWithoutInflation} = params
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

  const byYear = _processByYearParams(params)
  const expectedReturns = blendReturns(params.returns.expected)

  const targetAllocation = {
    ...params.targetAllocation,
    regularPortfolio: {
      ...params.targetAllocation.regularPortfolio,
      forSPAW: _normalizeGlidePath(
        params.targetAllocation.regularPortfolio.forSPAW
      ),
    },
  }

  const legacy = (() => {
    const {total} = params.legacy
    const external = _.sum(
      params.legacy.external.map(x =>
        nominalToReal(x, params.inflation, numYears)
      )
    )
    const target = Math.max(total - external, 0)
    return {total, external, target}
  })()

  const result = {
    ...paramsWithoutInflation,

    spendingCeiling:
      params.spendingCeiling === null
        ? null
        : Math.max(params.spendingCeiling, params.withdrawals.lmp),
    returns: _processReturnsParams(params),
    targetAllocation,
    byYear,

    preCalculations: {
      forTPAW: {
        netPresentValue: {
          savings: _getNetPresentValueArr(
            expectedReturns({stocks: 0}),
            byYear.map(x => x.savings)
          ),
          withdrawals: {
            lmp: _getNetPresentValueArr(
              expectedReturns({stocks: 0}),
              byYear.map((x, i) =>
                i < asYFN(withdrawalStartYear) ? 0 : params.withdrawals.lmp
              )
            ),
            essential: _getNetPresentValueArr(
              expectedReturns({stocks: 0}),
              byYear.map(x => x.withdrawals.essential)
            ),
            discretionary: _getNetPresentValueArr(
              expectedReturns(params.targetAllocation.regularPortfolio.forTPAW),
              byYear.map(x => x.withdrawals.discretionary)
            ),
          },
        },
      },
      forSPAW: (() => {
        const rate = targetAllocation.regularPortfolio.forSPAW.map(stocks =>
          expectedReturns({stocks})
        )
        const _npv = _.curry(_getNetPresentValueArr)(rate)
        return {
          netPresentValue: {
            savings: _npv(byYear.map(x => x.savings)),
            withdrawals: {
              lmp: _npv(byYear.map(x => params.withdrawals.lmp)),
              essential: _npv(byYear.map(x => x.withdrawals.essential)),
              discretionary: _npv(byYear.map(x => x.withdrawals.discretionary)),
            },
            legacy: _npv(
              _.times(numYears, i =>
                i === numYears - 1 ? legacy.target / (1 + rate[i]) : 0
              )
            ),
          },
          cumulative1PlusGOver1PlusR: _cumulative1PlusGOver1PlusR(
            rate,
            params.scheduledWithdrawalGrowthRate
          ),
        }
      })(),
    },
    legacy,
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
    withdrawals: {essential: 0, discretionary: 0},
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
          nominalToReal({value, nominal}, params.inflation, yearsFromNow)
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

function _processReturnsParams(params: TPAWParams) {
  const {returns} = params
  const n = historicalReturns.length

  const historicalAdjusted = (() => {
    switch (returns.historical.type) {
      case 'default': {
        const adjustFn = (rate: number, adjustment: number) =>
          Math.exp(Math.log(1 + rate) - adjustment) - 1

        const actualReturns = {
          stocks: _.sumBy(historicalReturns, x => x.stocks) / n,
          bonds: _.sumBy(historicalReturns, x => x.bonds) / n,
        }
        const adjustment =
          returns.historical.adjust.type === 'by'
            ? returns.historical.adjust
            : returns.historical.adjust.type === 'to'
            ? {
                stocks: actualReturns.stocks - returns.historical.adjust.stocks,
                bonds: actualReturns.bonds - returns.historical.adjust.bonds,
              }
            : returns.historical.adjust.type === 'toExpected'
            ? {
                stocks: actualReturns.stocks - returns.expected.stocks,
                bonds: actualReturns.bonds - returns.expected.bonds,
              }
            : noCase(returns.historical.adjust)
        return historicalReturns.map(x => ({
          stocks: adjustFn(x.stocks, adjustment.stocks),
          bonds: adjustFn(x.bonds, adjustment.bonds),
        }))
      }
      case 'fixed': {
        const {stocks, bonds} = returns.historical
        return _.times(n, () => ({stocks, bonds}))
      }
      default:
        noCase(returns.historical)
    }
  })()

  return {...returns, historicalAdjusted}
}

const _getNetPresentValueArr = (rate: number | number[], amounts: number[]) => {
  const n = amounts.length
  const r = typeof rate === 'number' ? _.times(n, () => rate) : rate
  assert(r.length === n)
  const result = _.times(n)
  for (let i = n - 1; i >= 0; i--) {
    result[i] = amounts[i] + (i === n - 1 ? 0 : result[i + 1] / (1 + r[i]))
  }
  const withCurrentYear = result
  const withoutCurrentYear = result.map((x, i) => x - amounts[i])
  return {withCurrentYear, withoutCurrentYear}
}

const _cumulative1PlusGOver1PlusR = (r: number[], g: number) => {
  const n = r.length
  const result = _.times(n)
  for (let i = n - 1; i >= 0; i--) {
    result[i] = 1 + (i === n - 1 ? 0 : (result[i + 1] * (1 + g)) / (1 + r[i]))
  }
  return result
}
