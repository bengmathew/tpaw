import {
  DEFAULT_SWR_WITHDRAWAL_PERCENT,
  EXPECTED_RETURN_PRESETS,
  fGet,
  GlidePath,
  historicalReturns,
  linearFnFomPoints,
  PlanParams,
  RISK_TOLERANCE_VALUES,
  SUGGESTED_INFLATION,
  ValueForYearRange,
} from '@tpaw/common'
import _ from 'lodash'
import { MarketData } from '../Pages/Common/GetMarketData'
import { nominalToReal } from '../Utils/NominalToReal'
import { assert, noCase } from '../Utils/Utils'
import { extendPlanParams, PlanParamsExt } from './PlanParamsExt'

export type PlanParamsProcessed = ReturnType<typeof processPlanParams>
export function processPlanParams(
  paramsExt: PlanParamsExt,
  marketData: MarketData,
) {
  const { numYears, asYFN, params, numRetirementYears } = paramsExt

  const _normalizeGlidePath = (glidePath: GlidePath) => {
    const stage1 = [
      { year: 0, stocks: glidePath.start.stocks },
      // Keeps only the first year of duplicates, and removes now and end.
      ..._.uniqBy(
        _.sortBy(
          glidePath.intermediate.map(({ year, stocks }) => ({
            year: asYFN(year),
            stocks,
          })),
          (x) => x.year,
        ).filter((x) => x.year !== 0 && x.year !== numYears - 1),
        (x) => x.year,
      ),
      { year: numYears - 1, stocks: glidePath.end.stocks },
    ]

    let result = _.times(numYears)
    _.times(stage1.length - 1, (i) => {
      const curr = stage1[i]
      const next = stage1[i + 1]
      const fn = linearFnFomPoints(
        curr.year,
        curr.stocks,
        next.year,
        next.stocks,
      )
      _.range(curr.year, next.year + 1).forEach((x) => (result[x] = fn(x)))
    })
    assert(result.length === numYears)
    return result
  }
  const returns = _processReturnsParams(params, marketData)

  const result = {
    strategy: params.strategy,
    people: params.people,
    currentPortfolioBalance: params.currentPortfolioBalance,
    byYear: _processByYearParams(paramsExt, marketData),
    adjustmentsToSpending: {
      tpawAndSPAW: (() => {
        const { spendingCeiling, spendingFloor, legacy } =
          params.adjustmentsToSpending.tpawAndSPAW
        return {
          spendingCeiling:
            spendingCeiling === null
              ? null
              : Math.max(spendingCeiling, params.risk.tpawAndSPAW.lmp),
          spendingFloor,
          legacy: (() => {
            const { total } = params.adjustmentsToSpending.tpawAndSPAW.legacy
            const external = _.sum(
              legacy.external.map((x) =>
                nominalToReal(
                  x,
                  processInflation(params.inflation, marketData),
                  numYears,
                ),
              ),
            )
            const target = Math.max(total - external, 0)
            return { total, external, target }
          })(),
        }
      })(),
    },

    risk: (() => {
      const { spawAndSWR, spaw, swr, tpaw, tpawAndSPAW } = params.risk

      const tpawGlidePath = _tpawGlidePath(paramsExt, returns)

      const legacyStockAllocation = _applyMerton(
        returns,
        tpaw.riskTolerance.at20 + tpaw.riskTolerance.forLegacyAsDeltaFromAt20,
        0, // Does not matter
      ).stockAllocation

      return {
        tpaw: {
          allocation: _normalizeGlidePath({
            start: { stocks: tpawGlidePath.now.stockAllocation },
            intermediate: [],
            end: { stocks: tpawGlidePath.atMaxAge.stockAllocation },
          }),
          allocationForLegacy: {
            stocks: legacyStockAllocation,
          },
        },
        tpawAndSPAW: {
          spendingTilt: _.times(numYears, (x) =>
            params.strategy === 'SPAW'
              ? spaw.spendingTilt
              : linearFnFomPoints(
                  0,
                  tpawGlidePath.now.spendingTilt,
                  numYears - 1,
                  tpawGlidePath.atMaxAge.spendingTilt,
                )(x),
          ),
          lmp: tpawAndSPAW.lmp,
        },
        spawAndSWR: {
          allocation: _normalizeGlidePath(spawAndSWR.allocation),
        },
        swr: {
          withdrawal:
            swr.withdrawal.type === 'default'
              ? {
                  type: 'asPercent' as const,
                  percent: DEFAULT_SWR_WITHDRAWAL_PERCENT(numRetirementYears),
                }
              : swr.withdrawal,
        },
      }
    })(),

    returns,
    sampling: params.sampling,
    original: params,
  }
  return result
}

const _tpawGlidePath = (
  paramsExt: PlanParamsExt,
  returns: ReturnType<typeof _processReturnsParams>,
) => {
  const { params, pickPerson, longerLivedPerson } = paramsExt
  const riskTolerance = (() => {
    const person = pickPerson(longerLivedPerson)
    const atAge = (x: number) =>
      Math.max(
        RISK_TOLERANCE_VALUES.DATA[0],
        linearFnFomPoints(
          20,
          params.risk.tpaw.riskTolerance.at20,
          person.ages.max,
          params.risk.tpaw.riskTolerance.at20 +
            params.risk.tpaw.riskTolerance.deltaAtMaxAge,
        )(x),
      )
    return {
      now: atAge(person.ages.current),
      atMaxAge: atAge(person.ages.max),
    }
  })()

  const deltaOverPlanningYears = riskTolerance.atMaxAge - riskTolerance.now

  const _currMerton = (riskTolerance: number) =>
    _applyMerton(returns, riskTolerance, params.risk.tpaw.timePreference)

  const maxRiskTolerance =
    _.find(
      RISK_TOLERANCE_VALUES.DATA,
      (x) => _currMerton(x).stockAllocation === 1,
    ) ?? fGet(_.last(RISK_TOLERANCE_VALUES.DATA))

  const riskToleranceAtMaxAge = Math.min(
    maxRiskTolerance,
    riskTolerance.atMaxAge,
  )

  return {
    now: _currMerton(riskToleranceAtMaxAge - deltaOverPlanningYears),
    atMaxAge: _currMerton(riskToleranceAtMaxAge),
  }
}

export const _applyMerton = (
  returns: ReturnType<typeof _processReturnsParams>,
  riskTolerance: number,
  timePreference: number,
) => {
  if (riskTolerance === 0) {
    return {
      spendingTilt: 0,
      stockAllocation: 0,
    }
  }
  
  const r = returns.expected.bonds
  const mu = returns.expected.stocks
  const sigmaPow2 = historicalReturns.stocks.log.variance
  const gamma = RISK_TOLERANCE_VALUES.riskToleranceToRRA(riskTolerance)

  const stockAllocation = Math.min(1, (mu - r) / (sigmaPow2 * gamma))
  const rho = timePreference

  const nu =
    (rho - (1 - gamma) * (Math.pow(mu - r, 2) / (2 * sigmaPow2 * gamma) + r)) /
    gamma

  const rOfPortfolio1 = historicalReturns.statsFn(
    returns.historicalAdjusted.map(
      ({ stocks }) =>
        stocks * stockAllocation +
        returns.expected.bonds * (1 - stockAllocation),
    ),
  ).expectedValue
  const rOfPortfolio = mu * stockAllocation + r * (1-stockAllocation)
  console.log(rOfPortfolio - rOfPortfolio1)
  const spendingTilt = rOfPortfolio - nu

  return { spendingTilt, stockAllocation }
}

function _processByYearParams(
  paramsExt: PlanParamsExt,
  marketData: MarketData,
) {
  const { asYFN, withdrawalStartYear, numYears, params } = paramsExt
  const {
    futureSavings,
    retirementIncome,
    adjustmentsToSpending: { extraSpending },
  } = params
  const withdrawalStart = asYFN(withdrawalStartYear)
  const lastWorkingYear = withdrawalStart > 0 ? withdrawalStart - 1 : 0
  const endYear = numYears - 1
  const byYear = _.times(numYears, (year) => ({
    futureSavingsAndRetirementIncome: 0,
    extraSpending: {
      essential: 0,
      discretionary: 0,
    },
    tpawAndSPAW: {
      risk: { lmp: year < withdrawalStart ? 0 : params.risk.tpawAndSPAW.lmp },
    },
  }))

  const exec = (
    values: ValueForYearRange[],
    minYear: number,
    maxYear: number,
    updater: (target: typeof byYear[0], value: number) => void,
  ) => {
    values.forEach(({ yearRange, value, nominal }) => {
      const normYearRange = extendPlanParams(params).asYFN(yearRange)
      const start = _.clamp(normYearRange.start, minYear, maxYear)
      const end = _.clamp(normYearRange.end, start, maxYear)
      _.range(start, end + 1).forEach((yearsFromNow) => {
        updater(
          byYear[yearsFromNow],
          nominalToReal(
            { value, nominal },
            processInflation(params.inflation, marketData),
            yearsFromNow,
          ),
        )
      })
    })
  }

  exec(
    futureSavings,
    0,
    lastWorkingYear,
    (t, v) => (t.futureSavingsAndRetirementIncome += v),
  )
  exec(
    retirementIncome,
    withdrawalStart,
    endYear,
    (t, v) => (t.futureSavingsAndRetirementIncome += v),
  )
  exec(
    extraSpending.essential,
    0,
    endYear,
    (t, v) => (t.extraSpending.essential += v),
  )
  exec(
    extraSpending.discretionary,
    0,
    endYear,
    (t, v) => (t.extraSpending.discretionary += v),
  )
  return byYear
}

export function processExpectedReturns(
  expected: PlanParams['returns']['expected'],
  marketData: MarketData,
) {
  switch (expected.type) {
    case 'manual':
      return { stocks: expected.stocks, bonds: expected.bonds }
    default:
      return EXPECTED_RETURN_PRESETS(expected.type, marketData)
  }
}

export function processInflation(
  inflation: PlanParams['inflation'],
  marketData: MarketData,
) {
  return inflation.type === 'suggested'
    ? SUGGESTED_INFLATION(marketData)
    : inflation.type === 'manual'
    ? inflation.value
    : noCase(inflation)
}

function _processReturnsParams(params: PlanParams, marketData: MarketData) {
  const { returns } = params
  const expected = processExpectedReturns(params.returns.expected, marketData)

  const historicalAdjusted = (() => {
    switch (returns.historical.type) {
      case 'default': {
        const adjustment = returns.historical.adjust
        const adjust = (type: 'stocks' | 'bonds') => {
          const historical = historicalReturns[type]

          const targetExpected =
            adjustment.type === 'to'
              ? adjustment[type]
              : adjustment.type === 'toExpected'
              ? expected[type]
              : adjustment.type === 'none'
              ? historical.expectedValue
              : adjustment.type === 'by'
              ? historical.expectedValue - adjustment[type]
              : noCase(adjustment)

          return historical.adjust(targetExpected)
        }

        return _.zipWith(
          adjust('stocks'),
          adjust('bonds'),
          (stocks, bonds) => ({ stocks, bonds }),
        )
      }
      case 'fixed': {
        const { stocks, bonds } = returns.historical
        return _.times(historicalReturns.raw.length, () => ({ stocks, bonds }))
      }
      default:
        noCase(returns.historical)
    }
  })()

  return { historical: returns.historical, expected, historicalAdjusted }
}
