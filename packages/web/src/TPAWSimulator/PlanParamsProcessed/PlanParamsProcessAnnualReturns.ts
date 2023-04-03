import {
  annualToMonthlyReturnRate,
  EXPECTED_ANNUAL_RETURN_PRESETS,
  historicalReturns,
  noCase,
  Params,
  PlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import { MarketData } from '../../Pages/Common/GetMarketData'
import { getAnnualToMonthlyRateConvertionCorrection } from './GetAnnualToMonthlyRateConvertionCorrection'

export const planParamsProcessExpectedAnnualReturns = (
  annualReturns: PlanParams['advanced']['annualReturns'],
  marketData: MarketData['latest'],
) =>
  annualReturns.expected.type === 'manual'
    ? {
        stocks: annualReturns.expected.stocks,
        bonds: annualReturns.expected.bonds,
      }
    : EXPECTED_ANNUAL_RETURN_PRESETS(annualReturns.expected.type, marketData)

export function planParamsProcessAnnualReturnsParams(
  params: Params,
  marketData: MarketData['latest'],
) {
  const { annualReturns } = params.plan.advanced
  const expectedAnnualReturns = planParamsProcessExpectedAnnualReturns(
    annualReturns,
    marketData,
  )
  const historicalMonthlyAdjusted = (() => {
    switch (annualReturns.historical.type) {
      case 'adjusted': {
        const { adjustment, correctForBlockSampling } = annualReturns.historical
        const adjust = (type: 'stocks' | 'bonds') => {
          const targetAnnualExpected =
            adjustment.type === 'to'
              ? adjustment[type]
              : adjustment.type === 'toExpected'
              ? expectedAnnualReturns[type]
              : adjustment.type === 'by'
              ? historicalReturns.monthly.annualStats[type].mean -
                adjustment[type]
              : noCase(adjustment)

          const correction = (() => {
            switch (params.plan.advanced.sampling) {
              case 'historical':
                return getAnnualToMonthlyRateConvertionCorrection.forHistoricalSequence(
                  type,
                )
              case 'monteCarlo':
                return correctForBlockSampling
                  ? getAnnualToMonthlyRateConvertionCorrection.forMonteCarlo(
                      params.plan.advanced.monteCarloSampling.blockSize,
                      type,
                    )
                  : 0
              default:
                noCase(params.plan.advanced.sampling)
            }
          })()

          return historicalReturns.monthly[type].adjust(
            annualToMonthlyReturnRate(targetAnnualExpected) - correction,
          )
        }

        return _.zipWith(
          adjust('stocks'),
          adjust('bonds'),
          (stocks, bonds) => ({ stocks, bonds }),
        )
      }
      case 'fixed': {
        const { stocks, bonds } = annualToMonthlyReturnRate(
          annualReturns.historical,
        )
        return _.times(historicalReturns.monthly.stocks.returns.length, () => ({
          stocks,
          bonds,
        }))
      }
      case 'unadjusted': {
        return _.zipWith(
          historicalReturns.monthly.stocks.returns,
          historicalReturns.monthly.bonds.returns,
          (stocks, bonds) => ({ stocks, bonds }),
        )
      }
      default:
        noCase(annualReturns.historical)
    }
  })()

  return {
    expectedAnnualReturns,
    historicalMonthlyAdjusted,
  }
}