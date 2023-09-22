import {
  annualToMonthlyReturnRate,
  block,
  EXPECTED_ANNUAL_RETURN_PRESETS,
  historicalReturns,
  MarketData,
  noCase,
  PlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import { getAnnualToMonthlyRateConvertionCorrection } from './GetAnnualToMonthlyRateConvertionCorrection'

export const planParamsProcessExpectedAnnualReturns = (
  annualReturns: PlanParams['advanced']['annualReturns'],
  currentMarketData: MarketData.Data[0],
) =>
  annualReturns.expected.type === 'manual'
    ? {
        stocks: annualReturns.expected.stocks,
        bonds: annualReturns.expected.bonds,
      }
    : EXPECTED_ANNUAL_RETURN_PRESETS(
        annualReturns.expected.type,
        currentMarketData,
      )

export function planParamsProcessAnnualReturnsParams(
  planParams: PlanParams,
  marketData: MarketData.Data[0],
) {
  const { annualReturns } = planParams.advanced
  const expectedAnnualReturns = planParamsProcessExpectedAnnualReturns(
    annualReturns,
    marketData,
  )
  const historicalMonthlyAdjusted = block(() => {
    const adjustByType = (type: 'stocks' | 'bonds'): number[] => {
      const spec = annualReturns.historical[type]
      switch (spec.type) {
        case 'adjustExpected': {
          const { adjustment, correctForBlockSampling } = spec
          const targetAnnualExpected =
            adjustment.type === 'toValue'
              ? adjustment.value
              : adjustment.type === 'toExpectedUsedForPlanning'
              ? expectedAnnualReturns[type]
              : adjustment.type === 'byValue'
              ? historicalReturns.monthly.annualStats[type].mean -
                adjustment.value
              : noCase(adjustment)

          const correction = (() => {
            switch (planParams.advanced.sampling.type) {
              case 'historical':
                return getAnnualToMonthlyRateConvertionCorrection.forHistoricalSequence(
                  type,
                )
              case 'monteCarlo':
                return correctForBlockSampling
                  ? getAnnualToMonthlyRateConvertionCorrection.forMonteCarlo(
                      planParams.advanced.sampling
                        .blockSizeForMonteCarloSampling,
                      type,
                    )
                  : 0
              default: {
                noCase(planParams.advanced.sampling.type)
              }
            }
          })()

          return historicalReturns.monthly[type].adjust(
            annualToMonthlyReturnRate(targetAnnualExpected) - correction,
          )
        }
        case 'fixed': {
          const value = annualToMonthlyReturnRate(
            spec.value.type === 'expectedUsedForPlanning'
              ? expectedAnnualReturns[type]
              : spec.value.type === 'manual'
              ? spec.value.value
              : noCase(spec.value),
          )
          return _.times(
            historicalReturns.monthly[type].returns.length,
            () => value,
          )
        }
        case 'rawHistorical': {
          return historicalReturns.monthly[type].returns
        }
        default:
          noCase(spec)
      }
    }

    return _.zipWith(
      adjustByType('stocks'),
      adjustByType('bonds'),
      (stocks, bonds) => ({
        stocks,
        bonds,
      }),
    )
  })

  // TODO: expectedReturns{annual, monthly}
  return {
    expectedAnnualReturns,
    historicalMonthlyAdjusted,
  }
}
