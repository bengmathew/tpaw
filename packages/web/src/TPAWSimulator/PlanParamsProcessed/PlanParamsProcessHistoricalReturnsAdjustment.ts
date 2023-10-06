import {
  annualToMonthlyReturnRate,
  block,
  fGet,
  getStatsWithLog,
  historicalReturns,
  noCase,
  PlanParams,
  sequentialAnnualReturnsFromMonthly,
} from '@tpaw/common'
import {
  getAnnualToMonthlyRateConvertionCorrection,
  sampledAnnualReturnStatsMap,
} from './GetAnnualToMonthlyRateConvertionCorrection'

export function planParamsProcessHistoricalReturnsAdjustment(
  planParams: PlanParams,
  expectedReturnForPlanning: { monthly: { stocks: number; bonds: number } },
) {
  const adjustMontlyReturnsByType = (type: 'stocks' | 'bonds') => {
    const { adjustExpectedReturn } =
      planParams.advanced.historicalReturnsAdjustment[type]
    const volatilityScale =
      type === 'stocks'
        ? planParams.advanced.historicalReturnsAdjustment[type].volatilityScale
        : planParams.advanced.historicalReturnsAdjustment[type].enableVolatility
        ? 1
        : 0
    const { monthlyTargetExpectedValue, correctForBlockSampling } = block(
      () => {
        switch (adjustExpectedReturn.type) {
          case 'none':
            return {
              monthlyTargetExpectedValue:
                historicalReturns.monthly.annualStats[type].ofBase
                  .expectedValue,
              correctForBlockSampling: false,
            }
          case 'toExpectedUsedForPlanning':
            return {
              monthlyTargetExpectedValue:
                expectedReturnForPlanning.monthly[type],
              correctForBlockSampling:
                adjustExpectedReturn.correctForBlockSampling,
            }
          case 'toAnnualExpectedReturn':
            return {
              monthlyTargetExpectedValue: annualToMonthlyReturnRate(
                adjustExpectedReturn.annualExpectedReturn,
              ),
              correctForBlockSampling:
                adjustExpectedReturn.correctForBlockSampling,
            }
          default:
            noCase(adjustExpectedReturn)
        }
      },
    )
    const correction = block(() => {
      switch (planParams.advanced.sampling.type) {
        case 'historical':
          return getAnnualToMonthlyRateConvertionCorrection.forHistoricalSequence(
            type,
          )
        case 'monteCarlo':
          return correctForBlockSampling
            ? getAnnualToMonthlyRateConvertionCorrection.forMonteCarlo(
                planParams.advanced.sampling.blockSizeForMonteCarloSampling,
                type,
              )
            : 0
        default: {
          noCase(planParams.advanced.sampling.type)
        }
      }
    })

    return historicalReturns.monthly[type].adjust(
      monthlyTargetExpectedValue - correction * Math.pow(volatilityScale, 2),
      volatilityScale,
    )
  }

  return {
    monthly: block(() => {
      const stocks = adjustMontlyReturnsByType('stocks')
      const bonds = adjustMontlyReturnsByType('bonds')
      const annualStats = {
        direct: {
          stocks: getStatsWithLog(sequentialAnnualReturnsFromMonthly(stocks)),
          bonds: getStatsWithLog(sequentialAnnualReturnsFromMonthly(bonds)),
        },
        estimatedSampledStats: {
          stocks: {
            ofLog: {
              variance:
                Math.pow(
                  planParams.advanced.historicalReturnsAdjustment.stocks
                    .volatilityScale,
                  2,
                ) *
                (planParams.advanced.sampling.type === 'monteCarlo'
                  ? fGet(
                      sampledAnnualReturnStatsMap.get(
                        planParams.advanced.sampling
                          .blockSizeForMonteCarloSampling,
                      ),
                    ).stocks.varianceOfLogAveragedOverThread
                  : planParams.advanced.sampling.type === 'historical'
                  ? historicalReturns.monthly.annualStats.stocks.ofLog.variance
                  : noCase(planParams.advanced.sampling.type)),
            },
          },
        },
      }
      return { stocks, bonds, annualStats }
    }),
  }
}
