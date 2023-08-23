import {
  annualToMonthlyReturnRate,
  fGet,
  getStats,
  historicalReturns,
  sequentialAnnualReturnsFromMonthly
} from '@tpaw/common'
import { sampledReturnsStatsTableMinified } from './SampledReturnsStatsTableMinified'

export const sampledAnnualReturnStatsMap = new Map(
  sampledReturnsStatsTableMinified.map(
    ([
      blockSize,
      stocks_mean,
      stocks_varianceOfLogAveragedOverThread,
      bonds_mean,
    ]) => [
      blockSize,
      {
        stocks: {
          mean: stocks_mean,
          varianceOfLogAveragedOverThread:
            stocks_varianceOfLogAveragedOverThread,
        },
        bonds: {
          mean: bonds_mean,
        },
      },
    ],
  ),
)

const _correction = (empiricalAnnualMean: number, actualMonthlyMean: number) =>
  annualToMonthlyReturnRate(empiricalAnnualMean) - actualMonthlyMean

export const getAnnualToMonthlyRateConvertionCorrection = {
  forMonteCarlo: (blockSize: number, type: 'stocks' | 'bonds') =>
    _correction(
      fGet(sampledAnnualReturnStatsMap.get(blockSize))[type].mean,
      historicalReturns.monthly[type].mean,
    ),

  forHistoricalSequence: (type: 'stocks' | 'bonds') => {
    const monthly = historicalReturns.monthly[type]
    const annualReturnStats = getStats(
      sequentialAnnualReturnsFromMonthly(monthly.returns),
    )
    return _correction(annualReturnStats.mean, monthly.mean)
  },
}
