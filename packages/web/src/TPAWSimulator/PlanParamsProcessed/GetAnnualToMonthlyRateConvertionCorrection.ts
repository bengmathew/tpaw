import {
  annualToMonthlyReturnRate,
  fGet,
  getStats,
  historicalReturns,
  MAX_AGE_IN_MONTHS,
  sequentialAnnualReturnsFromMonthly
} from '@tpaw/common'
import _ from 'lodash'
import { formatPercentage } from '../../Utils/FormatPercentage'
import { TPAWRunInWorker } from '../Worker/TPAWRunInWorker'
import { getTPAWRunInWorkerSingleton } from '../Worker/UseTPAWWorker'
import { sampledReturnsStatsTableMinified } from './SampledReturnsStatsTableMinified'

type SampledReturnStats = Awaited<
  ReturnType<TPAWRunInWorker['getSampledReturnStats']>
>
export type SampledReturnsStatsTableRaw = {
  stocks: SampledReturnStats
  bonds: SampledReturnStats
  blockSize: number
}[]
export async function generateSampledAnnualReturnStatsTable() {
  const result = [] as SampledReturnsStatsTableRaw
  const start0 = performance.now()

  const blockSizes = _.range(1, MAX_AGE_IN_MONTHS + 1)
  // const blockSizes = _.range(fGet(_.last(sampledReturnsStatsTableRaw)).blockSize+1, MAX_AGE_IN_MONTHS + 1)
  let i = 0
  for (const blockSize of blockSizes) {
    const getSampledReturnStats = async (returns: number[]) =>
      await getTPAWRunInWorkerSingleton().getSampledReturnStats(
        returns,
        blockSize,
        900000 * MAX_AGE_IN_MONTHS, // 1000000 seems to cause some issues.
      )

    const getCorrection = async (
      monthly: typeof historicalReturns.monthly.stocks,
    ): Promise<SampledReturnStats> => {
      const start = performance.now()
      const sampledReturnStats = await getSampledReturnStats(monthly.returns)

      // Test output for the first few blockSizes.
      if (i < 0) {
        const correction = _correction(
          sampledReturnStats.oneYear.mean,
          monthly.mean,
        )

        const testTargetAnnual = 0.04
        const testTargetMonthly = annualToMonthlyReturnRate(testTargetAnnual)
        const testWithCorrectionActual = (
          await getSampledReturnStats(
            monthly.adjust(testTargetMonthly - correction),
          )
        ).oneYear.mean
        const testWithoutCorrectionActual = (
          await getSampledReturnStats(monthly.adjust(testTargetMonthly))
        ).oneYear.mean

        const formatP = formatPercentage(10)
        console.log(`                  Correction: ${formatP(correction)}`)
        console.log(
          `   Test Diff With Correction: ${formatP(
            Math.abs(testTargetAnnual - testWithCorrectionActual),
          )}`,
        )
        console.log(
          `Test Diff Without Correction: ${formatP(
            Math.abs(testTargetAnnual - testWithoutCorrectionActual),
          )}`,
        )
      }
      console.log(`    Time: ${timeStr(performance.now() - start)}`)

      i++
      return sampledReturnStats
    }

    const timeStr = (time: number) => (time / 1000).toFixed(2)

    const start = performance.now()
    console.log(``)
    console.log(`----------------------------------`)
    console.log(`Block Size: ${blockSize}`)
    console.log(`----------------------------------`)
    console.log('STOCKS')
    const stocks = await getCorrection(historicalReturns.monthly.stocks)

    console.log('BONDS')
    const bonds = await getCorrection(historicalReturns.monthly.bonds)

    result.push({ stocks, bonds, blockSize })

    const completed = result.length
    const remaining = blockSizes.length - completed
    console.log(`Completed: ${completed}`)
    console.log(`Remaining: ${remaining}`)
    console.log(`Total time: ${timeStr(performance.now() - start0)}`)
    const currTime = performance.now() - start
    console.log(`Curr time: ${timeStr(currTime)}`)
    console.log(
      `Estimated time left (hours): ${timeStr(
        (currTime * remaining) / 60 / 60,
      )}`,
    )
  }
  return result
}

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
