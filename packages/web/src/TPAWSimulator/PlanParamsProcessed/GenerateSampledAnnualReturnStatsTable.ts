import {
  MAX_AGE_IN_MONTHS,
  historicalReturns,
  annualToMonthlyReturnRate,
} from '@tpaw/common'
import _ from 'lodash'
import { formatPercentage } from '../../Utils/FormatPercentage'
import { getTPAWRunInWorkerSingleton } from '../Worker/UseTPAWWorker'

type _T = {
  n: number
  mean: number
  ofLog: {
    mean: number
    varianceAveragedOverThread: number
  }
}
type SampledReturnStats = {
  oneYear: _T
  fiveYear: _T
  tenYear: _T
  thirtyYear: _T
}
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

const _correction = (empiricalAnnualMean: number, actualMonthlyMean: number) =>
  annualToMonthlyReturnRate(empiricalAnnualMean) - actualMonthlyMean