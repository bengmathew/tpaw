import { useEffect, useState } from 'react'
import { useMarketData } from '../../Pages/App/WithMarketData'
import { asyncEffect } from '../../Utils/AsyncEffect'
import { fGet } from '../../Utils/Utils'
import { PlanParamsProcessed } from '../PlanParamsProcessed'
import { TPAWRunInWorker, TPAWRunInWorkerResult } from './TPAWRunInWorker'

// SHould be singleton so multiple uses of useTPAWWorker all use the same
// workers, and therefore the same random draws.
let _singleton: TPAWRunInWorker | null = null
export const getTPAWRunInWorkerSingleton = () => {
  if (!_singleton) _singleton = new TPAWRunInWorker()
  return _singleton
}

export type UseTPAWWorkerResult = TPAWRunInWorkerResult & {
  args: {
    numRuns: number
    params: PlanParamsProcessed
    percentiles: number[]
  }
}

export function useTPAWWorker(
  params: PlanParamsProcessed | null,
  numRuns: number,
  percentiles: number[],
) {
  const [result, setResult] = useState<UseTPAWWorkerResult | null>(null)
  const marketData = useMarketData()

  useEffect(() => {
    if (!params) {
      setResult(null)
    } else {
      return asyncEffect(async (status) => {
        const args = { numRuns, params, percentiles }
        const data = await getTPAWRunInWorkerSingleton().runSimulations(
          status,
          numRuns,
          params,
          percentiles,
          marketData,
        )
        if (status.canceled) return
        setResult({ ...fGet(data), args })
      })
    }
  }, [marketData, numRuns, params, percentiles])

  return result
}

export async function clearMemoizedRandom() {
  await getTPAWRunInWorkerSingleton().clearMemoizedRandom()
}
