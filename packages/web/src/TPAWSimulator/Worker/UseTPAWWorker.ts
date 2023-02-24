import { useEffect, useState } from 'react'
import { asyncEffect } from '../../Utils/AsyncEffect'
import { SimpleRange } from '../../Utils/SimpleRange'
import { fGet } from '../../Utils/Utils'
import { PlanParamsProcessed } from '../PlanParamsProcessed/PlanParamsProcessed'
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
    percentileRange:SimpleRange
  }
}

export function useTPAWWorker(
  params: PlanParamsProcessed | null,
  numRuns: number,
  percentileRange: SimpleRange,
) {
  const [result, setResult] = useState<UseTPAWWorkerResult | null>(null)

  useEffect(() => {
    if (!params) {
      setResult(null)
    } else {
      return asyncEffect(async (status) => {
        const args = { numRuns, params, percentileRange }
        const data = await getTPAWRunInWorkerSingleton().runSimulations(
          status,
          numRuns,
          params,
          [percentileRange.start, 50, percentileRange.end],
        )
        if (status.canceled) return
        setResult({ ...fGet(data), args })
      })
    }
  }, [numRuns, params, percentileRange])

  return result
}

export async function clearMemoizedRandom() {
  await getTPAWRunInWorkerSingleton().clearMemoizedRandom()
}
