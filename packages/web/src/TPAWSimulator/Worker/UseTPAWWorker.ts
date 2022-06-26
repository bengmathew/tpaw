import {useEffect, useState} from 'react'
import {asyncEffect} from '../../Utils/AsyncEffect'
import {fGet} from '../../Utils/Utils'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {TPAWRunInWorker, TPAWRunInWorkerResult} from './TPAWRunInWorker'

// SHould be singleton so multiple uses of useTPAWWorker all use the same
// workers, and therefore the same random draws.
let _singleton: TPAWRunInWorker | null = null
const _getSingleton = () => {
  if (!_singleton) _singleton = new TPAWRunInWorker()
  return _singleton
}

export type UseTPAWWorkerResult = TPAWRunInWorkerResult & {
  args: {
    numRuns: number
    params: TPAWParamsProcessed
    percentiles: number[]
  }
}

export function useTPAWWorker(
  params: TPAWParamsProcessed | null,
  numRuns: number,
  percentiles: number[]
) {
  const [result, setResult] = useState<UseTPAWWorkerResult | null>(null)

  useEffect(() => {
    if (!params) {
      setResult(null)
    } else {
      return asyncEffect(async status => {
        const args = {numRuns, params, percentiles}
        const data = await _getSingleton().runSimulations(
          status,
          numRuns,
          params,
          percentiles
        )
        if (status.canceled) return
        setResult({...fGet(data), args})
      })
    }
  }, [numRuns, params, percentiles])

  return result
}

export async function clearMemoizedRandom() {
  await _getSingleton().clearMemoizedRandom()
}
