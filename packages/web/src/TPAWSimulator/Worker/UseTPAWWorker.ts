import {useEffect, useRef, useState} from 'react'
import {asyncEffect} from '../../Utils/AsyncEffect'
import {fGet} from '../../Utils/Utils'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {TPAWRunInWorker, TPAWRunInWorkerResult} from './TPAWRunInWorker'

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

  const workerRef = useRef<TPAWRunInWorker | null>(null)
  useEffect(() => {
    workerRef.current = new TPAWRunInWorker()
    return () => workerRef.current?.terminate()
  }, [])

  useEffect(() => {
    if (!params) {
      setResult(null)
    } else {
      return asyncEffect(async status => {
        const args = {numRuns, params, percentiles}
        const data = await fGet(workerRef.current).runSimulations(
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
