import { useEffect, useState } from 'react'
import { asyncEffect } from '../../Utils/AsyncEffect'
import { fGet } from '../../Utils/Utils'
import { ParamsExtended } from '../ExtentParams'
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
  params: PlanParamsProcessed
  paramsExt: ParamsExtended
}

export function useTPAWWorker(
  params: PlanParamsProcessed,
  paramsExt: ParamsExtended,
) {
  const [result, setResult] = useState<UseTPAWWorkerResult | null>(null)

  useEffect(() => {
    return asyncEffect(async (status) => {
      

      const data = await getTPAWRunInWorkerSingleton().runSimulations(
        status,
        params,
        paramsExt,
      )
      if (status.canceled) return
      setResult({ ...fGet(data), params, paramsExt })
    })
  }, [params, paramsExt])

  return result
}

export async function clearMemoizedRandom() {
  await getTPAWRunInWorkerSingleton().clearMemoizedRandom()
}
