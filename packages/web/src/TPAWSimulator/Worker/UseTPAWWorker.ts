import { NonPlanParams } from '@tpaw/common'
import { useCallback, useEffect, useState } from 'react'
import { useNonPlanParams } from '../../Pages/PlanRoot/PlanRootHelpers/WithNonPlanParams'
import { asyncEffect } from '../../Utils/AsyncEffect'
import { fGet } from '../../Utils/Utils'
import { PlanParamsExtended } from '../ExtentPlanParams'
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
  planParamsExt: PlanParamsExtended
  nonPlanParams: NonPlanParams
}

export function useTPAWWorker(
  params: PlanParamsProcessed,
  planParamsExt: PlanParamsExtended,
) {
  const { nonPlanParams } = useNonPlanParams()
  const [result, setResult] = useState<UseTPAWWorkerResult | null>(null)
  const [key, setKey] = useState(0)

  useEffect(() => {
    return asyncEffect(async (status) => {
      const data = await getTPAWRunInWorkerSingleton().runSimulations(
        status,
        params,
        planParamsExt,
        nonPlanParams,
      )
      if (status.canceled) return
      setResult({ ...fGet(data), params, planParamsExt, nonPlanParams })
    })
  }, [params, planParamsExt, key, nonPlanParams])

  const reRun = useCallback(() => {
    setKey((x) => x + 1)
  }, [])
  return { result, reRun }
}
