import { getDefaultPlanParams, PlanParams, planParamsGuard } from '@tpaw/common'
import { chain, json } from 'json-guard'
import { useEffect, useMemo, useState } from 'react'
import { AppError } from './AppError'

export function usePlanParams() {
  const [params, setParams] = useState(
    () =>
      _parseExternalParams(window.localStorage.getItem('params')) ??
      getDefaultPlanParams(),
  )
  const guardResult = useMemo(() => planParamsGuard(params), [params])
  if (guardResult.error)
    throw new AppError('invalidParameters', guardResult.message)

  useEffect(() => {
    window.localStorage.setItem('params', JSON.stringify(params))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params])

  return { params, setParams }
}
function _parseExternalParams(
  str: string | string[] | undefined | null,
): PlanParams | null {
  if (typeof str !== 'string') return null
  const guardResult = chain(json, planParamsGuard)(str)
  if (guardResult.error)
    throw new AppError('invalidParameters', `${guardResult.message}`)
  return guardResult.value
}
