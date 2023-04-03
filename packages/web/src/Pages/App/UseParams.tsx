import {
  NonPlanParams,
  PlanParams,
  Params,
  getDefaultPlanParams,
  planParamsGuard,
} from '@tpaw/common'
import { chain, json } from 'json-guard'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useAssertConst } from '../../Utils/UseAssertConst'
import { AppError } from './AppError'

export type SetPlanParams = (
  x: PlanParams | ((x: PlanParams, y: Params) => PlanParams),
) => void
export type SetNonPlanParams = (
  x: NonPlanParams | ((x: NonPlanParams, y: Params) => NonPlanParams),
) => void

export function useParams() {
  const [currentTimeWithoutOffset, setCurrentTimeWithoutOffset] = useState(() =>
    DateTime.local(),
  )

  const [params, setParamsIn] = useState(
    () =>
      _parseExternalParams(window.localStorage.getItem('params')) ??
      getDefaultPlanParams(currentTimeWithoutOffset), // offset will be 0 for default.
  )

  const currentTime = useMemo(
    () =>
      params.nonPlan.dev.currentTimeFastForward.shouldFastForward
        ? currentTimeWithoutOffset.plus({
            years: params.nonPlan.dev.currentTimeFastForward.years,
            months: params.nonPlan.dev.currentTimeFastForward.months,
            days: params.nonPlan.dev.currentTimeFastForward.days,
            hours: params.nonPlan.dev.currentTimeFastForward.hours,
          })
        : currentTimeWithoutOffset,
    [params, currentTimeWithoutOffset],
  )

  // So setParams gets latest value without having dependencies.
  const currenTimeRef = useRef(currentTime)
  currenTimeRef.current = currentTime

  useEffect(() => {
    const interval = window.setInterval(
      () => setCurrentTimeWithoutOffset(DateTime.local()),
      1 * 60 * 1000, // Every 1 min.
    )
    return () => window.clearInterval(interval)
  }, [])

  // Automatically timestamp on parameter change.
  const setPlanParams = useCallback<SetPlanParams>((args) => {
    setParamsIn((params) => {
      const newPlanParams =
        typeof args === 'function' ? args(params.plan, params) : args
      if (newPlanParams === params.plan) return params
      const currentTime = currenTimeRef.current
      newPlanParams.timestamp = currentTime.valueOf()
      const newFullParams = _.cloneDeep(params)
      newFullParams.plan = newPlanParams
      if (newFullParams.nonPlan.defaultTimezone.type === 'auto') {
        newFullParams.nonPlan.defaultTimezone.ianaTimezoneName =
          currentTime.zoneName
      }
      planParamsGuard(newFullParams).force()
      return newFullParams
    })
  }, [])

  const setNonPlanParams = useCallback<SetNonPlanParams>((args) => {
    setParamsIn((params) => {
      const newNonPlanParams =
        typeof args === 'function' ? args(params.nonPlan, params) : args
      if (newNonPlanParams === params.nonPlan) return params
      const currentTime = currenTimeRef.current

      const newFullParams = _.cloneDeep(params)
      newFullParams.nonPlan = newNonPlanParams
      if (newFullParams.nonPlan.defaultTimezone.type === 'auto') {
        newFullParams.nonPlan.defaultTimezone.ianaTimezoneName =
          currentTime.zoneName
      }
      planParamsGuard(newFullParams).force()
      return newFullParams
    })
  }, [])

  const resetParams = useCallback((params: Params) => setParamsIn(params), [])

  useEffect(() => {
    window.localStorage.setItem('params', JSON.stringify(params))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params])

  const result = useMemo(() => {
    return {
      params,
      currentTime,
      setPlanParams,
      setNonPlanParams,
      resetParams,
    }
  }, [currentTime, params, resetParams, setNonPlanParams, setPlanParams])
  useAssertConst([resetParams, setNonPlanParams, setPlanParams])
  return result
}

function _parseExternalParams(
  str: string | string[] | undefined | null,
): Params | null {
  if (typeof str !== 'string') return null
  const guardResult = chain(json, planParamsGuard)(str)
  if (guardResult.error)
    throw new AppError('invalidParameters', `${guardResult.message}`)
  return guardResult.value
}
