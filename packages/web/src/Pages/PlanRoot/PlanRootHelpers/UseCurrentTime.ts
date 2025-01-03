import { CalendarMonth, Guards, getZonedTimeFns, letIn } from '@tpaw/common'
import {
  chain,
  constant,
  gte,
  integer,
  json,
  number,
  object,
  string,
} from 'json-guard'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useIANATimezoneName } from './WithNonPlanParams'

type SetFastForwardSpec = (
  fastForward: Omit<FastForwardSpec, 'v' | 'planId'>,
) => void

export type CurrentTimeInfo = {
  startTime: number
  currentTimestamp: number
  nowAsCalendarDay: CalendarMonth
  forceUpdateCurrentTime: () => number
  fastForwardInfo:
    | {
        isFastForwarding: true
        spec: FastForwardSpec
        actualCurrentTimestamp: number
        setFastForwardSpec: SetFastForwardSpec
      }
    | {
        isFastForwarding: false
        setFastForwardSpec: SetFastForwardSpec
      }
}

export const useCurrentTime = ({
  planId,
  startTime: startTimeIn,
}: {
  planId: string
  startTime?: number
}): CurrentTimeInfo => {
  const { ianaTimezoneName, getZonedTime } = useIANATimezoneName()

  const startTimeBeforeFastForwardRef = useRef(startTimeIn ?? Date.now())

  const [currentTimestampWithoutFastForward, setCurrentTimeWithoutFastForward] =
    useState(startTimeBeforeFastForwardRef.current)
  useEffect(() => {
    const interval = window.setInterval(
      () => setCurrentTimeWithoutFastForward(Date.now()),
      1000 * 60 * 15, // Every 15 minutes
    )
    return () => window.clearInterval(interval)
  }, [])

  // ------- FAST FORWARD -------
  const [fastForwardSpecIn, setFastForwardSpecIn] =
    useState<FastForwardSpec | null>(() => _FastForward.read())
  const fastForwardSpec =
    fastForwardSpecIn && planId === fastForwardSpecIn.planId
      ? fastForwardSpecIn
      : null
  useEffect(() => _FastForward.write(fastForwardSpec), [fastForwardSpec])
  const setFastForwardSpec: SetFastForwardSpec = (x) =>
    setFastForwardSpecIn({ ...x, v: 1, planId })
  const fastForwardInfo: CurrentTimeInfo['fastForwardInfo'] = fastForwardSpec
    ? {
        isFastForwarding: true,
        spec: fastForwardSpec,
        actualCurrentTimestamp: currentTimestampWithoutFastForward,
        setFastForwardSpec,
      }
    : {
        isFastForwarding: false,
        setFastForwardSpec,
      }

  const startTime = useMemo(
    () =>
      _FastForward.apply(
        startTimeBeforeFastForwardRef.current,
        ianaTimezoneName,
        fastForwardSpec,
      ),
    [ianaTimezoneName, fastForwardSpec],
  )

  const currentTimestamp = useMemo(
    () =>
      _FastForward.apply(
        currentTimestampWithoutFastForward,
        ianaTimezoneName,
        fastForwardSpec,
      ),
    [currentTimestampWithoutFastForward, ianaTimezoneName, fastForwardSpec],
  )

  const forceUpdateCurrentTime = useCallback(() => {
    const newTime = Date.now()
    setCurrentTimeWithoutFastForward(newTime)
    return _FastForward.apply(newTime, ianaTimezoneName, fastForwardSpec)
  }, [fastForwardSpec, ianaTimezoneName])

  const nowAsCalendarDay = useMemo(
    () =>
      letIn(getZonedTime(currentTimestamp), (nowAsDateTime) => ({
        year: nowAsDateTime.year,
        month: nowAsDateTime.month,
      })),
    [currentTimestamp, getZonedTime],
  )
  return {
    startTime,
    currentTimestamp,
    nowAsCalendarDay,
    forceUpdateCurrentTime,
    fastForwardInfo,
  }
}

export type FastForwardSpec = {
  v: 1
  planId: string
  years: number
  months: number
  days: number
  hours: number
}

namespace _FastForward {
  const _guard = object({
    v: constant(1),
    planId: Guards.uuid,
    years: chain(number, integer, gte(0)),
    months: chain(number, integer, gte(0)),
    days: chain(number, integer, gte(0)),
    hours: chain(number, integer, gte(0)),
  })
  export const read = () => {
    const src = window.localStorage.getItem('UseCurrentTime_FastForward')
    if (!src) return null
    const guardResult = chain(string, json, _guard)(src)
    if (guardResult.error) return null
    return guardResult.value
  }

  export const write = (fastForward: FastForwardSpec | null) => {
    if (!fastForward) {
      window.localStorage.removeItem('UseCurrentTime_FastForward')
      return
    }
    window.localStorage.setItem(
      'UseCurrentTime_FastForward',
      JSON.stringify(fastForward),
    )
  }

  export const apply = (
    timestamp: number,
    ianaTimezoneName: string,
    fastForward: FastForwardSpec | null,
  ) => {
    const dateTime = getZonedTimeFns(ianaTimezoneName)(timestamp)
    return fastForward
      ? dateTime
          .plus({
            years: fastForward.years,
            months: fastForward.months,
            days: fastForward.days,
            hours: fastForward.hours,
          })
          .toMillis()
      : timestamp
  }
}
