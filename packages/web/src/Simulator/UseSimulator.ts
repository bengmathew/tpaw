import { assertFalse, block, fGet, PlanParams } from '@tpaw/common'
import _ from 'lodash'
import { useEffect, useMemo, useRef, useState } from 'react'
import { CalendarDayFns } from '../Utils/CalendarDayFns'
import {
  normalizePlanParams,
  PlanParamsNormalized,
} from './NormalizePlanParams/NormalizePlanParams'
import {
  DailyMarketSeriesSrc,
  simulateOnServer,
  SimulateOnServerResult,
} from './SimulateOnServer/SimulateOnServer'
import { usePortfolioBalanceEstimationCache } from './UsePortfolioBalanceEstimationCache'

const DEBOUNCE_TIME_MS = 500

export type SimulationResult = SimulateOnServerResult & {
  planParamsNormOfResult: PlanParamsNormalized
  dailyMarketSeriesSrcOfResult: DailyMarketSeriesSrc
  ianaTimezoneNameIfDatedPlanOfResult: string | null
  percentilesOfResult: { low: number; mid: number; high: number }
  numOfSimulationForMonteCarloSamplingOfResult: number
  randomSeedOfResult: number
}

export function useSimulator(
  planId: string,
  dailyMarketSeriesSrc: DailyMarketSeriesSrc,
  planParamsHistoryUpToActualCurrentTimestamp: {
    id: string
    params: PlanParams
  }[],
  simulationTimestamp: number,
  ianaTimezoneNameIfDatedPlan: string | null,
  percentiles: { low: number; mid: number; high: number },
  numOfSimulationForMonteCarloSampling: number,
  randomSeed: number,
) {
  const now = Date.now()
  const [state, setState] = useState<
    | {
        isRunning: true
        simulationStartTimestamp: {
          countingFromThisSimulation: number
          countingFromTheFirstDebouncedSimulation: number
        }
        prevResult: SimulationResult | null
      }
    | { isRunning: false; result: SimulationResult }
  >({
    isRunning: true,
    simulationStartTimestamp: {
      countingFromThisSimulation: now,
      countingFromTheFirstDebouncedSimulation: now,
    },
    prevResult: null,
  })

  const planParamsHistoryUpToSimulationTimestamp = useMemo(() => {
    return planParamsHistoryUpToActualCurrentTimestamp.slice(
      0,
      _.sortedLastIndexBy<{ params: { timestamp: number } }>(
        planParamsHistoryUpToActualCurrentTimestamp,
        { params: { timestamp: simulationTimestamp } },
        (x) => x.params.timestamp,
      ),
    )
  }, [planParamsHistoryUpToActualCurrentTimestamp, simulationTimestamp])

  const lastHistoryItem = fGet(_.last(planParamsHistoryUpToSimulationTimestamp))
  const planParamsNorm = useMemo(
    () =>
      normalizePlanParams(lastHistoryItem.params, {
        timestamp: simulationTimestamp,
        calendarDay: lastHistoryItem.params.datingInfo.isDated
          ? CalendarDayFns.fromTimestamp(
              simulationTimestamp,
              fGet(ianaTimezoneNameIfDatedPlan),
            )
          : null,
      }),
    [lastHistoryItem.params, simulationTimestamp, ianaTimezoneNameIfDatedPlan],
  )

  const { applyCache } = usePortfolioBalanceEstimationCache(planId)

  const lastRunTimestampRef = useRef<number>(0)

  useEffect(() => {
    const abortController = new AbortController()
    const now = Date.now()
    const timeSinceLastRun = now - lastRunTimestampRef.current
    lastRunTimestampRef.current = now
    setState((prev) => ({
      isRunning: true,
      simulationStartTimestamp: {
        countingFromThisSimulation: now,
        countingFromTheFirstDebouncedSimulation: prev.isRunning
          ? prev.simulationStartTimestamp
              .countingFromTheFirstDebouncedSimulation
          : now,
      },
      prevResult: prev.isRunning ? prev.prevResult : prev.result,
    }))
    const timeout = window.setTimeout(
      () => {
        block(async () => {
          abortController.signal.throwIfAborted()
          const simulateOnServerResult = await simulateOnServer(
            abortController.signal,
            dailyMarketSeriesSrc,
            applyCache(
              planParamsHistoryUpToSimulationTimestamp,
              simulationTimestamp,
              ianaTimezoneNameIfDatedPlan
                ? (timestamp) =>
                    CalendarDayFns.fromTimestamp(
                      timestamp,
                      ianaTimezoneNameIfDatedPlan,
                    )
                : () => assertFalse(),
            ),
            [percentiles.low, percentiles.mid, percentiles.high],
            planParamsNorm,
            numOfSimulationForMonteCarloSampling,
            randomSeed,
          )
          abortController.signal.throwIfAborted()

          const result: SimulationResult = {
            ...simulateOnServerResult,
            // This is needed because when dealing with the result, we need to run
            // calculations based on the exact args of the result (eg, numMonths in
            // planParamsNorm) and not the latest one that is available on
            // SimulationInfo for which the result might still be pending. We could
            // put this in and args object, but adding "OfResult" to the name forces
            // us to think about the choice explicitly.
            planParamsNormOfResult: planParamsNorm,
            dailyMarketSeriesSrcOfResult: dailyMarketSeriesSrc,
            ianaTimezoneNameIfDatedPlanOfResult: ianaTimezoneNameIfDatedPlan,
            percentilesOfResult: percentiles,
            numOfSimulationForMonteCarloSamplingOfResult:
              numOfSimulationForMonteCarloSampling,
            randomSeedOfResult: randomSeed,
          }
          setState({ isRunning: false, result })
        }).catch((e) => {
          if (abortController.signal.aborted) return
          throw e
        })
      },
      Math.max(0, DEBOUNCE_TIME_MS - timeSinceLastRun),
    )

    return () => {
      window.clearTimeout(timeout)
      abortController.abort()
    }
  }, [
    planParamsHistoryUpToSimulationTimestamp,
    simulationTimestamp,
    ianaTimezoneNameIfDatedPlan,
    numOfSimulationForMonteCarloSampling,
    randomSeed,
    dailyMarketSeriesSrc,
    applyCache,
    planParamsNorm,
    percentiles,
  ])
  const isRunningInfo = useMemo(
    () =>
      state.isRunning
        ? ({
            isRunning: true,
            simulationStartTimestamp: state.simulationStartTimestamp,
          } as const)
        : ({ isRunning: false } as const),
    [state],
  )
  return {
    isRunningInfo,
    simulationResult: state.isRunning ? state.prevResult : state.result,
    planParamsId: lastHistoryItem.id,
    planParamsNorm,
  }
}
