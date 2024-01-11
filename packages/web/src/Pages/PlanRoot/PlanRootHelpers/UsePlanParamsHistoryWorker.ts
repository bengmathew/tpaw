import { PlanParams, assert, fGet } from '@tpaw/common'
import _ from 'lodash'
import { useEffect, useState } from 'react'
import {
  SimulationWorkerArgs,
  SimulationWorkerResult,
} from '../../../UseSimulator/Simulator/SimulationWorkerAPI'
import { CurrentPortfolioBalance } from './CurrentPortfolioBalance'
import { useMarketData } from './WithMarketData'
import { useIANATimezoneName } from './WithNonPlanParams'

export const useParseAndMigratePlanParamsHistoryInWorker = (
  planParamsHistoryStr: readonly { id: string; params: string }[] | null,
) => {
  const [state, setState] = useState<
    | { type: 'initial' }
    | { type: 'processing'; taskID: string }
    | { type: 'processed'; result: { id: string; params: PlanParams }[] }
  >({ type: 'initial' })

  useEffect(() => {
    if (!planParamsHistoryStr) return

    const taskID = _.uniqueId()
    const worker = _getWorkerSingleton()
    const start = performance.now()
    const onMessage = ({ data }: MessageEvent<SimulationWorkerResult>): void =>
      setState((prev) => {
        if (prev.type !== 'processing' || prev.taskID !== data.taskID)
          return prev
        assert(data.type === 'parseAndMigratePlanParams')
        return { type: 'processed', result: data.result }
      })

    worker.addEventListener('message', onMessage)

    _planParamsCachedInWorker.clear()
    planParamsHistoryStr.forEach((x) => _planParamsCachedInWorker.add(x.id))
    const message: SimulationWorkerArgs = {
      type: 'parseAndMigratePlanParams',
      taskID: taskID,
      args: { isPreBase: true, planParamsHistoryStr },
    }
    worker.postMessage(message)
    setState({ type: 'processing', taskID })
    return () => worker.removeEventListener('message', onMessage)
  }, [planParamsHistoryStr])

  return state.type === 'processed' ? state.result : null
}

export const useCurrentPortfolioBalanceGetMonthInfoInWorker = (
  planId: string,
  isPreBase: boolean,
  planParamsHistory: readonly { id: string; params: PlanParams }[] | null,
) => {
  const { ianaTimezoneName } = useIANATimezoneName()
  const { marketData } = useMarketData()

  const [state, setState] = useState<
    | { type: 'initial' }
    | {
        type: 'processing'
        taskID: string
        planParamsHistory: readonly { id: string; params: PlanParams }[]
      }
    | {
        type: 'processed'
        result: CurrentPortfolioBalance.ByMonthInfo
        planParamsHistory: readonly { id: string; params: PlanParams }[]
      }
  >({ type: 'initial' })

  useEffect(() => {
    if (!planParamsHistory) return

    const taskID = _.uniqueId()
    const worker = _getWorkerSingleton()
    const start = performance.now()
    const onMessage = ({ data }: MessageEvent<SimulationWorkerResult>) =>
      setState((prev) => {
        if (prev.type !== 'processing' || prev.taskID !== data.taskID)
          return prev
        assert(data.type === 'estimateCurrentPortfolioBalance')
        return { type: 'processed', result: data.result, planParamsHistory }
      })
    worker.addEventListener('message', onMessage)

    const planParamsHistoryWithCacheInfo = planParamsHistory.map((x) =>
      _planParamsCachedInWorker.has(x.id)
        ? { cached: true as const, id: x.id }
        : { cached: false as const, id: x.id, params: x.params },
    )
    _planParamsCachedInWorker.clear()
    planParamsHistory.forEach((x) => _planParamsCachedInWorker.add(x.id))
    const estimationTimestamp = fGet(_.last(planParamsHistory)).params.timestamp
    const message: SimulationWorkerArgs = {
      type: 'estimateCurrentPortfolioBalance',
      taskID,
      args: {
        planId,
        isPreBase,
        planParamsHistory: planParamsHistoryWithCacheInfo,
        estimationTimestamp,
        ianaTimezoneName,
        marketData,
      },
    }
    worker.postMessage(message)
    setState({ type: 'processing', taskID, planParamsHistory })
    return () => {
      worker.removeEventListener('message', onMessage)
    }
    // Everything except planParamsHistory is const.
  }, [ianaTimezoneName, marketData, planId, planParamsHistory, isPreBase])

  return state.type === 'processed' &&
    state.planParamsHistory === planParamsHistory
    ? state.result
    : null
}

const _planParamsCachedInWorker = new Set<string>()
// SHould be singleton so estimate can resuse it's cache.
let _workerSingleton: Worker | null = null
const _getWorkerSingleton = () => {
  if (!_workerSingleton)
    _workerSingleton = new Worker(
      new URL('../../../UseSimulator/Simulator/SimulationWorker.ts', import.meta.url),
    )
  return _workerSingleton
}
