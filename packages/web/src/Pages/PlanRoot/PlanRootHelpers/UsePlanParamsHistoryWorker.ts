import { PlanParams, assert } from '@tpaw/common'
import _ from 'lodash'
import { useEffect, useState } from 'react'
import {
  SimulationWorkerArgs,
  SimulationWorkerResult,
} from '../../../Simulator/Simulator/SimulationWorkerAPI'

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
    const onMessage = ({
      data,
    }: MessageEvent<SimulationWorkerResult>): void => {
      if (data.type === 'error') {
        throw new Error(`Error in worker: ${data.message}`)
      }
      setState((prev) => {
        if (prev.type !== 'processing' || prev.taskID !== data.taskID)
          return prev
        assert(data.type === 'parseAndMigratePlanParams')
        return { type: 'processed', result: data.result }
      })
    }

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

const _planParamsCachedInWorker = new Set<string>()
// SHould be singleton so estimate can resuse it's cache.
let _workerSingleton: Worker | null = null
const _getWorkerSingleton = () => {
  if (!_workerSingleton)
    _workerSingleton = new Worker(
      new URL('../../../Simulator/Simulator/SimulationWorker', import.meta.url),
    )
  return _workerSingleton
}
