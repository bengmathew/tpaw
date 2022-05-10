import {useEffect, useRef, useState} from 'react'
import {asyncEffect} from '../../Utils/AsyncEffect'
import {nominalToReal} from '../../Utils/NominalToReal'
import {fGet} from '../../Utils/Utils'
import {TPAWParams, ValueForYearRange} from '../TPAWParams'
import {TPAWParamsProcessed} from '../TPAWParamsProcessed'
import {TPAWRunInWorker, TPAWRunInWorkerResult} from './TPAWRunInWorker'

type _Args = {numRuns: number; params: TPAWParamsProcessed; percentiles: number[]}
export type UseTPAWWorkerResult = TPAWRunInWorkerResult & {args: _Args}

type State =
  | null
  | {isRunning: true; args: _Args; prevInfo: UseTPAWWorkerResult | null}
  | {isRunning: false; resultInfo: UseTPAWWorkerResult}

export function useTPAWWorker(
  params: TPAWParamsProcessed,
  numRuns: number,
  percentiles: number[]
) {
  const [state, setState] = useState<State>(null)

  const workerRef = useRef<TPAWRunInWorker | null>(null)
  useEffect(() => {
    workerRef.current = new TPAWRunInWorker()
    return () => workerRef.current?.terminate()
  }, [])

  useEffect(() => {
    const args = {numRuns, params, percentiles}
    setState(state => {
      return {
        isRunning: true,
        args,
        prevInfo: !state
          ? null
          : state.isRunning
          ? state.prevInfo
          : state.resultInfo,
      }
    })
  }, [numRuns, params, percentiles])

  useEffect(() => {
    if (!state || !state.isRunning) return
    return asyncEffect(async status => {
      const data = await fGet(workerRef.current).runSimulations(
        status,
        state.args.numRuns,
        state.args.params,
        state.args.percentiles
      )
      if (status.canceled) return
      setState({
        isRunning: false,
        resultInfo: {...fGet(data), args: state.args},
      })
    })
  }, [state])

  return !state
    ? ({isRunning: true, resultInfo: null} as const)
    : state.isRunning
    ? ({isRunning: true, resultInfo: state.prevInfo} as const)
    : ({isRunning: false, resultInfo: state.resultInfo} as const)
}
