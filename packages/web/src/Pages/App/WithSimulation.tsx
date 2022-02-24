import _ from 'lodash'
import React, {ReactNode, useMemo} from 'react'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {
  useTPAWWorker,
  UseTPAWWorkerResult,
} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import {createContext} from '../../Utils/CreateContext'
import {StateObj} from '../../Utils/UseStateObj'
import {useTPAWParams} from './UseTPAWParams'

export type SimulationInfo = {
  params: StateObj<TPAWParams>['value']
  setParams: StateObj<TPAWParams>['set']
  tpawResult: UseTPAWWorkerResult
  numRuns: number
  percentiles: number[]
  highlightPercentiles: number[]
}
const [Context, useSimulation] = createContext<SimulationInfo>('TPAW')

const numRuns = 500
const highlightPercentiles = [5, 25, 50, 75, 95]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))

export {useSimulation}

export const WithSimulation = ({children}: {children: ReactNode}) => {
  const {params, setParams} = useTPAWParams()
  const {resultInfo: tpawResult} = useTPAWWorker(params, numRuns, percentiles)
  const value = useMemo(
    () => ({
      params,
      setParams,
      numRuns,
      tpawResult, // Note, tpawResult will lag params. To get the exact params for the result, use the params object inside tpawResult.
      percentiles,
      highlightPercentiles,
    }),
    [params, setParams, tpawResult]
  )
  if (!_hasValue(value)) return <></>
  return <Context.Provider value={value}>{children}</Context.Provider>
}

const _hasValue = (x: {
  tpawResult: UseTPAWWorkerResult | null
}): x is {tpawResult: UseTPAWWorkerResult} => x.tpawResult !== null
