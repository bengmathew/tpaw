import _ from 'lodash'
import React, {ReactNode, useMemo} from 'react'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {
  extendTPAWParams,
  TPAWParamsExt,
} from '../../TPAWSimulator/TPAWParamsExt'
import {
  processTPAWParams,
  TPAWParamsProcessed,
} from '../../TPAWSimulator/TPAWParamsProcessed'
import {
  useTPAWWorker,
  UseTPAWWorkerResult,
} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import {createContext} from '../../Utils/CreateContext'
import {useTPAWParams} from './UseTPAWParams'

export type SimulationInfo = {
  paramSpace: 'a' | 'b'
  setParamSpace: (space: 'a' | 'b') => void
  params: TPAWParams
  paramsProcessed: TPAWParamsProcessed
  paramsExt: TPAWParamsExt
  setParams: (params: TPAWParams | ((params: TPAWParams) => TPAWParams)) => void
  tpawResult: UseTPAWWorkerResult
  numRuns: number
  percentiles: number[]
  highlightPercentiles: number[]
}
const [Context, useSimulation] = createContext<SimulationInfo>('Simulation')

const numRuns = 500
const highlightPercentiles = [5, 25, 50, 75, 95]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))

export {useSimulation}

export const WithSimulation = ({children}: {children: ReactNode}) => {
  const {paramSpace, setParamSpace, params, setParams} = useTPAWParams()
  const paramsProcessed = useMemo(() => processTPAWParams(params), [params])
  const {resultInfo: tpawResult} = useTPAWWorker(
    paramsProcessed,
    numRuns,
    percentiles
  )
  const value = useMemo(
    () => ({
      paramSpace,
      setParamSpace,
      params,
      paramsProcessed: processTPAWParams(params),
      paramsExt: extendTPAWParams(params),
      setParams,
      numRuns,
      percentiles,
      highlightPercentiles,
      // Note, tpawResult will lag params. To get the exact params for the
      // result, use the params object inside tpawResult.
      tpawResult: tpawResult ?? null,
    }),
    [paramSpace, params, setParamSpace, setParams, tpawResult]
  )
  if (!_hasValue(value)) return <></>
  return <Context.Provider value={value}>{children}</Context.Provider>
}

const _hasValue = (x: {
  tpawResult: UseTPAWWorkerResult | null
}): x is {tpawResult: UseTPAWWorkerResult} => x.tpawResult !== null
