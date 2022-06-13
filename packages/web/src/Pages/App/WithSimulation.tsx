import _ from 'lodash'
import React, {Dispatch, ReactNode, useMemo, useState} from 'react'
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
import {fGet, noCase} from '../../Utils/Utils'
import {useTPAWParams} from './UseTPAWParams'

export type SimulationInfoPerParam = {
  params: TPAWParams
  paramsProcessed: TPAWParamsProcessed
  paramsExt: TPAWParamsExt
  tpawResult: UseTPAWWorkerResult | null
  numRuns: number
  percentiles: number[]
  highlightPercentiles: number[]
}

export type SimulationInfo = {
  paramSpace: 'a' | 'b'
  setParamSpace: (space: 'a' | 'b') => void
  setByStrategy: Dispatch<boolean>
  setParams: (params: TPAWParams | ((params: TPAWParams) => TPAWParams)) => void
  byStrategy: {
    tpaw: SimulationInfoPerParam
    spaw: SimulationInfoPerParam
  } | null
} & Omit<SimulationInfoPerParam, 'tpawResult'> & {
    tpawResult: UseTPAWWorkerResult
  }

const [Context, useSimulation] = createContext<SimulationInfo>('Simulation')

const numRuns = 500
const highlightPercentiles = [5, 25, 50, 75, 95]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))

export {useSimulation}

export const WithSimulation = ({children}: {children: ReactNode}) => {
  const {paramSpace, setParamSpace, params, setParams} = useTPAWParams()
  const [byStrategy, setByStrategy] = useState(false)
  const forBaseParams = fGet(useForParams(params))
  const altStrategy: TPAWParams['strategy'] =
    params.strategy === 'TPAW'
      ? 'SPAW'
      : params.strategy === 'SPAW'
      ? 'TPAW'
      : noCase(params.strategy)
  const forAltParams = useForParams(
    byStrategy ? {...params, strategy: altStrategy} : null
  )

  const value = useMemo(
    () => ({
      paramSpace,
      setParamSpace,
      setParams,
      ...forBaseParams,
      byStrategy: forAltParams
        ? {
            tpaw: altStrategy === 'TPAW' ? forAltParams : forBaseParams,
            spaw: altStrategy === 'SPAW' ? forAltParams : forBaseParams,
          }
        : null,
      setByStrategy,
    }),
    [
      altStrategy,
      forAltParams,
      forBaseParams,
      paramSpace,
      setParamSpace,
      setParams,
    ]
  )
  if (!_hasValue(value)) return <></>
  return <Context.Provider value={value}>{children}</Context.Provider>
}

const _hasValue = (x: {
  tpawResult: UseTPAWWorkerResult | null
}): x is {tpawResult: UseTPAWWorkerResult} => x.tpawResult !== null

function useForParams(
  params: TPAWParams | null
): SimulationInfoPerParam | null {
  const paramsProcessed = useMemo(
    () => (params ? processTPAWParams(params) : null),
    [params]
  )
  const tpawResult = useTPAWWorker(paramsProcessed, numRuns, percentiles)
  return useMemo(
    () =>
      params
        ? {
            params,
            paramsProcessed: fGet(paramsProcessed),
            paramsExt: extendTPAWParams(params),
            numRuns,
            percentiles,
            highlightPercentiles,
            // Note, tpawResult will lag params. To get the exact params for the
            // result, use the params object inside tpawResult.
            tpawResult,
          }
        : null,
    [params, paramsProcessed, tpawResult]
  )
}
