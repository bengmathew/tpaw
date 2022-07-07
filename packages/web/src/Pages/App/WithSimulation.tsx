import _ from 'lodash'
import {Dispatch, ReactNode, useMemo, useState} from 'react'
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
import {fGet} from '../../Utils/Utils'
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
  setCompareSharpeRatio: (x: boolean) => void
  setNumRuns: Dispatch<number>
  setParams: (params: TPAWParams | ((params: TPAWParams) => TPAWParams)) => void
  forSharpeRatioComparison: {
    tpaw: SimulationInfoPerParam
    spaw: SimulationInfoPerParam
  } | null
} & Omit<SimulationInfoPerParam, 'tpawResult'> & {
    tpawResult: UseTPAWWorkerResult
  }

const [Context, useSimulation] = createContext<SimulationInfo>('Simulation')

const highlightPercentiles = [5, 25, 50, 75, 95]
// const highlightPercentiles = [10, 90]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))

export {useSimulation}

export const WithSimulation = ({children}: {children: ReactNode}) => {
  const [numRuns, setNumRuns] = useState(500)
  const {paramSpace, setParamSpace, params, setParams} = useTPAWParams()
  const [compareSharpeRatio, setCompareSharpeRatio] = useState(false)

  const paramsForSharpeRatioComparison: {
    tpaw: TPAWParams
    spaw: TPAWParams
  } | null = useMemo(() => {
    if (!compareSharpeRatio) return null
    const clone = _.cloneDeep(params)
    clone.legacy = {external: [], total: 0}
    clone.withdrawals = {
      essential: [],
      discretionary: [],
      lmp: clone.withdrawals.lmp,
    }
    clone.spendingCeiling = null
    clone.spendingFloor = null
    return {
      tpaw: {...clone, strategy: 'TPAW'},
      spaw: {...clone, strategy: 'SPAW'},
    }
  }, [compareSharpeRatio, params])

  const forTPAWSharpeRatio = useForParams(
    paramsForSharpeRatioComparison?.tpaw ?? null,
    numRuns
  )
  const forSPAWSharpeRatio = useForParams(
    paramsForSharpeRatioComparison?.spaw ?? null,
    numRuns
  )

  const forSharpeRatioComparison = useMemo(
    () =>
      compareSharpeRatio
        ? {tpaw: fGet(forTPAWSharpeRatio), spaw: fGet(forSPAWSharpeRatio)}
        : null,
    [forSPAWSharpeRatio, forTPAWSharpeRatio, compareSharpeRatio]
  )

  const forBase = fGet(useForParams(params, numRuns))

  const value = useMemo(() => {
    return {
      paramSpace,
      setParamSpace,
      setNumRuns,
      setParams,
      ...forBase,
      forSharpeRatioComparison,
      setCompareSharpeRatio,
    }
  }, [forBase, forSharpeRatioComparison, paramSpace, setParamSpace, setParams])
  if (!_hasValue(value)) return <></>
  return <Context.Provider value={value}>{children}</Context.Provider>
}

const _hasValue = (x: {
  tpawResult: UseTPAWWorkerResult | null
}): x is {tpawResult: UseTPAWWorkerResult} => x.tpawResult !== null

function useForParams(
  params: TPAWParams | null,
  numRuns: number
): SimulationInfoPerParam | null {
  const paramsProcessed = useMemo(
    () => (params ? processTPAWParams(extendTPAWParams(params)) : null),
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
    [numRuns, params, paramsProcessed, tpawResult]
  )
}
