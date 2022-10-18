import _ from 'lodash'
import {Dispatch, ReactNode, useMemo, useState} from 'react'
import {resolveTPAWRiskPreset} from '../../TPAWSimulator/DefaultParams'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {
  extendTPAWParams,
  getNumYears,
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
import {useMarketData} from './WithMarketData'

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
  setCompareRewardRiskRatio: (x: boolean) => void
  setNumRuns: Dispatch<number>
  setParams: (params: TPAWParams | ((params: TPAWParams) => TPAWParams)) => void
  forRewardRiskRatioComparison: {
    tpaw: SimulationInfoPerParam
    spaw: SimulationInfoPerParam
    swr: SimulationInfoPerParam
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
  const [compareRewardRiskRatio, setCompareRewardRiskRatio] = useState(false)

  const paramsForRewardRiskRatioComparison: {
    tpaw: TPAWParams
    spaw: TPAWParams
    swr: TPAWParams
  } | null = useMemo(() => {
    if (!compareRewardRiskRatio) return null
    const clone = _.cloneDeep(params)
    clone.extraSpending = {
      essential: [],
      discretionary: [],
    }

    clone.legacy.tpawAndSPAW = {
      external: [],
      total: 0,
    }
    clone.risk = resolveTPAWRiskPreset(params.risk, getNumYears(params))
    clone.risk.tpawAndSPAW.spendingTilt = 0
    clone.risk.tpawAndSPAW.spendingCeiling = null
    clone.risk.tpawAndSPAW.spendingFloor = null
    return {
      tpaw: {...clone, strategy: 'TPAW'},
      spaw: {...clone, strategy: 'SPAW'},
      swr: {...clone, strategy: 'SWR'},
    }
  }, [compareRewardRiskRatio, params])

  const forTPAWRewardRiskRatio = useForParams(
    paramsForRewardRiskRatioComparison?.tpaw ?? null,
    numRuns
  )
  const forSPAWRewardRiskRatio = useForParams(
    paramsForRewardRiskRatioComparison?.spaw ?? null,
    numRuns
  )
  const forSWRRewardRiskRatio = useForParams(
    paramsForRewardRiskRatioComparison?.swr ?? null,
    numRuns
  )

  const forRewardRiskRatioComparison = useMemo(
    () =>
      compareRewardRiskRatio
        ? {
            tpaw: fGet(forTPAWRewardRiskRatio),
            spaw: fGet(forSPAWRewardRiskRatio),
            swr: fGet(forSWRRewardRiskRatio),
          }
        : null,
    [
      forSPAWRewardRiskRatio,
      forTPAWRewardRiskRatio,
      forSWRRewardRiskRatio,
      compareRewardRiskRatio,
    ]
  )

  const forBase = fGet(useForParams(params, numRuns))

  const value = useMemo(() => {
    return {
      paramSpace,
      setParamSpace,
      setNumRuns,
      setParams,
      ...forBase,
      forRewardRiskRatioComparison,
      setCompareRewardRiskRatio,
    }
  }, [
    forBase,
    forRewardRiskRatioComparison,
    paramSpace,
    setParamSpace,
    setParams,
  ])
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
  const marketData = useMarketData()
  const paramsProcessed = useMemo(
    () =>
      params ? processTPAWParams(extendTPAWParams(params), marketData) : null,
    [params, marketData]
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
