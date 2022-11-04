import { PlanParams, resolveTPAWRiskPreset } from '@tpaw/common'
import _ from 'lodash'
import { Dispatch, ReactNode, useMemo, useState } from 'react'
import {
  extendPlanParams,
  getNumYears,
  PlanParamsExt
} from '../../TPAWSimulator/PlanParamsExt'
import {
  PlanParamsProcessed,
  processPlanParams
} from '../../TPAWSimulator/PlanParamsProcessed'
import {
  useTPAWWorker,
  UseTPAWWorkerResult
} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import { createContext } from '../../Utils/CreateContext'
import { fGet } from '../../Utils/Utils'
import { usePlanParams } from './UsePlanParams'
import { useMarketData } from './WithMarketData'

export type SimulationInfoPerParam = {
  params: PlanParams
  paramsProcessed: PlanParamsProcessed
  paramsExt: PlanParamsExt
  tpawResult: UseTPAWWorkerResult | null
  numRuns: number
  percentiles: number[]
  highlightPercentiles: number[]
}

export type SimulationInfo = {
  setCompareRewardRiskRatio: (x: boolean) => void
  setNumRuns: Dispatch<number>
  setParams: (params: PlanParams | ((params: PlanParams) => PlanParams)) => void
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

export { useSimulation }

import React from 'react'
import { WithURLPlanParams } from './WithURLPlanParams'

export const WithSimulation = React.memo(
  ({ children }: { children: ReactNode }) => {
    return (
      <WithURLPlanParams>
        <_Body>{children}</_Body>
      </WithURLPlanParams>
    )
  },
)

const _Body = ({ children }: { children: ReactNode }) => {
  const [numRuns, setNumRuns] = useState(500)
  const { params, setParams } = usePlanParams()
  const [compareRewardRiskRatio, setCompareRewardRiskRatio] = useState(false)

  const paramsForRewardRiskRatioComparison: {
    tpaw: PlanParams
    spaw: PlanParams
    swr: PlanParams
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
      tpaw: { ...clone, strategy: 'TPAW' },
      spaw: { ...clone, strategy: 'SPAW' },
      swr: { ...clone, strategy: 'SWR' },
    }
  }, [compareRewardRiskRatio, params])

  const forTPAWRewardRiskRatio = useForParams(
    paramsForRewardRiskRatioComparison?.tpaw ?? null,
    numRuns,
  )
  const forSPAWRewardRiskRatio = useForParams(
    paramsForRewardRiskRatioComparison?.spaw ?? null,
    numRuns,
  )
  const forSWRRewardRiskRatio = useForParams(
    paramsForRewardRiskRatioComparison?.swr ?? null,
    numRuns,
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
    ],
  )

  const forBase = fGet(useForParams(params, numRuns))

  const value = useMemo(() => {
    return {
      setNumRuns,
      setParams,
      ...forBase,
      forRewardRiskRatioComparison,
      setCompareRewardRiskRatio,
    }
  }, [forBase, forRewardRiskRatioComparison, setParams])
  if (!_hasValue(value)) return <></>
  return <Context.Provider value={value}>{children}</Context.Provider>
}

const _hasValue = (x: {
  tpawResult: UseTPAWWorkerResult | null
}): x is { tpawResult: UseTPAWWorkerResult } => x.tpawResult !== null

function useForParams(
  params: PlanParams | null,
  numRuns: number,
): SimulationInfoPerParam | null {
  const marketData = useMarketData()
  const paramsProcessed = useMemo(
    () =>
      params ? processPlanParams(extendPlanParams(params), marketData) : null,
    [params, marketData],
  )
  const tpawResult = useTPAWWorker(paramsProcessed, numRuns, percentiles)
  return useMemo(
    () =>
      params
        ? {
            params,
            paramsProcessed: fGet(paramsProcessed),
            paramsExt: extendPlanParams(params),
            numRuns,
            percentiles,
            highlightPercentiles,
            // Note, tpawResult will lag params. To get the exact params for the
            // result, use the params object inside tpawResult.
            tpawResult,
          }
        : null,
    [numRuns, params, paramsProcessed, tpawResult],
  )
}
