import { PlanParams } from '@tpaw/common'
import _ from 'lodash'
import { Dispatch, ReactNode, useMemo, useState } from 'react'
import {
  extendPlanParams,
  PlanParamsExt,
} from '../../TPAWSimulator/PlanParamsExt'
import {
  PlanParamsProcessed,
  processPlanParams,
} from '../../TPAWSimulator/PlanParamsProcessed'
import {
  useTPAWWorker,
  UseTPAWWorkerResult,
} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import { createContext } from '../../Utils/CreateContext'
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
  setNumRuns: Dispatch<number>
  setParams: (params: PlanParams | ((params: PlanParams) => PlanParams)) => void
} & Omit<SimulationInfoPerParam, 'tpawResult'> & {
    tpawResult: UseTPAWWorkerResult
  }

const [Context, useSimulation] = createContext<SimulationInfo>('Simulation')

// const highlightPercentiles = [5, 25, 50, 75, 95]
const highlightPercentiles = [5, 50, 95]
// const percentiles = [5, 50, 95]
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
  const marketData = useMarketData()

  const paramsProcessed = useMemo(
    () => processPlanParams(extendPlanParams(params), marketData),
    [params, marketData],
  )
  const tpawResult = useTPAWWorker(paramsProcessed, numRuns, percentiles)

  const value = useMemo(() => {
    return {
      setNumRuns,
      setParams,
      params,
      paramsProcessed,
      paramsExt: extendPlanParams(params),
      numRuns,
      percentiles,
      highlightPercentiles,
      tpawResult, // // Note, tpawResult will lag params. To get the exact params for the
      // result, use the params object inside tpawResult.
    }
  }, [numRuns, params, paramsProcessed, setParams, tpawResult])
  if (!_hasValue(value)) return <></>
  return <Context.Provider value={value}>{children}</Context.Provider>
}

const _hasValue = (x: {
  tpawResult: UseTPAWWorkerResult | null
}): x is { tpawResult: UseTPAWWorkerResult } => x.tpawResult !== null
