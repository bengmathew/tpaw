import { assert } from '@tpaw/common'
import { useEffect, useMemo, useState } from 'react'
import { asyncEffect } from '../Utils/AsyncEffect'
import { PlanParamsNormalized } from './NormalizePlanParams/NormalizePlanParams'
import { PlanParamsProcessed } from './PlanParamsProcessed/PlanParamsProcessed'
import {
  SimulationArgs,
  SimulationResult,
  Simulator,
} from './Simulator/Simulator'
import * as Rust from '@tpaw/simulator'



// Singleton so this is created only one for speedup.
let _singleton: Simulator | null = null
export const getSimulatorSingleton = () => {
  if (!_singleton) _singleton = new Simulator()
  return _singleton
}

export function useSimulator(
  currentPortfolioBalanceAmount: number,
  planParamsRust: Rust.PlanParamsRust,
  marketData: Rust.DataForMarketBasedPlanParamValues,
  planParamsNorm: PlanParamsNormalized,
  planParamsProcessed: PlanParamsProcessed,
  numOfSimulationForMonteCarloSampling: number,
  randomSeed: number,
) {
  const [result, setResult] = useState<SimulationResult | null>(null)

  const args: SimulationArgs = useMemo(() => {
    return {
      currentPortfolioBalanceAmount,
      planParamsRust,
      marketData,
      planParamsNorm,
      planParamsProcessed,
      numOfSimulationForMonteCarloSampling,
      randomSeed,
    }
  }, [
    currentPortfolioBalanceAmount,
    numOfSimulationForMonteCarloSampling,
    planParamsRust,
    marketData,
    planParamsNorm,
    planParamsProcessed,
    randomSeed,
  ])
  useEffect(() => {
    return asyncEffect(async (status) => {
      const data = await getSimulatorSingleton().runSimulations(status, args)
      if (status.canceled) return
      assert(data)
      // sendAnalyticsEvent('simulation_time', { runTime: data.perf.main[6][1] })
      setResult(data)
    })
  }, [args])
  return { result, resultIsCurrent: !!result && result.args === args }
}
