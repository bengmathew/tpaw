import {
  DEFAULT_MONTE_CARLO_SIMULATION_SEED,
  getDefaultNonPlanParams,
} from '@tpaw/common'
import { useMemo } from 'react'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'

export const useIsPlanInputDevSimulationsModified = () => {
  const { randomSeed, planParamsNorm, defaultPlanParams } = useSimulation()
  const { nonPlanParams } = useNonPlanParams()
  const defaultNonPlanParams = useMemo(
    () => getDefaultNonPlanParams(Date.now()),
    [],
  )
  return (
    randomSeed !== DEFAULT_MONTE_CARLO_SIMULATION_SEED ||
    planParamsNorm.advanced.sampling.forMonteCarlo.staggerRunStarts !==
      defaultPlanParams.advanced.sampling.forMonteCarlo.staggerRunStarts ||
    nonPlanParams.numOfSimulationForMonteCarloSampling !==
      defaultNonPlanParams.numOfSimulationForMonteCarloSampling ||
    planParamsNorm.advanced.historicalMonthlyLogReturnsAdjustment
      .overrideToFixedForTesting !==
      defaultPlanParams.advanced.historicalMonthlyLogReturnsAdjustment
        .overrideToFixedForTesting
  )
}
