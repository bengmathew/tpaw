import {
  DEFAULT_MONTE_CARLO_SIMULATION_SEED,
  assert,
  partialDefaultDatelessPlanParams,
  getDefaultNonPlanParams,
} from '@tpaw/common'
import { useMemo } from 'react'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'

export const useIsPlanInputDevSimulationsModified = () => {
  const { randomSeed, planParamsNorm } = useSimulation()
  const { nonPlanParams } = useNonPlanParams()
  const defaultNonPlanParams = useMemo(
    () => getDefaultNonPlanParams(Date.now()),
    [],
  )
  assert(partialDefaultDatelessPlanParams.advanced.sampling.type === 'monteCarlo')

  const isStaggerRunStartsModified =
    planParamsNorm.advanced.sampling.type === 'monteCarlo' &&
    planParamsNorm.advanced.sampling.data.staggerRunStarts !==
      partialDefaultDatelessPlanParams.advanced.sampling.data.staggerRunStarts

  const isOverrideToFixedForTestingModified =
    planParamsNorm.advanced.historicalReturnsAdjustment.standardDeviation
      .overrideToFixedForTesting !==
    partialDefaultDatelessPlanParams.advanced.historicalReturnsAdjustment.standardDeviation
      .overrideToFixedForTesting

  const isNumberOfSimulationsModified =
    nonPlanParams.numOfSimulationForMonteCarloSampling !==
    defaultNonPlanParams.numOfSimulationForMonteCarloSampling

  const isRandomSeedModified =
    randomSeed !== DEFAULT_MONTE_CARLO_SIMULATION_SEED
  return (
    isRandomSeedModified ||
    isStaggerRunStartsModified ||
    isNumberOfSimulationsModified ||
    isOverrideToFixedForTestingModified
  )
}
