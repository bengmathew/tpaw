import {
  assert,
  partialDefaultDatelessPlanParams,
  getDefaultNonPlanParams,
} from '@tpaw/common'
import { useMemo } from 'react'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import { DEFAULT_MONTE_CARLO_SIMULATION_SEED, useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'
import _ from 'lodash'

export const useIsPlanInputDevOverrideHistoricalReturnsToFixedForTestingModified =
  () => {
    const { planParamsNormInstant } = useSimulationInfo()
    return !_.isEqual(
      planParamsNormInstant.advanced.historicalReturnsAdjustment
        .overrideToFixedForTesting,
      partialDefaultDatelessPlanParams.advanced.historicalReturnsAdjustment
        .overrideToFixedForTesting,
    )
  }

export const useIsPlanInputDevSimulationsMainCardModified = () => {
  const { randomSeed, planParamsNormInstant } = useSimulationInfo()
  const { nonPlanParams } = useNonPlanParams()
  const defaultNonPlanParams = useMemo(
    () => getDefaultNonPlanParams(Date.now()),
    [],
  )
  assert(
    partialDefaultDatelessPlanParams.advanced.sampling.type === 'monteCarlo',
  )

  const isStaggerRunStartsModified =
    planParamsNormInstant.advanced.sampling.type === 'monteCarlo' &&
    planParamsNormInstant.advanced.sampling.data.staggerRunStarts !==
      partialDefaultDatelessPlanParams.advanced.sampling.data.staggerRunStarts

  const isNumberOfSimulationsModified =
    nonPlanParams.numOfSimulationForMonteCarloSampling !==
    defaultNonPlanParams.numOfSimulationForMonteCarloSampling

  const isRandomSeedModified =
    randomSeed !== DEFAULT_MONTE_CARLO_SIMULATION_SEED
  return (
    isRandomSeedModified ||
    isStaggerRunStartsModified ||
    isNumberOfSimulationsModified
  )
}
export const useIsPlanInputDevSimulationsModified = () => {
  const isMainCardModified = useIsPlanInputDevSimulationsMainCardModified()
  const isOverrideToFixedForTestingModified =
    useIsPlanInputDevOverrideHistoricalReturnsToFixedForTestingModified()

  return isMainCardModified || isOverrideToFixedForTestingModified
}
