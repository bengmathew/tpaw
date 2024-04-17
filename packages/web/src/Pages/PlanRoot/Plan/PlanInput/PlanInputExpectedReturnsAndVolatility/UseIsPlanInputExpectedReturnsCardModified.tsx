import _ from 'lodash'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'


export const useIsPlanInputExpectedReturnsAndVolatilityModified = () => {
  const isExpectedCardModified = useIsPlanInputExpectedReturnsCardModified()
  const isBondVolatilityModified = useIsPlanInputBondVolatilityCardModified()
  const isStockVolatilityModified = useIsPlanInputStockVolatilityCardModified()

  return (
    isExpectedCardModified ||
    isBondVolatilityModified ||
    isStockVolatilityModified
  )
}

export const useIsPlanInputExpectedReturnsCardModified = () => {
  const { planParamsNorm, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.returnsStatsForPlanning.expectedValue,
    planParamsNorm.advanced.returnsStatsForPlanning.expectedValue
  )
}
export const useIsPlanInputStockVolatilityCardModified = () => {
  const { planParamsNorm, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.returnsStatsForPlanning.standardDeviation.stocks,
    planParamsNorm.advanced.returnsStatsForPlanning.standardDeviation.stocks
  )
}

export const useIsPlanInputBondVolatilityCardModified = () => {
  const { planParamsNorm, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.historicalReturnsAdjustment.standardDeviation
      .bonds,
    planParamsNorm.advanced.historicalReturnsAdjustment.standardDeviation.bonds
  )
}
