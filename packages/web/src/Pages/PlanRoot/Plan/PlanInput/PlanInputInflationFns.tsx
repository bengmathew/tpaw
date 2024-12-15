import { PlanParams, partialDefaultDatelessPlanParams } from '@tpaw/common'
import _ from 'lodash'
import { useSimulationInfo } from '../../PlanRootHelpers/WithSimulation'

export const inflationTypeLabel = ({
  type,
}: {
  type: PlanParams['advanced']['annualInflation']['type']
}) => {
  switch (type) {
    case 'suggested':
      return 'Breakeven Inflation Rate'
    case 'manual':
      return 'Manual'
  }
}

export const useIsPlanInputInflationModified = () => {
  const { planParamsNormInstant } = useSimulationInfo()
  return !_.isEqual(
    partialDefaultDatelessPlanParams.advanced.annualInflation,
    planParamsNormInstant.advanced.annualInflation,
  )
}
