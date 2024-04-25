import { PlanParams, partialDefaultDatelessPlanParams } from '@tpaw/common'
import _ from 'lodash'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'

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
  const { planParamsNorm } = useSimulation()
  return !_.isEqual(
    partialDefaultDatelessPlanParams.advanced.annualInflation,
    planParamsNorm.advanced.annualInflation,
  )
}
