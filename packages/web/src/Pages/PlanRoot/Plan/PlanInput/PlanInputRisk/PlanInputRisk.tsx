import { noCase } from '@tpaw/common'
import React from 'react'
import { PlanParamsNormalized } from '../../../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import {
  PlanInputRiskSPAW,
  PlanInputRiskSPAWSummary,
} from './PlanInputRiskSPAW'
import { PlanInputRiskSWR, PlanInputRiskSWRSummary } from './PlanInputRiskSWR'
import {
  PlanInputRiskTPAW,
  PlanInputRiskTPAWSummary,
} from './PlanInputRiskTPAW'

export const PlanInputRisk = React.memo((props: PlanInputBodyPassThruProps) => {
  const { planParamsNormInstant } = useSimulationInfo()

  return (
    <PlanInputBody {...props}>
      <>
        {planParamsNormInstant.advanced.strategy === 'TPAW' ? (
          <PlanInputRiskTPAW props={props} />
        ) : planParamsNormInstant.advanced.strategy === 'SPAW' ? (
          <PlanInputRiskSPAW props={props} />
        ) : planParamsNormInstant.advanced.strategy === 'SWR' ? (
          <PlanInputRiskSWR props={props} />
        ) : (
          noCase(planParamsNormInstant.advanced.strategy)
        )}
      </>
    </PlanInputBody>
  )
})

export const PlanInputRiskSummary = React.memo(
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    switch (planParamsNorm.advanced.strategy) {
      case 'TPAW':
        return <PlanInputRiskTPAWSummary planParamsNormInstant={planParamsNorm} />
      case 'SPAW':
        return (
          <PlanInputRiskSPAWSummary planParamsNormInstant={planParamsNorm} />
        )
      case 'SWR':
        return (
          <PlanInputRiskSWRSummary planParamsNormInstant={planParamsNorm} />
        )
      default:
        noCase(planParamsNorm.advanced.strategy)
    }
  },
)
