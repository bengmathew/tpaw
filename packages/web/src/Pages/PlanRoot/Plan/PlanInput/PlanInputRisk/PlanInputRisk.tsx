import { noCase } from '@tpaw/common'
import React from 'react'
import { PlanParamsNormalized } from '../../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
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
  const { planParamsNorm } = useSimulation()

  return (
    <PlanInputBody {...props}>
      <>
        {planParamsNorm.advanced.strategy === 'TPAW' ? (
          <PlanInputRiskTPAW props={props} />
        ) : planParamsNorm.advanced.strategy === 'SPAW' ? (
          <PlanInputRiskSPAW props={props} />
        ) : planParamsNorm.advanced.strategy === 'SWR' ? (
          <PlanInputRiskSWR props={props} />
        ) : (
          noCase(planParamsNorm.advanced.strategy)
        )}
      </>
    </PlanInputBody>
  )
})

export const PlanInputRiskSummary = React.memo(
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    switch (planParamsNorm.advanced.strategy) {
      case 'TPAW':
        return <PlanInputRiskTPAWSummary planParamsNorm={planParamsNorm} />
      case 'SPAW':
        return <PlanInputRiskSPAWSummary planParamsNorm={planParamsNorm} />
      case 'SWR':
        return <PlanInputRiskSWRSummary planParamsNorm={planParamsNorm} />
      default:
        noCase(planParamsNorm.advanced.strategy)
    }
  },
)
