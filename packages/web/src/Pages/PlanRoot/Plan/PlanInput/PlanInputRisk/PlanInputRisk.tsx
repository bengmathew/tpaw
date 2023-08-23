import { noCase } from '@tpaw/common'
import React from 'react'
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
  const { planParams } = useSimulation()

  return (
    <PlanInputBody {...props}>
      <>
        {planParams.advanced.strategy === 'TPAW' ? (
          <PlanInputRiskTPAW props={props} />
        ) : planParams.advanced.strategy === 'SPAW' ? (
          <PlanInputRiskSPAW props={props} />
        ) : planParams.advanced.strategy === 'SWR' ? (
          <PlanInputRiskSWR props={props} />
        ) : (
          noCase(planParams.advanced.strategy)
        )}
      </>
    </PlanInputBody>
  )
})

export const PlanInputRiskSummary = React.memo(() => {
  const { planParams } = useSimulation()
  switch (planParams.advanced.strategy) {
    case 'TPAW':
      return <PlanInputRiskTPAWSummary />
    case 'SPAW':
      return <PlanInputRiskSPAWSummary />
    case 'SWR':
      return <PlanInputRiskSWRSummary />
    default:
      noCase(planParams.advanced.strategy)
  }
})
