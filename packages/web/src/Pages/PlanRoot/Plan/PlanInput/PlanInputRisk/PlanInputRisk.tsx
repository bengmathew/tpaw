import { PlanParams, noCase } from '@tpaw/common'
import React from 'react'
import { PlanParamsExtended } from '../../../../../UseSimulator/ExtentPlanParams'
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

export const PlanInputRiskSummary = React.memo(
  ({
    planParamsExt,
    defaultPlanParams,
  }: {
    planParamsExt: PlanParamsExtended
    defaultPlanParams: PlanParams
  }) => {
    const { planParams } = planParamsExt
    switch (planParams.advanced.strategy) {
      case 'TPAW':
        return (
          <PlanInputRiskTPAWSummary
            planParams={planParams}
            defaultPlanParams={defaultPlanParams}
          />
        )
      case 'SPAW':
        return <PlanInputRiskSPAWSummary planParamsExt={planParamsExt} />
      case 'SWR':
        return <PlanInputRiskSWRSummary planParamsExt={planParamsExt} />
      default:
        noCase(planParams.advanced.strategy)
    }
  },
)
