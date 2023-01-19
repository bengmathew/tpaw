import React from 'react'
import { useSimulation } from '../../../App/WithSimulation'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import { PlanInputRiskSPAW } from './PlanInputRiskSPAW'
import { PlanInputRiskSWR } from './PlanInputRiskSWR'
import { PlanInputRiskTPAW } from './PlanInputRiskTPAW'

export const PlanInputRisk = React.memo((props: PlanInputBodyPassThruProps) => {
  const { params } = useSimulation()

  return (
    <PlanInputBody {...props}>
      <>
        {params.strategy === 'TPAW' ? (
          <PlanInputRiskTPAW props={props} />
        ) : params.strategy === 'SPAW' ? (
          <PlanInputRiskSPAW props={props} />
        ) : (
          <PlanInputRiskSWR props={props} />
        )}
      </>
    </PlanInputBody>
  )
})
