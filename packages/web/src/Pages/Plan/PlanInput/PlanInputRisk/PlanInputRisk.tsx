import { noCase } from '@tpaw/common'
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
        {params.advanced.strategy === 'TPAW' ? (
          <PlanInputRiskTPAW props={props} />
        ) : params.advanced.strategy === 'SPAW' ? (
          <PlanInputRiskSPAW props={props} />
        ) : params.advanced.strategy === 'SWR' ? (
          <PlanInputRiskSWR props={props} />
        ) : (
          noCase(params.advanced.strategy)
        )}
      </>
    </PlanInputBody>
  )
})
