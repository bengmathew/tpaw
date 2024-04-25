import {
  PlanParams,
  assert,
  fGet,
  getFullDatedDefaultPlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import React, { useCallback, useEffect, useState } from 'react'
import { appPaths } from '../../../AppPaths'
import { useCurrentTime } from '../PlanRootHelpers/UseCurrentTime'
import { useWorkingPlan } from '../PlanRootHelpers/UseWorkingPlan'
import { useIANATimezoneName } from '../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationParams,
  WithSimulation,
  useSimulationParamsForPlanMode,
} from '../PlanRootHelpers/WithSimulation'
import { PlanLocalStorage } from './PlanLocalStorage'

export const PlanRootLocalImpl = React.memo(
  ({
    reload,
    pdfReportInfo,
  }: {
    reload: () => void
    pdfReportInfo: SimulationParams['pdfReportInfo']
  }) => {
    const { ianaTimezoneName } = useIANATimezoneName()

    const [startingSrc] = useState(
      () =>
        PlanLocalStorage.read() ??
        PlanLocalStorage.getDefault(
          getFullDatedDefaultPlanParams(Date.now(), ianaTimezoneName),
        ),
    )
    const planPaths = appPaths.guest
    const currentTimeInfo = useCurrentTime({ planId: startingSrc.state.planId })
    assert(
      currentTimeInfo.currentTimestamp >=
        fGet(_.last(startingSrc.state.planParamsPostBase)).params.timestamp,
    )
    const workingPlanInfo = useWorkingPlan(
      currentTimeInfo,
      startingSrc.state,
      planPaths,
    )

    const rebase = workingPlanInfo.rebase
    useEffect(() => {
      rebase?.({ hard: true })
    }, [rebase])

    useEffect(() => {
      PlanLocalStorage.write({
        v: 1,
        planId: workingPlanInfo.workingPlan.planId,
        planParamsPostBase: workingPlanInfo.workingPlan.planParamsPostBase,
        reverseHeadIndex: workingPlanInfo.workingPlan.reverseHeadIndex,
      })
    }, [workingPlanInfo.workingPlan])

    const reset = useCallback(
      (planParams: PlanParams) => {
        PlanLocalStorage.clear()
        PlanLocalStorage.write(PlanLocalStorage.getDefault(planParams).state)
        reload()
      },
      [reload],
    )

    const simulationParams = useSimulationParamsForPlanMode(
      planPaths,
      currentTimeInfo,
      workingPlanInfo,
      startingSrc.planMigratedFromVersion,
      null,
      { src: 'localMain', reset },
      pdfReportInfo,
    )

    return <WithSimulation params={simulationParams} />
  },
)
