import {
  PlanParams,
  SomePlanParams,
  assert,
  fGet,
  planParamsMigrate,
} from '@tpaw/common'
import _ from 'lodash'
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import * as uuid from 'uuid'
import { appPaths } from '../../../AppPaths'
import { useNavGuard } from '../../../Utils/UseNavGuard'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { useCurrentTime } from '../PlanRootHelpers/UseCurrentTime'
import {
  WorkingPlanSrc,
  useWorkingPlan,
} from '../PlanRootHelpers/UseWorkingPlan'
import {
  SimulationParams,
  WithSimulation,
  useSimulationParamsForPlanMode,
} from '../PlanRootHelpers/WithSimulation'
import { PlanRootLinkUnsavedWarningAlert } from './PlanRootLinkUnsavedWarningAlert'

export const PlanRootLinkImpl = React.memo(
  ({
    startingParams,
    startingParamsOverride,
    reset,
    pdfReportInfo,
  }: {
    startingParams: SomePlanParams
    startingParamsOverride: PlanParams | null
    reset: (planParams: PlanParams | null) => void
    pdfReportInfo: SimulationParams['pdfReportInfo']
  }) => {
    const planPaths = appPaths.link
    const [startingParamsMigrated] = useState(() =>
      planParamsMigrate(startingParams),
    )

    const [startingSrc] = useState<WorkingPlanSrc>(() => ({
      planId: uuid.v4(),
      planParamsPostBase: [
        {
          id: uuid.v4(),
          change: { type: 'startFromURL', value: null },
          params: startingParamsOverride ?? startingParamsMigrated,
        },
      ],
      reverseHeadIndex: 0,
    }))
    const currentTimeInfo = useCurrentTime({ planId: startingSrc.planId })
    assert(
      currentTimeInfo.currentTimestamp >=
        fGet(_.last(startingSrc.planParamsPostBase)).params.timestamp,
    )
    const workingPlanInfo = useWorkingPlan(
      currentTimeInfo,
      startingSrc,
      planPaths,
    )

    
    const isModified =
      fGet(_.last(workingPlanInfo.planParamsUndoRedoStack.undos)).params !==
      startingParamsMigrated

    const [forceNav, setForceNav] = useState(false)

    const { navGuardState, resetNavGuardState } = useNavGuard(
      isModified && !forceNav,
      planPaths,
    )

    const rebase = workingPlanInfo.rebase
    useEffect(() => {
      rebase?.({ hard: true })
    }, [rebase])

    const simulationParams = useSimulationParamsForPlanMode(
      planPaths,
      currentTimeInfo,
      workingPlanInfo,
      'v' in startingParams ? startingParams.v : 1,
      null,
      {
        src: 'link',
        isModified,
        reset,
        setForceNav: () => setForceNav(true),
      },
      pdfReportInfo,
    )
    const urlUpdater = useURLUpdater()

    return (
      <>
        <WithSimulation params={simulationParams} />{' '}
        <PlanRootLinkUnsavedWarningAlert
          // If it is a browser nav, the browser alert is good enough.
          show={navGuardState.isTriggered && !navGuardState.isBrowserNav}
          onCancel={resetNavGuardState}
          onLeave={() => {
            assert(navGuardState.isTriggered && !navGuardState.isBrowserNav)
            setForceNav(true)
            resetNavGuardState()
            window.setTimeout(() => urlUpdater.push(navGuardState.url), 0)
          }}
        />
      </>
    )
  },
)
