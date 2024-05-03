import {
  PlanParams,
  assert,
  block,
  fGet,
  letIn,
  planParamsMigrate,
} from '@tpaw/common'
import _ from 'lodash'
import React, { useEffect, useState } from 'react'
import * as uuid from 'uuid'
import { appPaths } from '../../../AppPaths'
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
import { PlanFileData } from './PlanFileData'
import { CenteredModal } from '../../Common/Modal/CenteredModal'
import { useNavGuard } from '../../../Utils/UseNavGuard'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'

export const PlanRootFileImpl = React.memo(
  ({
    pdfReportInfo,
    src,
    setSrc,
    reset,
  }: {
    pdfReportInfo: SimulationParams['pdfReportInfo']
    src: {
      filename: string | null
      originalData: PlanFileData
      resetData: PlanFileData | null
    }
    setSrc: (filename: string | null, data: PlanFileData) => void
    reset: (planParams: PlanParams | null) => void
  }) => {
    const urlUpdater = useURLUpdater()
    const [filename] = useState(src.filename)
    const data = src.resetData ?? src.originalData
    const [workingPlanSrc] = useState<WorkingPlanSrc>(() => ({
      planId: uuid.v4(),
      planParamsPostBase: data.planParamsHistory.map((x) => ({
        ...x,
        paramsUnmigrated: x.params,
        params: planParamsMigrate(x.params),
      })),
      reverseHeadIndex: data.reverseHeadIndex,
    }))

    //TODO: Move this to common location
    const planMigratedFromVersion = letIn(
      // Intentionally last and not the one pointer to by reverseHeadIndex.
      fGet(_.last(data.planParamsHistory)).params,
      (lastParamsUnmigrated) =>
        'v' in lastParamsUnmigrated ? lastParamsUnmigrated.v : (1 as const),
    )

    const planPaths = appPaths.file
    const currentTimeInfo = useCurrentTime({
      planId: workingPlanSrc.planId,
    })
    assert(
      currentTimeInfo.currentTimestamp >=
        fGet(_.last(workingPlanSrc.planParamsPostBase)).params.timestamp,
    )
    const workingPlanInfo = useWorkingPlan(
      currentTimeInfo,
      workingPlanSrc,
      planPaths,
    )

    const isModified = block(() => {
      const currPlanParams = letIn(
        fGet(_.last(workingPlanInfo.planParamsUndoRedoStack.undos)),
        (x) => x.paramsUnmigrated ?? x.params,
      )
      const srcPlanParams =
        src.originalData.planParamsHistory[
          src.originalData.planParamsHistory.length -
            1 -
            src.originalData.reverseHeadIndex
        ].params

      return currPlanParams !== srcPlanParams
    })

    const rebase = workingPlanInfo.rebase
    useEffect(() => {
      rebase?.({ hard: true })
    }, [rebase])

    const [forceNav, setForceNav] = useState(false)
    const { navGuardState, resetNavGuardState } = useNavGuard(
      isModified && !forceNav,
      planPaths,
    )

    const simulationParams = useSimulationParamsForPlanMode(
      planPaths,
      currentTimeInfo,
      workingPlanInfo,
      planMigratedFromVersion,
      null,
      {
        src: 'file',
        setSrc: (filename: string | null, src: PlanFileData) =>
          setSrc(filename, src),
        reset,
        setForceNav: () => setForceNav(true),
        plan: {
          filename,
          convertedToFilePlanAtTimestamp: data.convertedToFilePlanAtTimestamp,
        },
        isModified,
      },
      pdfReportInfo,
    )

    return (
      <>
        <WithSimulation params={simulationParams} />
        <CenteredModal
          className="dialog-outer-div"
          // If it is a browser nav, the browser alert is good enough.
          show={navGuardState.isTriggered && !navGuardState.isBrowserNav}
          onOutsideClickOrEscape={null}
        >
          <h2 className=" dialog-heading">Unsaved Changes</h2>
          <div className=" dialog-content-div relative ">
            <p className=" p-base">
              You have made changes to this plan since you opened it. Leaving
              this page will discard these changes.
            </p>
          </div>
          <div className=" dialog-button-div">
            <button
              className=" dialog-button-cancel"
              onClick={resetNavGuardState}
            >
              Cancel
            </button>
            <button
              className=" dialog-button-warning"
              onClick={() => {
                assert(navGuardState.isTriggered && !navGuardState.isBrowserNav)
                setForceNav(true)
                resetNavGuardState()
                window.setTimeout(() => urlUpdater.push(navGuardState.url), 0)
              }}
            >
              Leave Page
            </button>
          </div>
        </CenteredModal>
      </>
    )
  },
)
