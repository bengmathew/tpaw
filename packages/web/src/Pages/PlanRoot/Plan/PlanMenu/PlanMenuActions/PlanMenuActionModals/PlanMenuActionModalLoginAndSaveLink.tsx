import React, { useState } from 'react'
import { appPaths } from '../../../../../../AppPaths'
import { useURLUpdater } from '../../../../../../Utils/UseURLUpdater'
import { MergeToServerLinkPlan } from '../../../../../App/WithMergeToServer'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import {
  SimulationInfoForLinkSrc,
  SimulationInfoForPlanMode,
} from '../../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalLabelInput } from './PlanMenuActionModalLabelInput'

export const PlanMenuActionModalLoginAndSaveLink = React.memo(
  ({
    show,
    onHide,
    simulationInfoForLinkSrc,
    simulationInfoForPlanMode,
  }: {
    show: boolean
    onHide: () => void
    simulationInfoForPlanMode: SimulationInfoForPlanMode
    simulationInfoForLinkSrc: SimulationInfoForLinkSrc
  }) => {
    return (
      <CenteredModal
        className="dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        <_Body
          onHide={onHide}
          simulationInfoForLinkSrc={simulationInfoForLinkSrc}
          simulationInfoForPlanMode={simulationInfoForPlanMode}
        />
      </CenteredModal>
    )
  },
)

// Body seperated out to recreate it at each show to reset state.
const _Body = React.memo(
  ({
    onHide,
    simulationInfoForLinkSrc,
    simulationInfoForPlanMode,
  }: {
    onHide: () => void
    simulationInfoForLinkSrc: SimulationInfoForLinkSrc
    simulationInfoForPlanMode: SimulationInfoForPlanMode
  }) => {
    const { setForceNav } = simulationInfoForLinkSrc
    const { planParamsUndoRedoStack } = simulationInfoForPlanMode

    const urlUpdater = useURLUpdater()

    const handleLabel = (label: string) => {
      MergeToServerLinkPlan.write({
        label,
        plan: {
          planParamsHistory: [
            ...planParamsUndoRedoStack.undos,
            ...planParamsUndoRedoStack.redos,
          ].map((x) => ({
            id: x.id,
            params: JSON.stringify(x.params),
            change: JSON.stringify(x.change),
          })),
          reverseHeadIndex: planParamsUndoRedoStack.redos.length,
        },
      })
      setForceNav()
      // Timeout for setForceNav() to take effect.
      window.setTimeout(
        () => urlUpdater.push(appPaths.login(appPaths.plans())),
        1,
      )
    }

    return (
      <PlanMenuActionModalLabelInput
        title="Save Plan to Account"
        initialLabel=""
        buttonLabel="Save to Account"
        onCancel={onHide}
        onAction={handleLabel}
        isRunning={false}
      />
    )
  },
)
