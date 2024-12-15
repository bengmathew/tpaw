import { assertFalse, noCase } from '@tpaw/common'
import React, { useMemo } from 'react'
import { NoDisplayOnOpacity0Transition } from '../../../../Utils/NoDisplayOnOpacity0Transition'
import { useSimulationInfo } from '../../PlanRootHelpers/WithSimulation'
import {
  PlanTransitionState,
  simplifyPlanTransitionState5,
} from '../PlanTransition'
import { PlanMenuLinkSrc } from './PlanMenuLinkSrc'
import { PlanMenuLocalSrc } from './PlanMenuLocalPlanSrc'
import { PlanMenuServerSrcHistoryMode } from './PlanMenuServerSrcHistoryMode'
import { PlanMenuServerSrcPlanMode } from './PlanMenuServerSrcPlanMode'
import { PlanMenuFileSrc } from './PlanMenuFileSrc'

export type PlanMenuSizing = {
  dynamic: Record<
    _PlanUndoTransitionState,
    { position: { right: number; top: number }; opacity: number }
  >
}

const _toPlanMenuTransitionState = simplifyPlanTransitionState5(
  {
    label: 'help',
    sections: [{ section: 'help', dialogMode: 'any' }],
  },
  {
    label: 'summaryDialog',
    sections: [{ section: 'summary', dialogMode: true }],
  },
  {
    label: 'inputDialog',
    sections: [{ section: 'rest', dialogMode: true }],
  },
  {
    label: 'summaryNotDialog',
    sections: [{ section: 'summary', dialogMode: false }],
  },
  {
    label: 'inputNotDialog',
    sections: [{ section: 'rest', dialogMode: false }],
  },
)
type _PlanUndoTransitionState = ReturnType<typeof _toPlanMenuTransitionState>
// Note: The plan menu, execpt for the undo/redo menu and rewind menu uses
// params from simulationResult similar to PlanResults and not directly from
// simulationInfo. This is because actions rely on results (e.g. portfolio
// balance, etc) which are available only in simulationResult and it is
// complicated and  sematically hard to reason about correctness if we use some
// data from simulationInfo.
export const PlanMenu = React.memo(
  ({
    sizing,
    planTransition,
  }: {
    sizing: PlanMenuSizing
    planTransition: { target: PlanTransitionState; duration: number }
  }) => {
    const targetSizing = useMemo(
      () => sizing.dynamic[_toPlanMenuTransitionState(planTransition.target)],
      [planTransition.target, sizing],
    )

    const { simulationInfoByMode, simulationInfoBySrc } = useSimulationInfo()

    return (
      <NoDisplayOnOpacity0Transition
        className="absolute overflow-hidden mt-2  z-20"
        style={{
          transitionProperty: 'transform, opacity',
          transitionDuration: `${planTransition.duration}ms`,
          transform: `translate(calc(${targetSizing.position.right}px - 100%),${targetSizing.position.top}px)`,
          opacity: `${targetSizing.opacity}`,
          left: '0px',
          top: '0px',
        }}
      >
        {simulationInfoBySrc.src === 'localMain' ? (
          simulationInfoByMode.mode === 'plan' ? (
            <PlanMenuLocalSrc
              simulationInfoForLocalMainSrc={simulationInfoBySrc}
              simulationInfoForPlanMode={simulationInfoByMode}
            />
          ) : simulationInfoByMode.mode === 'history' ? (
            assertFalse()
          ) : (
            noCase(simulationInfoByMode)
          )
        ) : simulationInfoBySrc.src === 'server' ? (
          simulationInfoByMode.mode === 'plan' ? (
            <PlanMenuServerSrcPlanMode
              simulationInfoForServerSrc={simulationInfoBySrc}
              simulationInfoForPlanMode={simulationInfoByMode}
            />
          ) : simulationInfoByMode.mode === 'history' ? (
            <PlanMenuServerSrcHistoryMode
              simulationInfoForServerSrc={simulationInfoBySrc}
              simulationInfoForHistoryMode={simulationInfoByMode}
            />
          ) : (
            noCase(simulationInfoByMode)
          )
        ) : simulationInfoBySrc.src === 'link' ? (
          simulationInfoByMode.mode === 'plan' ? (
            <PlanMenuLinkSrc
              simulationInfoForLinkSrc={simulationInfoBySrc}
              simulationInfoForPlanMode={simulationInfoByMode}
            />
          ) : simulationInfoByMode.mode === 'history' ? (
            assertFalse()
          ) : (
            noCase(simulationInfoByMode)
          )
        ) : simulationInfoBySrc.src === 'file' ? (
          simulationInfoByMode.mode === 'plan' ? (
            <PlanMenuFileSrc
              simulationInfoForFileSrc={simulationInfoBySrc}
              simulationInfoForPlanMode={simulationInfoByMode}
            />
          ) : simulationInfoByMode.mode === 'history' ? (
            assertFalse()
          ) : (
            noCase(simulationInfoByMode)
          )
        ) : (
          noCase(simulationInfoBySrc)
        )}
      </NoDisplayOnOpacity0Transition>
    )
  },
)
