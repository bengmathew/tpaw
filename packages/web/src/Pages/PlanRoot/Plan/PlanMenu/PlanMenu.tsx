import { assertFalse, noCase } from '@tpaw/common'
import React, { useMemo } from 'react'
import { NoDisplayOnOpacity0Transition } from '../../../../Utils/NoDisplayOnOpacity0Transition'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import {
    PlanTransitionState,
    simplifyPlanTransitionState5,
} from '../PlanTransition'
import { PlanMenuLinkPlanMode } from './PlanMenuLinkPlanMode'
import { PlanMenuLocalPlanMode } from './PlanMenuLocalPlanMode'
import { PlanMenuServerHistoryMode } from './PlanMenuServerHistoryMode'
import { PlanMenuServerPlanMode } from './PlanMenuServerPlanMode'

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

    const { simulationInfoByMode, simulationInfoBySrc } = useSimulation()

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
            <PlanMenuLocalPlanMode
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
            <PlanMenuServerPlanMode
              simulationInfoForServerSrc={simulationInfoBySrc}
              simulationInfoForPlanMode={simulationInfoByMode}
            />
          ) : simulationInfoByMode.mode === 'history' ? (
            <PlanMenuServerHistoryMode
              simulationInfoForServerSrc={simulationInfoBySrc}
              simulationInfoForHistoryMode={simulationInfoByMode}
            />
          ) : (
            noCase(simulationInfoByMode)
          )
        ) : simulationInfoBySrc.src === 'link' ? (
          simulationInfoByMode.mode === 'plan' ? (
            <PlanMenuLinkPlanMode
              simulationInfoForLinkSrc={simulationInfoBySrc}
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
