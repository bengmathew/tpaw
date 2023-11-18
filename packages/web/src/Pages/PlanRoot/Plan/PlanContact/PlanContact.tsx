import React, { useMemo } from 'react'
import { NoDisplayOnOpacity0Transition } from '../../../../Utils/NoDisplayOnOpacity0Transition'
import {
  PlanTransitionState,
  simplifyPlanTransitionState5,
} from '../PlanTransition'
import { usePlanColors } from '../UsePlanColors'
import { PlanContactContent } from './PlanContactContent'

export type PlanContactSizing = {
  dynamic: Record<
    _PlanContactTransitionState,
    { position: { right: number; bottom: number }; opacity: number }
  >
}

const _toPlanContactTransitionState = simplifyPlanTransitionState5(
  {
    label: 'help',
    sections: [{ section: 'help', dialogMode: 'any'}],
  },
  {
    label: 'summaryDialog',
    sections: [{ section: 'summary', dialogMode: true}],
  },
  {
    label: 'inputDialog',
    sections: [{ section: 'rest', dialogMode: true}],
  },
  {
    label: 'summaryNotDialog',
    sections: [{ section: 'summary', dialogMode: false}],
  },
  {
    label: 'inputNotDialog',
    sections: [{ section: 'rest', dialogMode: false}],
  },
)
type _PlanContactTransitionState = ReturnType<
  typeof _toPlanContactTransitionState
>

export const PlanContact = React.memo(
  ({
    sizing,
    planTransition,
  }: {
    sizing: PlanContactSizing
    planTransition: { target: PlanTransitionState; duration: number }
  }) => {
    const targetSizing = useMemo(
      () =>
        sizing.dynamic[_toPlanContactTransitionState(planTransition.target)],
      [planTransition.target, sizing],
    )
    const planColors = usePlanColors()

    return (
      <NoDisplayOnOpacity0Transition
        className="absolute overflow-hidden z-20 rounded-full"
        style={{
          transitionProperty: 'transform, opacity',
          transitionDuration: `${planTransition.duration}ms`,
          transform: `translate(calc(${targetSizing.position.right}px - 100%), calc(${targetSizing.position.bottom}px - 100%))`,
          opacity: `${targetSizing.opacity}`,
          left: '0px',
          top: '0px',
        }}
      >
        <PlanContactContent className="" colors={planColors.results} />
      </NoDisplayOnOpacity0Transition>
    )
  },
)
