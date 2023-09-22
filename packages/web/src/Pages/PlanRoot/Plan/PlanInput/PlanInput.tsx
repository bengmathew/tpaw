import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Padding,
  Size,
  XY,
  originCSSStyle,
  sizeCSSStyle,
} from '../../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../../Utils/NoDisplayOnOpacity0Transition'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { noCase } from '../../../../Utils/Utils'
import { useGetSectionURL } from '../Plan'
import {
  PlanTransitionState,
  simplifyPlanTransitionState4,
} from '../PlanTransition'
import { PlanInputType } from './Helpers/PlanInputType'
import { useGetPlanInputVisibility } from './Helpers/UseGetPlanInputVisibility'
import { PlanInputAge } from './PlanInputAge/PlanInputAge'
import { PlanInputBodyPassThruProps } from './PlanInputBody/PlanInputBody'
import { PlanInputCurrentPortfolioBalance } from './PlanInputCurrentPortfolioBalance'
import { PlanInputDevHistoricalReturns } from './PlanInputDev/PlanInputDevHistoricalReturns'
import { PlanInputDevMisc } from './PlanInputDev/PlanInputDevMisc'
import { PlanInputDevSimulations } from './PlanInputDev/PlanInputDevSimulations'
import { PlanInputDevFastForward } from './PlanInputDev/PlanInputDevTime'
import { PlanInputExpectedReturns } from './PlanInputExpectedReturns'
import { PlanInputExtraSpending } from './PlanInputExtraSpending'
import { PlanInputFutureSavings } from './PlanInputFutureSavings'
import { PlanInputIncomeDuringRetirement } from './PlanInputIncomeDuringRetirement'
import { PlanInputInflation } from './PlanInputInflation'
import { PlanInputLegacy } from './PlanInputLegacy'
import { PlanInputRisk } from './PlanInputRisk/PlanInputRisk'
import { PlanInputSimulation } from './PlanInputSimulation'
import { PlanInputSpendingCeilingAndFloor } from './PlanInputSpendingCeilingAndFloor'
import { PlanInputStrategy } from './PlanInputStrategy'

type _FixedSizingByMode = {
  size: Size
  padding: ({ left: number; right: number } | { horz: number }) & {
    top: number
  }
}
export type PlanInputSizing = {
  dynamic: Record<PlanInputTransitionState, { origin: XY; opacity: number }>
  fixed: {
    dialogMode: _FixedSizingByMode
    notDialogMode: _FixedSizingByMode
    cardPadding: Padding
  }
}

export const toPlanInputTransitionStateByType = (type: PlanInputType) =>
  simplifyPlanTransitionState4(
    { label: 'dialogIn', sections: [{ name: type, dialogMode: true }] },
    {
      label: 'dialogOut',
      sections: [{ name: 'rest', dialogMode: true }],
    },
    { label: 'notDialogIn', sections: [{ name: type, dialogMode: false }] },
    {
      label: 'notDialogOut',
      sections: [{ name: 'rest', dialogMode: false }],
    },
  )

export type PlanInputTransitionState = ReturnType<
  ReturnType<typeof toPlanInputTransitionStateByType>
>

export const PlanInput = React.memo(
  ({
    layout,
    sizing,
    planTransition,
    planInputType,
  }: {
    layout: 'laptop' | 'desktop' | 'mobile'
    planInputType: PlanInputType
    sizing: PlanInputSizing
    planTransition: { target: PlanTransitionState; duration: number }
  }) => {
    // This is a constant, so don't calculate this each time.
    const [toPlanInputTransitionState] = useState(() =>
      toPlanInputTransitionStateByType(planInputType),
    )

    const targetSizing = useMemo(
      () => sizing.dynamic[toPlanInputTransitionState(planTransition.target)],
      [planTransition.target, sizing, toPlanInputTransitionState],
    )
    const fixedSizing = planTransition.target.dialogMode
      ? sizing.fixed.dialogMode
      : sizing.fixed.notDialogMode

    return (
      <NoDisplayOnOpacity0Transition
        className={`absolute`}
        style={{
          transitionProperty: 'transform, opacity',
          transitionDuration: `${planTransition.duration}ms`,
          transform: `translate(${targetSizing.origin.x}px,${targetSizing.origin.y}px)`,
          opacity: `${targetSizing.opacity}`,
          ...sizeCSSStyle(fixedSizing.size),
          ...originCSSStyle({ x: 0, y: 0 }),
        }}
        data-type={planInputType}
      >
        <_Body
          planInputType={planInputType}
          props={{
            layout,
            sizing: sizing.fixed,
            type: planInputType,
          }}
        />
      </NoDisplayOnOpacity0Transition>
    )
  },
  // )
)

const _Body = React.memo(
  ({
    planInputType,
    props,
  }: {
    planInputType: PlanInputType
    props: PlanInputBodyPassThruProps
  }) => {
    const visibility = useGetPlanInputVisibility()(planInputType)
    const show = visibility.visible && !visibility.disabled
    const summarySectionURL = useGetSectionURL()('summary')
    const urlUpdater = useURLUpdater()
    const handleHide = () => urlUpdater.replace(summarySectionURL)
    const handleHideRef = useRef(handleHide)
    handleHideRef.current = handleHide
    useEffect(() => {
      if (!show) handleHideRef.current()
    }, [show])

    if (!show) return <></>
    switch (planInputType) {
      case 'age':
        return <PlanInputAge {...props} />
      case 'current-portfolio-balance':
        return <PlanInputCurrentPortfolioBalance {...props} />
      case 'future-savings':
        return <PlanInputFutureSavings {...props} />
      case 'income-during-retirement':
        return <PlanInputIncomeDuringRetirement {...props} />
      case 'extra-spending':
        return <PlanInputExtraSpending {...props} />
      case 'legacy':
        return <PlanInputLegacy {...props} />
      case 'spending-ceiling-and-floor':
        return <PlanInputSpendingCeilingAndFloor {...props} />
      case 'risk':
        return <PlanInputRisk {...props} />
      case 'strategy':
        return <PlanInputStrategy {...props} />
      case 'expected-returns':
        return <PlanInputExpectedReturns {...props} />
      case 'inflation':
        return <PlanInputInflation {...props} />
      case 'simulation':
        return <PlanInputSimulation {...props} />
      case 'dev-misc':
        return <PlanInputDevMisc {...props} />
      case 'dev-simulations':
        return <PlanInputDevSimulations {...props} />
      case 'dev-historical-returns':
        return <PlanInputDevHistoricalReturns {...props} />
      case 'dev-time':
        return <PlanInputDevFastForward {...props} />
      default:
        noCase(planInputType)
    }
  },
)
