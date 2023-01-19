import React, { useMemo, useState } from 'react'
import {
  originCSSStyle,
  Padding,
  Size,
  sizeCSSStyle,
  XY,
} from '../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../Utils/NoDisplayOnOpacity0Transition'
import { noCase } from '../../../Utils/Utils'
import {
  PlanTransitionState,
  simplifyPlanTransitionState4,
} from '../PlanTransition'
import { PlanInputType } from './Helpers/PlanInputType'
import { PlanInputAge } from './PlanInputAge/PlanInputAge'
import { PlanInputBodyPassThruProps } from './PlanInputBody/PlanInputBody'
import { PlanInputCompareStrategies } from './PlanInputCompareStrategies'
import { PlanInputCurrentPortfolioBalance } from './PlanInputCurrentPortfolioBalance'
import { PlanInputDev } from './PlanInputDev'
import { PlanInputExpectedReturns } from './PlanInputExpectedReturns'
import { PlanInputExtraSpending } from './PlanInputExtraSpending'
import { PlanInputFutureSavings } from './PlanInputFutureSavings'
import { PlanInputIncomeDuringRetirement } from './PlanInputIncomeDuringRetirement'
import { PlanInputInflation } from './PlanInputInflation'
import { PlanInputLegacy } from './PlanInputLegacy'
import { PlanInputRisk } from './PlanInputRisk/PlanInputRisk'
import { PlanInputSimulation } from './PlanInputSimulation'
import { PlanInputSpendingCeilingAndFloor } from './PlanInputSpendingCeilingAndFloor'

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
    { label: 'dialogModeIn', sections: [{ name: type, dialogMode: true }] },
    {
      label: 'dialogModeOut',
      sections: [{ name: 'rest', dialogMode: true }],
    },
    { label: 'notDialogModeIn', sections: [{ name: type, dialogMode: false }] },
    {
      label: 'notDialogModeOut',
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

    const props: PlanInputBodyPassThruProps = {
      layout,
      sizing: sizing.fixed,
      type: planInputType,
    }
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
        {(() => {
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
              return <PlanInputCompareStrategies {...props} />
            case 'expected-returns':
              return <PlanInputExpectedReturns {...props} />
            case 'inflation':
              return <PlanInputInflation {...props} />
            case 'simulation':
              return <PlanInputSimulation {...props} />
            case 'dev':
              return <PlanInputDev {...props} />
            default:
              noCase(planInputType)
          }
        })()}
      </NoDisplayOnOpacity0Transition>
    )
  },
  // )
)
