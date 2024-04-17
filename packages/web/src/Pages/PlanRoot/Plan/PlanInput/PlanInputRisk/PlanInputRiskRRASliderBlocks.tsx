import { PLAN_PARAMS_CONSTANTS } from '@tpaw/common'
import React from 'react'
import { SliderPointerState } from '../../../../Common/Inputs/SliderInput/UseSliderPointerState'
import { SliderSizing } from '../../../../Common/Inputs/SliderInput/UseSliderSizing'

export const PlanInputRiskRRASliderBlocks = React.memo(
  ({
    index,
    sizing,
    pointerState,
  }: {
    index: number
    sizing: SliderSizing
    pointerState: SliderPointerState
  }) => {
    const { pixelXToDataX } = sizing
    const pointerIndex = pointerState.drag?.index ?? index
    return (
      <>
        {PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.segments.map(
          (segment, i) => (
            <rect
              key={i}
              className={`fill-current 
            ${
              segment.containsIndex(pointerIndex)
                ? 'text-gray-600'
                : 'text-gray-300'
            }
            `}
              x={pixelXToDataX.inverse(segment.startIndex)}
              y={-6}
              width={pixelXToDataX.inverse(segment.count - 1)}
              height={6}
            />
          ),
        )}
      </>
    )
  },
)
