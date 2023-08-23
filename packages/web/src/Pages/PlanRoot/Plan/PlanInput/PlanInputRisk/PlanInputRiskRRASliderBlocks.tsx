import { RISK_TOLERANCE_VALUES } from '@tpaw/common'
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
        {RISK_TOLERANCE_VALUES.SEGMENTS.map((segment, i) => (
          <rect
            key={i}
            className={`fill-current 
            ${
              segment.containsIndex(pointerIndex)
                ? 'text-gray-400'
                : 'text-gray-200'
            }
            `}
            x={pixelXToDataX.inverse(segment.startIndex)}
            y={-6}
            width={pixelXToDataX.inverse(segment.count - 1)}
            height={6}
          />
        ))}
      </>
    )
  },
)
