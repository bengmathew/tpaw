import React from 'react'
import { animated, useSpring } from 'react-spring'
import { lineFromPoints } from '../../../../Utils/SVG/LineFromPoints'

export const SliderTicks = React.memo(
  ({
    values,
    type,
    scale,
    inDrag,
    activeRange,
  }: {
    inDrag: boolean
    values: number[]
    type: 'small' | 'large'
    scale: (x: number) => number
    activeRange: {start: number; end: number}
  }) => {
    const size = type === 'small' ? 6 : 12

    const spring = useSpring(inDrag ? {size} : {size:0})

    return (
      <g>
        {values.map((value, i) => {
          const x = scale(value)
          return (
            <animated.path
              key={i}
              className={`stroke-current 
              ${type === 'small' ? 'text-gray-500' : 'text-gray-600'}`}
              strokeWidth={type === 'small' ? 1 : 1}
              opacity={
                value < activeRange.start || value > activeRange.end ? 0.3 : 1
              }
              d={spring.size.to(size =>
                lineFromPoints([
                  [x, 0],
                  [x, -size],
                ])
              )}
            />
          )
        })}
      </g>
    )
  }
)
