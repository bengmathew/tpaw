import React from 'react'
import { SliderSizing } from './UseSliderSizing'

export const SliderTicks = React.memo(
  ({
    ticks,
    sizing: { pixelXToDataX },
    inDrag,
  }: {
    inDrag: boolean
    ticks: ('small' | 'large' | 'none')[]
    sizing: SliderSizing
  }) => {
    return (
      <g>
        {ticks.map((type, index) => {
          const x = pixelXToDataX.inverse(index)

          if (type === 'none')
            return <React.Fragment key={index}></React.Fragment>

          const size = type === 'small' ? 6 : 12
          return (
            <rect
              key={index}
              className={`fill-current 
              ${type === 'small' ? 'text-gray-500' : 'text-gray-700'}`}
              x={x - 0.5}
              y={0}
              width={1}
              height={1}
              style={{
                transition: 'transform 300ms',
                transform: `scaleY(${inDrag ? -size : 0})`,
              }}
            />
          )
        })}
      </g>
    )
  },
)
