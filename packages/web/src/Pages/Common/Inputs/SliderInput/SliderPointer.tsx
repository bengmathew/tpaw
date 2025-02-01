import _ from 'lodash'
import React, { useMemo } from 'react'
import { SliderPointerState } from './UseSliderPointerState'
import { SliderSizing } from './UseSliderSizing'

export const SliderPointer = React.memo(
  ({
    index,
    pointerState: { drag, focus },
    format,
    sizing,
  }: {
    index: number
    sizing: SliderSizing
    pointerState: SliderPointerState
    format: (index: number) => string
  }) => {
    const { plotSize, pixelXToDataX } = sizing
    const sliderIndex = drag?.index ?? index

    const id = useMemo(() => _.uniqueId(), [])

    const fontSize = plotSize.top * 0.5
    const pointerY = 1
    const pointerHeight = plotSize.bottom
    const pointerAspectRatio = 24 / 30
    const pointerWidth = pointerAspectRatio * pointerHeight
    const scale = pointerHeight / 34
    const focusW = pointerWidth + 4
    const focusH = plotSize.bottom - 1

    return (
      <g transform={`translate(${pixelXToDataX.inverse(sliderIndex)}, 0)`}>
        <text
          className="bg-gray-200 "
          x={0}
          y={0}
          fontSize={fontSize}
          textAnchor="middle"
          style={{
            transition: 'transform 300ms',
            transform: `translateY(${-plotSize.top * (drag ? 0.5 : 0.3)}px)`,
          }}
        >
          {format(sliderIndex)}
        </text>
        <rect
          className="fill-current text-gray-300"
          x={-focusW / 2}
          y={1}
          width={focusW}
          height={focusH}
          rx={5}
          style={{
            transition: 'opacity 100ms',
            opacity: focus ? '1' : '0',
          }}
        />
        <rect
          className="fill-current text-gray-800"
          x={-1}
          y={0}
          width={2}
          height={1}
          style={{
            transition: 'transform 300ms',
            transform: `translateY(4px) scaleY(${drag ? -10 : 0})`,
          }}
        />

        <g
          transform={`scale(${scale}, ${scale}) rotate(180, 16, 16) translate(${16}, ${-pointerY})`}
        >
          <defs>
            <mask id={`roundPointerHandleMask-${id}`}>
              <rect x={0} cy={0} width={32} height={32} fill="white" />
              <circle
                cx={0}
                cy={0}
                r={1}
                fill="black"
                style={{
                  transition: 'transform 300ms',
                  transform: `translate(16px, 13.5px) scale(${drag ? 8 : 5}) `,
                }}
              />
            </mask>
          </defs>
          {/* path  */}
          <path
            mask={`url(#roundPointerHandleMask-${id})`}
            className="fill-current text-gray-800 pointer-events-none "
            d="M 24 4.051 C 17.172 -2.171 6.169 1.332 4.195 10.356 C 3.239 14.722 4.79 19.257 8.218 22.125 C 11.019 24.454 13.359 27.286 15.118 30.476 C 15.293 30.8 15.632 31.001 16 31 C 16.368 31 16.706 30.798 16.88 30.474 L 16.962 30.32 C 18.733 27.169 21.069 24.37 23.852 22.063 C 29.325 17.333 29.395 8.87 24 4.051 Z"
          />
        </g>
      </g>
    )
  },
)
