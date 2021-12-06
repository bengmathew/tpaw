import { Text } from '@visx/text'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { animated, config, useSpring } from 'react-spring'
import { lineFromPoints } from '../../../../Utils/SVG/LineFromPoints'
import { SVGSizing } from '../../../../Utils/SVG/UseSVGSizing'
import { PointerXAxisDragResult } from './UsePointerXAxisDrag'

const AnimatedText = animated(Text)

export const SliderPointerRound = React.memo(
  ({
    control,
    dragProps,
    formatValue,
    sizing,
  }: {
    sizing: SVGSizing
    control: Omit<PointerXAxisDragResult, 'props'>
    dragProps: PointerXAxisDragResult['props'] | null
    formatValue: (x: number) => string
  }) => {
    const {inDrag, focus, displayValue} = control
    const height = sizing.target.height

    const x = sizing.scale.x(displayValue)
    const id = useMemo(() => _.uniqueId(), [])

    const labelY = -(height * 0.3)
    const fontSize = height * 0.45
    const pointerY = 1
    const dragPointerDY = 1.5
    const pointerHeight = height - 6
    const pointerAspectRatio = 24 / 30
    const pointerWidth = pointerAspectRatio * pointerHeight
    const scale = pointerHeight / 30
    const focusW = pointerWidth + 4
    const focusH = height - dragPointerDY

    const spring = useSpring({
      to: inDrag
        ? {
            pointerDy: dragPointerDY,
            holeRadius: 8,
            focusOpacity: 1,
            line: [5.5, 13],
            labelDy: -height * 0.3,
          }
        : focus
        ? {
            pointerDy: 0,
            holeRadius: 5,
            focusOpacity: 1,
            line: [4, 0],
            labelDy: 0,
          }
        : {
            pointerDy: 0,
            holeRadius: 5,
            focusOpacity: 0,
            line: [4, 0],
            labelDy: 0,
          },
      config: config.stiff,
    })

    return (
      <g transform={`translate(${x}, 0)`}>
        <AnimatedText
          className="bg-gray-200 "
          x={0}
          y={0}
          fontSize={fontSize}
          dy={spring.labelDy.to(dy => labelY + dy)}
          textAnchor="middle"
          verticalAnchor="end"
        >
          {formatValue(displayValue)}
        </AnimatedText>
        <animated.rect
          className="fill-current text-gray-300"
          opacity={spring.focusOpacity}
          x={-focusW / 2}
          y={0}
          width={focusW}
          height={spring.pointerDy.to(dy =>  dy + focusH)}
          rx={5}
        />
        <animated.path
          className="stroke-current text-darkGray"
          strokeWidth="2"
          d={spring.line.to((y, h) =>
            lineFromPoints([
              [0, y],
              [0, y - h],
            ])
          )}
        />
        <animated.g
          transform={spring.pointerDy.to(
            pointerDy =>
              ` scale(${scale}, ${scale}) rotate(180, 16, 16) translate(${16}, ${-(
                pointerY + pointerDy
              )})`
          )}
        >
          <defs>
            <mask id={`roundPointerHandleMask-${id}`}>
              <rect x={0} cy={0} width={32} height={32} fill="white" />
              <animated.circle
                cx={16}
                cy={14}
                r={spring.holeRadius}
                fill="black"
              />
            </mask>
          </defs>
          {/* path  */}
          <path
            mask={`url(#roundPointerHandleMask-${id})`}
            className="fill-current text-darkGray pointer-events-none "
            d="M 24 4.051 C 17.172 -2.171 6.169 1.332 4.195 10.356 C 3.239 14.722 4.79 19.257 8.218 22.125 C 11.019 24.454 13.359 27.286 15.118 30.476 C 15.293 30.8 15.632 31.001 16 31 C 16.368 31 16.706 30.798 16.88 30.474 L 16.962 30.32 C 18.733 27.169 21.069 24.37 23.852 22.063 C 29.325 17.333 29.395 8.87 24 4.051 Z"
          />
        </animated.g>
        <rect
          className="outline-none fill-current text-gray-300 opacity-0"
          x={(-focusW) / 2}
          y={sizing.viewBox.y}
          height={sizing.viewBox.height}
          width={focusW}
          {...(dragProps ?? {})}
        />
      </g>
    )
  }
)
