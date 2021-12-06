import { Text } from '@visx/text'
import React from 'react'
import { animated, config, useSpring } from 'react-spring'
import { SVGSizing } from '../../../../Utils/SVG/UseSVGSizing'
import { PointerXAxisDragResult } from './UsePointerXAxisDrag'
const AnimatedText = animated(Text)

export const SliderPointerEdge = React.memo(
  ({
    control,
    dragProps,
    formatValue,
    sizing,
    type,
  }: {
    sizing: SVGSizing
    control: Omit<PointerXAxisDragResult, 'props'>
    dragProps: PointerXAxisDragResult['props'] | null
    formatValue: (x: number) => string
    type: 'left' | 'right'
  }) => {
    const {inDrag, focus, displayValue} = control
    const x = sizing.scale.x(displayValue)
    const height = sizing.viewBox.height / 2

    const labelY = -(height * 0.3)
    const fontSize = height * 0.45

    const pointerY = 1
    const dragPointerDY = 1.5
    const pointerHeight = height - 8
    const pointerAspectRatio = 0.5
    const pointerWidth = pointerAspectRatio * pointerHeight
    const pointerRadius = 5

    const focusW = pointerWidth * 2.5
    const focusH = height - dragPointerDY
    const focusX = type === 'left' ? -(focusW - focusW / 3) : -focusW / 3

    const spring = useSpring({
      to: inDrag
        ? {
            pointerDy: dragPointerDY,
            focusOpacity: 0.5,
            labelDy: -height * 0.3,
          }
        : focus
        ? {
            pointerDy: 0,
            focusOpacity: 0.5,
            labelDy: 0,
          }
        : {
            pointerDy: 0,
            focusOpacity: 0,
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
          textAnchor={type === 'right' ? 'start' : 'end'}
          verticalAnchor="end"
        >
          {formatValue(displayValue)}
        </AnimatedText>
        <animated.rect
          className="fill-current text-gray-300"
          opacity={spring.focusOpacity}
          x={focusX}
          y={0}
          width={focusW}
          height={spring.pointerDy.to(dy => dy + focusH)}
          rx={5}
        />
        <g>
          <animated.path
            className="fill-current stroke-current text-darkGray"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            d={spring.pointerDy.to(
              dy => `M 0,${pointerY + dy + 2} 
                v ${pointerHeight - pointerRadius} 
                a ${pointerRadius} ${pointerRadius} 0  0 ${
                type === 'right' ? '0' : '1'
              } ${pointerRadius * (type === 'right' ? 1 : -1)}, ${pointerRadius}
                h ${
                  (pointerWidth - pointerRadius) * (type === 'right' ? 1 : -1)
                }
                Z
                `
            )}
          />
        </g>
        <rect
          className="outline-none fill-current text-gray-300 opacity-0"
          x={focusX}
          y={sizing.viewBox.y}
          height={sizing.viewBox.height}
          width={focusW}
          {...(dragProps ?? {})}
        />
      </g>
    )
  }
)
