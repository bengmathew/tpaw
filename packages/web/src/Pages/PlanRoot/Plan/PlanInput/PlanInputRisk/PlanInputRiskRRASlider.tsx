import { fGet, PLAN_PARAMS_CONSTANTS } from '@tpaw/common'
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { PaddingHorz } from '../../../../../Utils/Geometry'
import { lineFromPoints } from '../../../../../Utils/SVG/LineFromPoints'
import { SliderPointer } from '../../../../Common/Inputs/SliderInput/SliderPointer'
import { SliderTicks } from '../../../../Common/Inputs/SliderInput/SliderTicks'
import { useSliderPointerState } from '../../../../Common/Inputs/SliderInput/UseSliderPointerState'
import { useSliderSizing } from '../../../../Common/Inputs/SliderInput/UseSliderSizing'
import { PlanInputRiskRRASliderBlocks } from './PlanInputRiskRRASliderBlocks'

export const PlanInputRiskRRASlider = React.memo(
  ({
    data,
    value,
    onChange: onChangeValue,
    className = '',
    ticks: ticksFn,
    format: formatValue,
    height,
    maxOverflowHorz,
  }: {
    data: number[]
    value: number
    onChange: (value: number) => void
    className?: string
    ticks: (value: number, index: number) => 'none' | 'small' | 'large'
    format: (value: number) => string
    height: number
    maxOverflowHorz: PaddingHorz
  }) => {
    const [width, setWidth] = useState(10)
    const sizing = useSliderSizing(width, height, maxOverflowHorz, data.length)
    const { viewBox, plotSize } = sizing
    const index = useMemo(() => data.indexOf(value), [data, value])
    const formatIndex = (index: number) => formatValue(data[index])
    const onChangeIndex = (index: number) => onChangeValue(data[index])
    const { pointerState, propsForPointerHTMLElement } = useSliderPointerState(
      index,
      sizing,
      onChangeIndex,
    )
    const ticks = data.map(ticksFn)

    const ref = useRef<HTMLDivElement>(null)
    useEffect(() => {
      // fGet() inside observer was crashing so get element ahead of time.
      // Possibly because of this sequence on cleanup. 1. ref set to null 2.
      // resize event 3. disconnect called.
      const element = fGet(ref.current)
      const observer = new ResizeObserver(() => {
        setWidth(element.getBoundingClientRect().width)
      })
      observer.observe(element)
      return () => observer.disconnect()
    }, [])

    const pointerIndex = pointerState.drag?.index ?? index
    const segment = useMemo(
      () =>
        fGet(
          PLAN_PARAMS_CONSTANTS.riskToleranceValues.SEGMENTS.find((x) =>
            x.containsIndex(pointerIndex),
          ),
        ),
      [pointerIndex],
    )

    return (
      <div className={`${className}`} ref={ref}>
        <div
          className="relative select-none"
          style={{
            // This is needed to make setPointerCapture() work in mobile.
            touchAction: 'none',
            height: `${height}px`,
          }}
        >
          <svg
            className="outline-none absolute top-0"
            style={{ left: `-${maxOverflowHorz.left}px` }}
            {...sizing.svgProps}
            {...propsForPointerHTMLElement}
          >
            {/* This was needed to make localPoint() work correctly. */}
            <rect {...viewBox} className=" fill-current opacity-0" />

            <PlanInputRiskRRASliderBlocks
              sizing={sizing}
              index={index}
              pointerState={pointerState}
            />

            {/* The horizontal line. */}
            <path
              className="stroke-current "
              strokeWidth={1}
              d={lineFromPoints([
                [0, 0],
                [plotSize.width, 0],
              ])}
            />

            <SliderTicks
              ticks={ticks}
              inDrag={pointerState.drag !== null}
              sizing={sizing}
            />
            <SliderPointer
              index={index}
              pointerState={pointerState}
              sizing={sizing}
              format={formatIndex}
            />
          </svg>
        </div>
        <p className="mt-2 ml-[15px] ">
          This is {segment.label.startsWith('A') ? 'an' : 'a'}{' '}
          <span className="font-bold bg-gray-300 px-2 py-0.5 rounded-md lowercase">
            {segment.label}
          </span>{' '}
          risk level.
        </p>
      </div>
    )
  },
)
