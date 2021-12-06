import React, {useState} from 'react'
import Measure from 'react-measure'
import {lineFromPoints} from '../../../../Utils/SVG/LineFromPoints'
import {useSVGSizing} from '../../../../Utils/SVG/UseSVGSizing'
import {assertFalse, fGet} from '../../../../Utils/Utils'
import {SliderAxis} from './SliderAxis'
import {SliderPointerEdge} from './SliderPointerEdge'
import {SliderPointerRound} from './SliderPointerRound'
import {SliderTicks} from './SliderTicks'
import {usePointerXAxisDrag} from './UsePointerXAxisDrag'

export const SliderInput = React.memo(
  ({
    pointers,
    onChange,
    className = '',
    domain,
    formatValue: formatValueIn,
    height,
  }: {
    pointers: {value: number; type: 'left' | 'normal' | 'right'}[]
    onChange: (values: number[]) => void
    className?: string
    domain: {
      value: number
      tick: 'none' | 'small' | 'large'
      special?: string
    }[]
    formatValue: (x: number) => string
    height: number
  }) => {
    const indexToValue = (index: number) => domain[Math.round(index)].value
    const valueToIndex = (value: number) => {
      const index = domain.findIndex(x => x.value === value)
      if (index === -1) assertFalse()
      return index
    }

    return (
      <_SliderInput
        pointers={pointers.map(({value, type}) => ({
          value: valueToIndex(value),
          type,
        }))}
        onChange={indexes => onChange(indexes.map(indexToValue))}
        formatValue={index => formatValueIn(indexToValue(index))}
        domain={domain}
        {...{className, height}}
      />
    )
  }
)

const _SliderInput = React.memo(
  ({
    pointers,
    onChange,
    className,
    formatValue,
    domain,
    height,
  }: {
    pointers: {value: number; type: 'left' | 'normal' | 'right'}[]
    onChange: (values: number[]) => void
    className: string
    domain: {
      tick: 'none' | 'small' | 'large'
      special?: string
    }[]
    formatValue: (x: number) => string
    height: number
  }) => {
    const [width, setWidth] = useState(0)

    const sizing = useSVGSizing({
      width,
      height,
      padding: {
        left: height / 2.75,
        right: height / 2.75,
        top: 0,
        bottom: height / 2,
      },
      xDomain: {start: 0, end: domain.length - 1},
      yDomain: {start: 0, end: height / 2},
    })
    const {scale, viewBoxStr, viewBox, target} = sizing

    const pointerState = usePointerXAxisDrag({
      values: pointers.map(x => x.value),
      sizing,
      onDone: onChange,
    })

    const activeRange = {
      start: pointers[0].type === 'left' ? pointerState[0].displayValue : 0,
      end:
        pointers[pointers.length - 1].type === 'right'
          ? pointerState[pointers.length - 1].displayValue
          : domain.length - 1,
    }

    return (
      <Measure bounds onResize={({bounds}) => setWidth(fGet(bounds?.width))}>
        {({measureRef}) => (
          <div
            className={`${className} relative flex flex-col justify-center items-center select-none`}
            ref={measureRef}
            // This is needed to make setPointerCapture() work in mobile.
            style={{touchAction: 'none', height: `${height}px`}}
          >
            <svg
              className="outline-none absolute"
              width={width}
              height={height}
              viewBox={viewBoxStr}
              {...(pointerState.length === 1 ? pointerState[0].props : {})}
            >
              {/* This was needed to make localPoint() work correctly. */}
              <rect
                {...viewBox}
                className="fill-current text-gray-200 opacity-0"
              />
              {/* <rect {...target} className="fill-current text-gray-300" /> */}

              {pointerState.map((control, i) => {
                const pointer = pointers[i]
                const props = {
                  key: i,
                  control,
                  dragProps: pointerState.length === 1 ? null : control.props,
                  sizing,
                  formatValue,
                }
                return pointer.type === 'normal' ? (
                  <SliderPointerRound {...props} />
                ) : (
                  <SliderPointerEdge {...props} type={pointer.type} />
                )
              })}

              <path
                className="stroke-current "
                strokeWidth={1}
                d={lineFromPoints([
                  [0, 0],
                  [scale.x(domain.length - 1), 0],
                ])}
              />
              <SliderAxis {...{sizing, activeRange}} />

              {(['small', 'large'] as const).map(type => (
                <SliderTicks
                  key={`${type}Slider`}
                  values={domain
                    .map((x, i) => (x.tick === type ? i : null))
                    .filter((i): i is number => i !== null)}
                  type={type}
                  inDrag={pointerState.some(x => x.inDrag)}
                  scale={scale.x}
                  activeRange={activeRange}
                />
              ))}

              {domain.map((x, i) =>
                x.special ? (
                  <path
                    key={i}
                    stroke={x.special}
                    strokeWidth={2}
                    d={lineFromPoints([
                      [scale.x(i), 0],
                      [scale.x(i), -10],
                    ])}
                  />
                ) : (
                  <React.Fragment key={i} />
                )
              )}
            </svg>
          </div>
        )}
      </Measure>
    )
  }
)
