import React, {CSSProperties, useEffect, useRef, useState} from 'react'
import Measure from 'react-measure'
import {assert, fGet} from '../../../Utils/Utils'
import {
  Chart,
  ChartAnimation,
  ChartPadding,
  ChartState,
  ChartXYRange,
} from './Chart'
import {ChartComponent} from './ChartComponent/ChartComponent'

export type ChartReactState<Data> = {
  padding:
    | ChartPadding
    | ((bounds: {width: number; height: number}) => ChartPadding)
  data: Data
  xyRange: ChartXYRange
  animation: ChartAnimation | null
}

export function ChartReact<Data>({
  state,
  components,
  className = '',
  style,
}: {
  state: ChartReactState<Data>
  animationForBoundsChange: ChartAnimation | null
  components: readonly ChartComponent<Data>[]
  className?: string
  style?: CSSProperties
}) {
  const chartRef = useRef<Chart<Data> | null>(null)
  const [bounds, setBounds] = useState<{width: number; height: number} | null>(
    null
  )

  useEffect(() => {

    if (bounds) {
      const chart = fGet(chartRef.current)
      chart.setState(_processState(state, bounds), state.animation)
    }
    // separate response for state and bounds.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state])

  useEffect(() => {
    if (bounds) {
      fGet(chartRef.current).setState(_processState(state, bounds), null)
    }

    // separate response for state and bounds.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds])

  useEffect(() => {
    chartRef.current?.setComponents(components)
  }, [components])

  useEffect(() => {
    return () => chartRef.current?.destroy()
  }, [])

  return (
    <div className={`${className}`} style={style}>
      <Measure
        bounds
        onResize={({bounds}) => {
          assert(bounds)
          setBounds({width: bounds.width, height: bounds.height})
        }}
      >
        {({measureRef}) => (
          <div className="w-full h-full" ref={measureRef}>
            {/* select-none because otherwise moving the chart pointer was
          selecting text  on mobile. Seems to work only on Android though. */}
            <div className=" select-none  absolute">
              {bounds && (
                <canvas
                  style={{touchAction: 'none'}}
                  ref={canvas => {
                    if (canvas && !chartRef.current) {
                      chartRef.current = new Chart(
                        canvas,
                        _processState(state, bounds),
                        components
                      )
                    }
                  }}
                />
              )}
            </div>
          </div>
        )}
      </Measure>
    </div>
  )
}

function _processState<Data>(
  {padding, data, xyRange}: ChartReactState<Data>,
  bounds: {width: number; height: number}
): ChartState<Data> {
  return {
    size: bounds,
    padding: typeof padding === 'function' ? padding(bounds) : padding,
    data: data,
    xyRange: xyRange,
  }
}
