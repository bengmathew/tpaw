import React, {CSSProperties, useEffect, useRef, useState} from 'react'
import Measure from 'react-measure'
import {assert, fGet} from '../../../Utils/Utils'
import {Chart, ChartPadding, ChartXYRange} from './Chart'
import {ChartComponent} from './ChartComponent/ChartComponent'

export function ChartReact<Data>({
  animated: {padding, data, xyRange},
  components,
  className = '',
  style,
  stateKey,
}: {
  animated: {
    padding:
      | ChartPadding
      | ((bounds: {width: number; height: number}) => ChartPadding)
    data: Data
    xyRange: ChartXYRange
  }
  components: readonly ChartComponent<Data>[]
  className?: string
  stateKey: string | number
  style?: CSSProperties
}) {
  const chartRef = useRef<Chart<Data> | null>(null)
  const [bounds, setBounds] = useState<{width: number; height: number} | null>(
    null
  )

  useEffect(() => {

    if (bounds) {
      fGet(chartRef.current).setState(
        {
          size: bounds,
          padding: typeof padding === 'function' ? padding(bounds) : padding,
          data,
          xyRange,
        },
        stateKey
      )
    }
  }, [bounds, data, xyRange, padding, stateKey])

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
                        {
                          size: bounds,
                          data,
                          xyRange,
                          padding:
                            typeof padding === 'function'
                              ? padding(bounds)
                              : padding,
                        },
                        stateKey,
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
