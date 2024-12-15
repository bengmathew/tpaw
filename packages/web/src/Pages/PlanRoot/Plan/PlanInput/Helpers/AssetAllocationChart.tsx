import { Power1 } from 'gsap'
import React, { useEffect, useRef, useState } from 'react'
import { rectExt } from '../../../../../Utils/Geometry'
import { fGet } from '../../../../../Utils/Utils'
import {
  ChartReact,
  ChartReactStatefull,
} from '../../../../Common/Chart/ChartReact'
import { getPlanResultsChartRange } from '../../PlanResults/PlanResultsChartCard/PlanResultsChart/GetPlanResultsChartRange'
import { getPlanResultsChartSalientPoints } from '../../PlanResults/PlanResultsChartCard/PlanResultsChart/GetPlanResultsChartSalientPoints'
import { PlanResultsChartData } from '../../PlanResults/PlanResultsChartCard/PlanResultsChart/PlanResultsChartData'
import { useChartData } from '../../WithPlanResultsChartData'

export const AssetAllocationChart = React.memo(
  ({
    className = '',
    type,
  }: {
    className?: string
    type:
      | 'asset-allocation-savings-portfolio'
      | 'asset-allocation-total-portfolio'
  }) => {
    const chartData = useChartData(type)

    const divRef = useRef<HTMLDivElement | null>(null)
    const [ref, setRef] = useState<ChartReactStatefull<{
      data: PlanResultsChartData
    }> | null>(null)

    const [width, setWidth] = useState(50)
    useEffect(() => {
      const div = fGet(divRef.current)
      const observer = new ResizeObserver(() =>
        setWidth(div.getBoundingClientRect().width),
      )
      observer.observe(div)
      return () => {
        observer.disconnect()
      }
    }, [])

    useEffect(() => {
      if (!ref) return
      const sizing = _sizing(width)
      fGet(divRef.current).style.height = `${sizing.position.height}px`
      ref.setSizing(sizing, null)
    }, [ref, width])

    useEffect(() => {
      if (!ref) return
      ref.setData(
        { data: chartData },
        {
          ease: Power1.easeOut,
          duration: 0.5,
        },
      )
    }, [chartData, ref])

    return (
      <div
        // pointer-events-none to allow scrolling while touch is on the chart.
        className={`${className} relative w-full border border-gray-200 bg-gray-50 rounded-md  pointer-events-none`}
        ref={divRef}
      >
        <ChartReact<{ data: PlanResultsChartData }>
          ref={setRef}
          starting={{
            data: { data: chartData },
            sizing: _sizing(width),
            propsFn: ({ data }) => ({
              dataRange: { x: data.displayRange.x, y: { start: 0, end: 1 } },
              includeWidthOfLastX: false,
            }),
            debug: false,
          }}
          components={components}
        />
      </div>
    )
  },
)

const _sizing = (width: number) => ({
  position: rectExt({ width, height: width * 0.4, x: 0, y: 0 }),
  padding: { left: 4, right: 4, top: 20, bottom: 20 },
})

const components = () => {
  const range = getPlanResultsChartRange('inline')
  return [range, getPlanResultsChartSalientPoints(range.ids)]
}
