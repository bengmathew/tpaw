import { Power1 } from 'gsap'
import _ from 'lodash'
import React, { useEffect, useRef, useState } from 'react'
import { rectExt } from '../../../../Utils/Geometry'
import { fGet } from '../../../../Utils/Utils'
import { useChartData } from '../../../App/WithChartData'
import { chartDrawDataLines } from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import { chartDrawDataRangeBand } from '../../../Common/Chart/ChartComponent/ChartDrawRangeBand'
import { ChartMinMaxYAxis } from '../../../Common/Chart/ChartComponent/ChartMinMaxYAxis'
import {
  ChartReact,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import { ChartUtils } from '../../../Common/Chart/ChartUtils/ChartUtils'
import { TPAWChartDataMain } from '../../PlanChart/TPAWChart/TPAWChartDataMain'

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
    const data = fGet(useChartData().byYearsFromNowPercentiles.get(type))

    const divRef = useRef<HTMLDivElement | null>(null)
    const [ref, setRef] =
      useState<ChartReactStatefull<TPAWChartDataMain> | null>(null)
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
      const xyRange = { x: data.months.displayRange, y: data.yDisplayRange }
      if (!ref) return
      ref.setState(data, xyRange, {
        ease: Power1.easeOut,
        duration: 0.5,
      })
    }, [data, ref])

    return (
      <div
        // pointer-events-none to allow scrolling while touch is on the chart.
        className={`${className} relative w-full border border-gray-200 bg-gray-50 rounded-md  pointer-events-none`}
        ref={divRef}
      >
        <ChartReact<TPAWChartDataMain>
          ref={setRef}
          starting={{
            data,
            xyRange: { x: data.months.displayRange, y: data.yDisplayRange },
            sizing: _sizing(width),
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

const components = () => [
  chartDrawDataRangeBand<TPAWChartDataMain>({
    fillStyle: ChartUtils.color.withOpacity(ChartUtils.color.gray[400], 0.5),
    dataFn: (data: TPAWChartDataMain) => ({
      min: fGet(_.first(data.percentiles)).data,
      max: fGet(_.last(data.percentiles)).data,
    }),
  }),

  chartDrawDataLines<TPAWChartDataMain>({
    lineWidth: 1.2,
    strokeStyle: ChartUtils.color.gray[500],
    dataFn: (data: TPAWChartDataMain) => {
      return {
        lines: data.percentiles
          .filter((x) => x.percentile === 50)
          .map((x) => x.data),
      }
    },
  }),

  new ChartMinMaxYAxis<TPAWChartDataMain>(
    (data, x) => data.yFormat(x),
    ChartUtils.color.gray[800],
    (data) => data.max.x,
    (data, x) => {
      return {
        min: data.percentiles[0].data(x),
        max: fGet(_.last(data.percentiles)).data(x),
      }
    },
  ),
]
