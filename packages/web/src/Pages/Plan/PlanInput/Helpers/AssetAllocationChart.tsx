import {Power1} from 'gsap'
import _ from 'lodash'
import React, {useEffect, useRef, useState} from 'react'
import {rectExt} from '../../../../Utils/Geometry'
import {assert, fGet} from '../../../../Utils/Utils'
import {useChartData} from '../../../App/WithChartData'
import {chartDrawDataLines} from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {ChartMinMaxYAxis} from '../../../Common/Chart/ChartComponent/ChartMinMaxYAxis'
import {ChartReact, ChartReactStatefull} from '../../../Common/Chart/ChartReact'
import {ChartUtils} from '../../../Common/Chart/ChartUtils/ChartUtils'
import {TPAWChartDataMain} from '../../PlanChart/TPAWChart/TPAWChartDataMain'

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
        setWidth(div.getBoundingClientRect().width)
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
      const xyRange = {x: data.years.displayRange, y: data.yDisplayRange}
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
            xyRange: {x: data.years.displayRange, y: data.yDisplayRange},
            sizing: _sizing(width),
          }}
          components={components}
        />
      </div>
    )
  }
)

const _sizing = (width: number) => ({
  position: rectExt({width, height: width * 0.4, x: 0, y: 0}),
  padding: {left: 4, right: 4, top: 20, bottom: 20},
})

const components = () => [
  chartDrawDataLines<TPAWChartDataMain>({
    lineWidth: 0.2,
    strokeStyle: ChartUtils.color.gray[500],
    dataFn: (data: TPAWChartDataMain) => {
      assert(data.series.type === 'percentiles')
      return {
        lines: data.series.percentiles
          .filter(x => !x.isHighlighted)
          .map(x => x.data),
      }
    },
  }),

  chartDrawDataLines<TPAWChartDataMain>({
    lineWidth: 1.2,
    strokeStyle: ChartUtils.color.gray[500],
    dataFn: (data: TPAWChartDataMain) => {
      assert(data.series.type === 'percentiles')
      return {
        lines: data.series.percentiles
          .filter(x => x.isHighlighted)
          .map(x => x.data),
      }
    },
  }),
  new ChartMinMaxYAxis<TPAWChartDataMain>(
    (data, x) => data.yFormat(x),
    ChartUtils.color.gray[800],
    data => data.max.x,
    (data, x) => {
      assert(data.series.type === 'percentiles')
      return {
        min: data.series.percentiles[0].data(x),
        max: fGet(_.last(data.series.percentiles)).data(x),
      }
    }
  ),
]
