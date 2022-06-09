import {Power1} from 'gsap'
import _ from 'lodash'
import React, {useEffect, useMemo, useRef, useState} from 'react'
import {rectExt} from '../../../../Utils/Geometry'
import {fGet} from '../../../../Utils/Utils'
import {useChartMainData} from '../../../App/WithChartMainData'
import {chartDrawDataLines} from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {ChartMinMaxYAxis} from '../../../Common/Chart/ChartComponent/ChartMinMaxYAxis'
import {
  ChartReact,
  ChartReactState,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import {ChartUtils} from '../../../Common/Chart/ChartUtils/ChartUtils'
import {TPAWChartDataMain} from '../../ChartPanel/TPAWChart/TPAWChartDataMain'

export const AssetAllocationChart = React.memo(
  ({className = ''}: {className?: string}) => {
    const data = fGet(useChartMainData().get('glide-path'))
    const [state, setState] = useState(() => _processData(data))

    useEffect(() => {
      setState(_processData(data))
    }, [data])

    const components = useMemo(
      () => [
        chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 0.2,
          strokeStyle: ChartUtils.color.gray[500],
          dataFn: (data: TPAWChartDataMain) => ({
            lines: data.percentiles
              .filter(x => !x.isHighlighted)
              .map(x => x.data),
          }),
        }),

        chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 1.2,
          strokeStyle: ChartUtils.color.gray[500],
          dataFn: (data: TPAWChartDataMain) => ({
            lines: data.percentiles
              .filter(x => x.isHighlighted)
              .map(x => x.data),
          }),
        }),
        new ChartMinMaxYAxis<TPAWChartDataMain>(
          (data, x) => data.yFormat(x),
          ChartUtils.color.gray[800],
          data => data.max.x,
          (data, x) => ({
            min: data.percentiles[0].data(x),
            max: fGet(_.last(data.percentiles)).data(x),
          })
        ),
      ],
      []
    )

    const divRef = useRef<HTMLDivElement | null>(null)
    const ref = useRef<ChartReactStatefull | null>(null)
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

    useEffect(()=>{
      const sizing = _sizing(width)
      fGet(divRef.current).style.height = `${sizing.position.height}px`
      fGet(ref.current).setSizing(sizing)
    }, [width])

    return (
      <div
        className={`${className} relative w-full border border-gray-200 bg-gray-50 rounded-md `}
        ref={divRef}
      >
        <ChartReact<TPAWChartDataMain>
          ref={ref}
          state={state}
          starting={{sizing: _sizing(width)}}
          components={components}
        />
      </div>
    )
  }
)

const _processData = (
  data: TPAWChartDataMain
): ChartReactState<TPAWChartDataMain> => ({
  data,
  xyRange: {
    x: data.years.displayRange,
    y: data.yDisplayRange,
  },
  animation: {ease: Power1.easeOut, duration: 0.5},
})

const _sizing = (width: number) => ({
  position: rectExt({width, height: width * 0.4, x: 0, y: 0}),
  padding: {left: 4, right: 4, top: 20, bottom: 20},
})
