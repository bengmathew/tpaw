import _ from 'lodash'
import React, {useMemo} from 'react'
import {fGet} from '../../../../Utils/Utils'
import {chartDrawDataLines} from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {ChartMinMaxYAxis} from '../../../Common/Chart/ChartComponent/ChartMinMaxYAxis'
import {ChartPointer} from '../../../Common/Chart/ChartComponent/ChartPointer'
// import {ChartPointerXAxis} from '../../../Common/Chart/ChartComponent/ChartPointerXAxis'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactState,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import {ChartUtils} from '../../../Common/Chart/ChartUtils/ChartUtils'
import {TPAWChartDataMain} from './TPAWChartDataMain'

export const TPAWChartMain = React.memo(
  React.forwardRef(
    (
      {
        state,
        starting,
      }: {
        state: ChartReactState<TPAWChartDataMain>
        starting: {sizing: ChartReactSizing}
      },
      ref: React.ForwardedRef<ChartReactStatefull>
    ) => {
      const components = useMemo(() => {
        const minorLine = chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 0.5,
          strokeStyle: ChartUtils.color.gray[400],
          dataFn: (data: TPAWChartDataMain) => ({
            lines: data.percentiles
              .filter(x => !x.isHighlighted)
              .map(x => x.data),
          }),
        })
        const majorLine = chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 1.2,
          strokeStyle: ChartUtils.color.gray[500],
          dataFn: (data: TPAWChartDataMain) => ({
            lines: data.percentiles
              .filter(x => x.isHighlighted)
              .map(x => x.data),
          }),
        })

        const minMaxYAxis = new ChartMinMaxYAxis<TPAWChartDataMain>(
          (data, x) => data.yFormat(x),
          ChartUtils.color.gray[800],
          data => data.max.x,
          (data, x) => ({
            min: data.percentiles[0].data(x),
            max: fGet(_.last(data.percentiles)).data(x),
          })
        )

        const pointer = new ChartPointer<TPAWChartDataMain>(
          data =>
            data.percentiles
              .filter(x => x.isHighlighted)
              .map(x => ({line: x.data, label: `${x.percentile}`})),
          (data, x) => data.years.display(x),
          (data: TPAWChartDataMain, x: number, type) =>
            x === data.years.max + 1
              ? type === 'short'
                ? 'L'
                : 'Legacy'
              : `${type === 'short' ? '' : 'Age '}${data.years.display(x)}`,
          (data, x) => data.yFormat(x),
          data => data.years.retirement
        )
        const byName = {minorLine, majorLine, minMaxYAxis, pointer}
        const arr = [minorLine, majorLine, minMaxYAxis, pointer]
        return {byName, arr}
      }, [])

      return (
        <ChartReact<TPAWChartDataMain>
          ref={ref}
          state={state}
          starting={starting}
          components={components.arr}
        />
      )
    }
  )
)
