import _ from 'lodash'
import React, {useCallback, useMemo} from 'react'
import {fGet, noCase} from '../../../../Utils/Utils'
import {ChartXYRange} from '../../../Common/Chart/Chart'
import {chartDrawDataLines} from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {ChartMinMaxYAxis} from '../../../Common/Chart/ChartComponent/ChartMinMaxYAxis'
import {ChartPointer} from '../../../Common/Chart/ChartComponent/ChartPointer'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import {ChartUtils} from '../../../Common/Chart/ChartUtils/ChartUtils'
import {TPAWChartDataMain} from './TPAWChartDataMain'

export const TPAWChartMain = React.memo(
  React.forwardRef(
    (
      {
        starting,
      }: {
        starting: {
          data: TPAWChartDataMain
          xyRange: ChartXYRange
          sizing: ChartReactSizing
        }
      },
      ref: React.ForwardedRef<ChartReactStatefull<TPAWChartDataMain>>
    ) => {
      const components = useCallback(() => {
        const minorLine = chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 0.5,
          strokeStyle: ChartUtils.color.gray[400],
          dataFn: (data: TPAWChartDataMain) =>
            data.series.type === 'percentiles'
              ? {
                  lines: data.series.percentiles
                    .filter(x => !x.isHighlighted)
                    .map(x => x.data),
                }
              : data.series.type === 'labeledLines'
              ? {
                  lines: _.times(
                    data.series.percentiles.length -
                      data.series.highlightedPercentiles.length
                  ).map(() => null),
                }
              : noCase(data.series),
        })
        const majorLine = chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 1.2,
          strokeStyle: ChartUtils.color.gray[500],
          dataFn: (data: TPAWChartDataMain) =>
            data.series.type === 'percentiles'
              ? {
                  lines: data.series.percentiles
                    .filter(x => x.isHighlighted)
                    .map(x => x.data),
                }
              : {
                  lines: [
                    data.series.lines[0].data,
                    ..._.times(
                      data.series.highlightedPercentiles.length - 2
                    ).map(() => null),
                    data.series.lines[1].data,
                  ],
                },
        })

        const minMaxYAxis = new ChartMinMaxYAxis<TPAWChartDataMain>(
          (data, x) => data.yFormat(x),
          ChartUtils.color.gray[800],
          data => data.max.x,
          (data, x) => {
            switch (data.series.type) {
              case 'percentiles':
                return {
                  min: data.series.percentiles[0].data(x),
                  max: fGet(_.last(data.series.percentiles)).data(x),
                }
              case 'labeledLines':
                const ys = data.series.lines.map(({data}) => data(x))
                return {
                  min: Math.min(...ys),
                  max: Math.max(...ys),
                }
              default:
                noCase(data.series)
            }
          }
        )

        const pointer = new ChartPointer<TPAWChartDataMain>(
          data => {
            switch (data.series.type) {
              case 'percentiles':
                return data.series.percentiles
                  .filter(x => x.isHighlighted)
                  .map(x => ({line: x.data, label: `${x.percentile}`}))
              case 'labeledLines': {
                const spaw = {
                  line: data.series.lines[1].data,
                  label: data.series.lines[1].label,
                }
                const tpaw = {
                  line: data.series.lines[0].data,
                  label: data.series.lines[0].label,
                }
                const tpawIsFirst =
                  tpaw.line(data.years.displayRange.start) >=
                  spaw.line(data.years.displayRange.start)

                const [first, second] = tpawIsFirst
                  ? [tpaw, spaw]
                  : [spaw, tpaw]
                return [
                  second,
                  ..._.times(data.series.highlightedPercentiles.length - 2).map(
                    () => null
                  ),
                  first,
                ]
              }
              default:
                noCase(data.series)
            }
          },
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
        return [minorLine, majorLine, minMaxYAxis, pointer]
      }, [])

      return (
        <ChartReact<TPAWChartDataMain>
          ref={ref}
          starting={starting}
          components={components}
        />
      )
    }
  )
)
