import _ from 'lodash'
import React from 'react'
import {formatPercentage} from '../../../../Utils/FormatPercentage'
import {fGet, noCase} from '../../../../Utils/Utils'
import {ChartXYRange} from '../../../Common/Chart/Chart'
import {chartDrawDataLines} from '../../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {chartDrawText} from '../../../Common/Chart/ChartComponent/ChartDrawText'
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

const components = () => {
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
              ..._.times(data.series.highlightedPercentiles.length - 3).map(
                () => null
              ),
              data.series.lines[1].data,
              data.series.lines[2].data,
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
          const tpaw = {
            line: data.series.lines[0].data,
            label: data.series.lines[0].label,
          }
          const spaw = {
            line: data.series.lines[1].data,
            label: data.series.lines[1].label,
          }
          const swr = {
            line: data.series.lines[2].data,
            label: data.series.lines[2].label,
          }
          const ordered = _.sortBy([tpaw, spaw, swr], x =>
            x.line(data.years.displayRange.start)
          )

          return [
            ordered[0],
            ordered[1],
            ..._.times(
              data.series.highlightedPercentiles.length - ordered.length
            ).map(() => null),
            ordered[2],
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
    data => data.years.retirement,
    data => data.type !== 'reward-risk-ratio-comparison'
  )

  const successRate = chartDrawText<TPAWChartDataMain>(
    (data, {size, padding}) => ({
      text: `Success Rate: ${formatPercentage(0)(data.successRate)}`,
      font: ChartUtils.getFont(17, '600'),
      fillStyle: ChartUtils.color.gray[700],
      textAlign: 'right',
      position: {
        graphX: size.width - padding.right + 5,
        graphY: size.height - padding.bottom + 55,
      },
    })
  )
  return [minorLine, majorLine, minMaxYAxis, pointer, successRate]
}
