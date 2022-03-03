import _ from 'lodash'
import {ChartContext} from '../ChartContext'
import {chartDataTransitionCurrNum} from '../ChartUtils/ChartDataTransition'
import {ChartComponent} from './ChartComponent'

export const chartDrawDataLines = <Data>({
  lineWidth,
  strokeStyle,
  dataFn,
}: {
  lineWidth: number
  strokeStyle: string
  dataFn: (data: Data) => {
    lines: ((x: number) => number)[]
    isXInGroup: (x: number) => boolean
  }
}): ChartComponent<Data> => ({
  draw: (context: ChartContext<Data>) => {
    const {ctx, stateTransition, currState} = context
    const {scale, plotArea, viewport} = currState

    const dataXs = _.range(
      Math.floor(scale.x.inverse(plotArea.x)),
      Math.ceil(scale.x.inverse(plotArea.x + plotArea.width)) + 1
    )

    ctx.beginPath()
    // Extend the plotArea on y axis below by lineWidth/2  otherwise
    // half the line width will be cropped if 0. Allow bleed into padding at top.
    ctx.rect(
      plotArea.x,
      viewport.y,
      plotArea.width,
      plotArea.y + plotArea.height + Math.ceil(lineWidth)
    )
    ctx.clip()
    ctx.lineWidth = lineWidth
    ctx.strokeStyle = strokeStyle
    ctx.beginPath()

    const {lines, isXInGroup} = dataFn(stateTransition.target.data)
    _.times(lines.length, i => {
      dataXs.forEach((dataX, j) => {
        const dataY = chartDataTransitionCurrNum(stateTransition, x =>
          dataFn(x.data).lines[i](dataX)
        )
        const graphX = scale.x(dataX)
        const graphY = scale.y(dataY)
        j === 0 || isXInGroup(dataX) !== isXInGroup(dataX - 1)
          ? ctx.moveTo(graphX, graphY)
          : ctx.lineTo(graphX, graphY)
      })
    })
    ctx.stroke()
  },
})
