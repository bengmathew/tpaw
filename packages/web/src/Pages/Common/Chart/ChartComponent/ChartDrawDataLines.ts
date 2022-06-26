import _ from 'lodash'
import {fGet} from '../../../../Utils/Utils'
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
    lines: (((x: number) => number) | null)[]
  }
}): ChartComponent<Data> => ({
  draw: (context: ChartContext<Data>) => {
    const {canvasContext: ctx, dataTransition, derivedState} = context
    const {scale, plotArea, viewport} = derivedState.curr

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

    const prevLines = dataFn(dataTransition.prev).lines
    const targetLines = dataFn(dataTransition.target).lines
    const mismatch = !_.isEqual(
      prevLines.map(x => x === null),
      targetLines.map(x => x === null)
    )

    prevLines.map((prevLine, i) => {
      const targetLine = targetLines[i]
      if (!mismatch && prevLine === null) return

      if (targetLine === null) return
      const transition = mismatch
        ? null
        : {
            prev: fGet(prevLine),
            target: targetLine,
            transition: dataTransition.transition,
          }

      dataXs.forEach((dataX, j) => {
        const dataY = transition
          ? chartDataTransitionCurrNum(transition, x => x(dataX))
          : targetLine(dataX)
        const graphX = scale.x(dataX)
        const graphY = scale.y(dataY)
        j === 0 ? ctx.moveTo(graphX, graphY) : ctx.lineTo(graphX, graphY)
      })
    })
    ctx.stroke()
  },
})
