import _ from 'lodash'
import {formatCurrency} from '../../../../Utils/FormatCurrency'
import {ChartComponent} from '../../../Common/Chart/ChartComponent/ChartComponent'
import {ChartContext} from '../../../Common/Chart/ChartContext'
import {chartDataTransitionCurrNumArr} from '../../../Common/Chart/ChartUtils/ChartDataTransition'
import {ChartUtils} from '../../../Common/Chart/ChartUtils/ChartUtils'
import {TPAWChartDataLegacy} from '../TPAWChart/TPAWChartDataLegacy'

const pad = {vert: 6, horz: 10}
const _fontSize = 11
const _lineLength = 8
export const chartDrawLegacy = (): ChartComponent<TPAWChartDataLegacy> => ({
  draw: (context: ChartContext<TPAWChartDataLegacy>) => {

    const {canvasContext: ctx, dataTransition, derivedState} = context

    const {scale, plotArea, viewport} = derivedState.curr
    const graphX = plotArea.right - _lineLength

    const majorDataYs = chartDataTransitionCurrNumArr(dataTransition, x =>
      x.percentiles.filter(x => x.isHighlighted).map(x => x.data)
    )
    const majorDataPercentiles = dataTransition.target.percentiles
      .filter(x => x.isHighlighted)
      .map(x => x.percentile)
    const minorDataYs = chartDataTransitionCurrNumArr(dataTransition, x =>
      x.percentiles.filter(x => !x.isHighlighted).map(x => x.data)
    )

    // Draw the minor lines.
    ctx.beginPath()
    minorDataYs.forEach(dataY => {
      const graphY = scale.y(dataY)
      ctx.moveTo(graphX, graphY)
      ctx.lineTo(graphX + _lineLength, graphY)
    })
    ctx.lineWidth = 0.3
    ctx.strokeStyle = ChartUtils.color.gray[900]
    ctx.stroke()

    // Draw the major lines.
    ctx.beginPath()
    majorDataYs.forEach(dataY => {
      const graphY = scale.y(dataY) 
      ctx.moveTo(graphX, graphY)
      ctx.lineTo(graphX + _lineLength, graphY)
    })
    ctx.lineWidth = 2
    ctx.strokeStyle = ChartUtils.color.gray[500]
    ctx.stroke()

    //Draw the targets.
    ctx.globalAlpha =  0.7
    ctx.beginPath()
    majorDataYs.forEach(dataY =>
      ctx.ellipse(graphX, scale.y(dataY), 4, 4, 0, 0, Math.PI * 4)
    )
    ctx.fillStyle = ChartUtils.color.gray[800]
    ctx.fill()

    // Draw the yLabels.
    ctx.textBaseline = 'alphabetic'
    ctx.font = ChartUtils.getMonoFont(_fontSize)
    ctx.fillStyle = ChartUtils.color.gray[900]
    const yLabelHeight = ctx.measureText('1').actualBoundingBoxAscent
    const yLabels = majorDataYs.map(y =>
      y < 1000
        ? `${formatCurrency(y)}`
        : y < 1000000
        ? `${formatCurrency(Math.round(y / 1000))}K`
        : `${formatCurrency(_.round(y / 1000000, 1), 1)}M`
    )
    const yLabelWidth = Math.max(...yLabels.map(y => ctx.measureText(y).width))
    const percentileLabelWidth = Math.max(
      ...majorDataPercentiles.map(x => ctx.measureText(`${x}`).width)
    )

    const yLabelsHeight = (yLabelHeight + pad.vert) * 5 + 15
    const yLabelsBottom = Math.max(
      scale.y(majorDataYs[2]),
      plotArea.y + yLabelsHeight
    )
    let yLabelGraphY = yLabelsBottom + yLabelHeight + pad.vert

    const percentileGraphX = plotArea.x + pad.horz
    const thGraphX = percentileGraphX + percentileLabelWidth
    const labelGraphX = thGraphX + 15

    const yLabelHandleX = labelGraphX + yLabelWidth

    majorDataYs.forEach((dataY, i) => {
      const graphY = scale.y(dataY)
      yLabelGraphY -= yLabelHeight + pad.vert

      ctx.fillStyle = ChartUtils.color.gray[900]
      ctx.textAlign = 'right'
      ctx.font = ChartUtils.getMonoFont(_fontSize)
      ctx.fillText(yLabels[i], labelGraphX + yLabelWidth, yLabelGraphY)

      ctx.fillStyle = ChartUtils.color.gray[700]
      ctx.textAlign = 'left'
      ctx.font = ChartUtils.getMonoFont(7)
      ctx.fillText('th', thGraphX, yLabelGraphY - 4)

      ctx.textAlign = 'right'
      ctx.font = ChartUtils.getMonoFont(_fontSize)
      ctx.fillText(
        `${majorDataPercentiles[i]}`,
        percentileGraphX + percentileLabelWidth,
        yLabelGraphY
      )

      // Draw the lines.
      ctx.beginPath()
      ctx.moveTo(graphX, graphY)
      const yLabelHandleY = yLabelGraphY - yLabelHeight / 2
      const lineWidth = graphX - yLabelHandleX
      ChartUtils.roundedLine(
        ctx,
        [
          {x: graphX, y: graphY},
          {x: yLabelHandleX + lineWidth * 0.4, y: yLabelHandleY},
          {x: yLabelHandleX, y: yLabelHandleY},
        ],
        10
      )
      ctx.strokeStyle = ChartUtils.color.gray[700]
      ctx.lineWidth = 1
      ctx.stroke()
    })
  },
})
