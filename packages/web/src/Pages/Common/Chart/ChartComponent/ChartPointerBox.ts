import _ from 'lodash'
import {RectExt, rectExt} from '../../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {ChartUtils} from '../ChartUtils/ChartUtils'
import {
  ChartPointerComponentDrawArg,
  ChartPointerComponentTargetArgs,
} from './ChartPointer'

const pad = {
  vert: {top: 8, between: 6, bottom: 11},
  horz: {edge: 8, between: 10},
}
const idealBoxXOffset = 35

const verticalLineDash = [10, 5]

export type ChartPointerBoxAnimatedState = {
  box: RectExt
  boxXTarget: number
}
export const chartPointerBox = (
  target: ChartPointerComponentTargetArgs,
) => {
  const {animated, notAnimated} = (() => {
    const {ctx, dataX, dataYInfos, chartStateDerived, formatX, formatY, showTh} = target
    const {plotArea, viewport, scale} = chartStateDerived
    const graphX = scale.x(dataX)
    const graphYs = dataYInfos.map(y => scale.y(y.dataY))

    // Step 1: Compute inside of box (not animated).
    const textInfos = {
      header: _measureText(ctx, formatX(dataX, 'long'), 14, 'bold'),
      lines: dataYInfos.map(({label: text, dataY}) => ({
        label: _measureText(ctx, text, 11),
        dataY: _measureText(ctx, formatY(dataY), 11),
      })),
      th: showTh ? _measureText(ctx, 'th', 7) : null,
    }
    const headerPosition = {x: pad.horz.edge, y: pad.vert.top}

    let labelGraphY =
      headerPosition.y + textInfos.header.height + pad.vert.between
    const labelGraphYs = textInfos.lines.map(({label}) => {
      const result = labelGraphY
      labelGraphY += label.height + pad.vert.between
      return result
    })

    const labelsMaxWidth = Math.max(...textInfos.lines.map(x => x.label.width))
    const labelRight = pad.horz.edge + labelsMaxWidth
    const notAnimated = {textInfos, headerPosition, labelGraphYs, labelRight}

    // Step 2: Compute the box (animated).
    const boxH = labelGraphY - pad.vert.between + pad.vert.bottom

    const dataYsMaxWidth = Math.max(...textInfos.lines.map(x => x.dataY.width))
    const horzBetweenPad =
      pad.horz.between *
      (viewport.width > 500
        ? 1
        : linearFnFomPoints(200, 0.5, 500, 1)(viewport.width))
    const boxW = Math.max(
      labelRight +
        (textInfos.th?.width ?? 0) +
        horzBetweenPad +
        dataYsMaxWidth +
        pad.horz.edge,
      pad.horz.edge * 2 + textInfos.header.width
    )

    const boxY = _.clamp(
      _.mean(graphYs) - boxH * 0.45,
      viewport.y,
      plotArea.y + plotArea.height - boxH - 10
    )

    let boxXOffset =
      idealBoxXOffset *
      (viewport.width > 500
        ? 1
        : linearFnFomPoints(200, 0.3, 500, 1)(viewport.width))
    const boxPad =
      viewport.width > 500
        ? 25
        : linearFnFomPoints(200, 0, 500, 25)(viewport.width)
    const boxSide =
      graphX + boxXOffset + boxW + boxPad < viewport.right ? 1 : -1
    const boxX = graphX + (boxSide === -1 ? -boxXOffset - boxW : boxXOffset)
    const boxXTarget = boxSide === -1 ? boxX + boxW : boxX
    const box = rectExt({x: boxX, y: boxY, height: boxH, width: boxW})

    const animated = {box, boxXTarget}
    return {notAnimated, animated}
  })()

  const draw = ({
    dataX,
    dataYInfos,
    chartStateDerived: {scale, plotArea},
    animatedProps: {box, boxXTarget},
  }: ChartPointerComponentDrawArg<ChartPointerBoxAnimatedState>) => {
    const {ctx} = target
    const {headerPosition, textInfos, labelGraphYs, labelRight} = notAnimated
    const graphX = scale.x(dataX)
    const graphYs = dataYInfos.map(y => scale.y(y.dataY))
    ctx.textBaseline = 'top'

    // Draw the box.
    ctx.beginPath()
    ChartUtils.roundRect(ctx, box, 10)
    ctx.fillStyle = ChartUtils.color.gray[700]
    ctx.fill()

    // Draw the header.
    ctx.font = textInfos.header.font
    ctx.textAlign = 'left'
    ctx.fillStyle = ChartUtils.color.theme1Light
    ctx.fillText(
      textInfos.header.text,
      box.x + headerPosition.x,
      box.y + headerPosition.y
    )

    // Draw the text lines.
    ctx.fillStyle = ChartUtils.color.gray[200]
    textInfos.lines.forEach(({label, dataY}, i) => {
      const y = box.y + labelGraphYs[i]

      // Draw the label.
      ctx.font = label.font
      ctx.textAlign = 'right'
      ctx.fillText(label.text, box.x + labelRight, y)

      // Draw the th.]
      if (textInfos.th) {
        ctx.font = textInfos.th.font
        ctx.textAlign = 'left'
        ctx.fillText(textInfos.th.text, box.x + labelRight, y - 2)
      }
      // Draw the dataY.
      ctx.font = dataY.font
      ctx.textAlign = 'right'
      ctx.fillText(dataY.text, box.right - pad.horz.edge, y)
    })

    // Draw the vertical line.
    ctx.globalAlpha = 0.3
    ctx.lineWidth = 1.5
    ctx.strokeStyle = ChartUtils.color.gray[500]
    ctx.setLineDash(verticalLineDash)
    ctx.lineDashOffset = verticalLineDash[0]
    ctx.beginPath()
    ctx.moveTo(graphX, plotArea.y + plotArea.height)
    ctx.lineTo(graphX, Math.min(...graphYs))
    ctx.stroke()
    ctx.setLineDash([])

    // Draw the target on the data line.
    ctx.globalAlpha = 0.7
    ctx.lineWidth = 2
    ctx.fillStyle = ChartUtils.color.gray[900]
    graphYs.forEach(graphY => {
      ctx.beginPath()
      ctx.ellipse(graphX, graphY, 4, 4, 0, 0, Math.PI * 4)
      ctx.fill()
    })

    // Draw the lines.
    ctx.globalAlpha = 1
    ctx.lineWidth = 1
    ctx.strokeStyle = ChartUtils.color.gray[700]
    graphYs.forEach((graphY, i) => {
      const graphYOnBox =
        box.y + labelGraphYs[i] + textInfos.lines[i].label.height / 2
      ctx.beginPath()

      const line = [
        {x: graphX, y: graphY},
        {x: graphX + (boxXTarget - graphX) * 0.6, y: graphYOnBox},
        {x: boxXTarget, y: graphYOnBox},
      ]
      ChartUtils.roundedLine(ctx, line, 10)
      ctx.stroke()
    })
  }

  return {animated, draw}
}

const _measureText = (
  ctx: CanvasRenderingContext2D,
  text: string,
  fontSize: number,
  style?: 'bold' | ''
) => {
  const font = ChartUtils.getMonoFont(fontSize, style)
  ctx.font = font
  const {width, actualBoundingBoxAscent, actualBoundingBoxDescent} =
    ctx.measureText(text)
  return {
    text,
    font,
    width,
    height: actualBoundingBoxAscent + actualBoundingBoxDescent,
  }
}
