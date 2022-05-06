import _ from 'lodash'
import {rectExt, RectExt} from '../../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {ChartStateDerived} from '../Chart'
import {ChartUtils} from '../ChartUtils/ChartUtils'
import {
  ChartPointerComponentDrawArg,
  ChartPointerComponentTargetArgs,
} from './ChartPointer'

const minPixelsPerTickForSmall = 15

export type ChartPointerXAxisAnimatedState = {
  pointerBoxWithoutX: {y: number; width: number; height: number}
  markDataX: number
}

export const chartPointerXAxis = <Data>({
  ctx,
  formatX: formatXIn,
  displayX,
  ...target
}: ChartPointerComponentTargetArgs) => {
  const formatX = (x: number) => formatXIn(x, 'short')
  const {animated, notAnimated} = (() => {
    const {dataX, markDataX, chartStateDerived} = target
    const {scale, plotArea} = chartStateDerived
    const graphX = scale.x(dataX)
    const type = _tickType(displayX, dataX)
    const tickInfo = _tickInfo(type)
    ctx.font = tickInfo.font
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    const pad = {top: 0, bottom: 2, horz: 4}
    const measure = ctx.measureText(formatX(dataX))
    const width = measure.width + pad.horz * 2
    const labelGraphY = plotArea.bottom + tickInfo.length + 2
    const y = plotArea.bottom
    const bottom = labelGraphY + measure.actualBoundingBoxDescent + pad.bottom

    const animated = {
      pointerBoxWithoutX: {y, width, height: bottom - y},
      markDataX,
    }

    const {scale: targetScale} = chartStateDerived
    const pixelsPerTickAtTarget = targetScale.x(1) - targetScale.x(0)
    const drawLabelForSmall = pixelsPerTickAtTarget > minPixelsPerTickForSmall
    const notAnimated = {drawLabelForSmall}
    return {animated, notAnimated}
  })()

  const draw = ({
    dataX,
    animatedProps: {pointerBoxWithoutX, markDataX},
    chartStateDerived,
  }: ChartPointerComponentDrawArg<ChartPointerXAxisAnimatedState>) => {
    const {scale, plotArea} = chartStateDerived
    const {drawLabelForSmall} = notAnimated

    const pixelsPerTick = scale.x(1) - scale.x(0)

    // Extend plotArea by 1 to help with floating point math.
    const min = Math.ceil(scale.x.inverse(plotArea.x - 1))
    const max = Math.floor(scale.x.inverse(plotArea.x + plotArea.width + 1)) + 1

    const dataXs = _.range(min, max)

    // Draw the ticks.
    dataXs.forEach(dataX => {
      ctx.save()
      _drawTick(
        ctx,
        chartStateDerived,
        false,
        true,
        displayX,
        dataX,
        formatX,
        drawLabelForSmall
      )
      ctx.restore()
    })

    const drawInMark = (
      markGraphY: number,
      dataX: number,
      includeLine: boolean
    ) => {
      const alphaFn = linearFnFomPoints(pixelsPerTick / 3, 1, pixelsPerTick, 0)
      ctx.globalAlpha = _.clamp(
        alphaFn(Math.abs(scale.x(dataX) - markGraphY)),
        0,
        1
      )
      _drawTick(
        ctx,
        chartStateDerived,
        true,
        includeLine,
        displayX,
        dataX,
        formatX,
        drawLabelForSmall
      )
    }

    // ---- MARK BOX ----
    const graphX = scale.x(dataX)
    const pointerBox = rectExt({
      ...pointerBoxWithoutX,
      x: graphX - pointerBoxWithoutX.width / 2,
    })

    const pointerGraphY = pointerBox.x + pointerBox.width / 2
    const markBox = _labelBox(
      ctx,
      {plotArea},
      formatX,
      Math.round(markDataX),
      scale.x(markDataX),
      false,
      displayX
    )

    const markGraphY = markBox.x + markBox.width / 2
    const markLine = rectExt({
      x: markBox.x + markBox.width / 2 - 3 / 2,
      y: plotArea.bottom - 20,
      width: 3,
      bottom: markBox.y + 3,
    })

    // Draw the mark box.
    ctx.beginPath()
    ChartUtils.roundRect(ctx, markBox, 10)
    ctx.fillStyle = ChartUtils.color.gray[700]
    ctx.fill()

    ctx.beginPath()
    ChartUtils.roundRect(ctx, markLine, 3)
    ctx.fillStyle = ChartUtils.color.gray[700]
    ctx.fill()

    ctx.beginPath()
    ChartUtils.roundRect(
      ctx,
      rectExt({
        x: markLine.x,
        bottom: plotArea.bottom + 3,
        width: markLine.width,
        y: linearFnFomPoints(
          0,
          markLine.y,
          pixelsPerTick,
          plotArea.bottom + 3
        )(Math.min(pixelsPerTick, Math.abs(pointerGraphY - markGraphY))),
      }),
      3
    )
    ctx.fillStyle = ChartUtils.color.theme1
    ctx.fill()

    // Draw over the mark box.
    ctx.save()
    ctx.beginPath()
    ChartUtils.roundRect(ctx, markLine, 3)
    ChartUtils.roundRect(ctx, markBox, 10)
    ctx.clip()
    drawInMark(markGraphY, Math.round(markDataX), false)
    drawInMark(markGraphY, Math.round(markDataX) - 1, false)
    drawInMark(markGraphY, Math.round(markDataX) + 1, false)
    ctx.restore()

    // ---- POINTER BOX ----
    // Draw the pointer box.
    ctx.beginPath()
    ChartUtils.roundRect(ctx, pointerBox, 10)
    ctx.fillStyle = ChartUtils.color.theme1
    ctx.fill()

    // Draw over the pointer box.
    ctx.save()
    ctx.beginPath()
    ChartUtils.roundRect(ctx, pointerBox, 10)
    ctx.clip()
    const pointerDataX = Math.round(scale.x.inverse(pointerGraphY))
    drawInMark(pointerGraphY, pointerDataX, pointerDataX !== markDataX)
    drawInMark(pointerGraphY, pointerDataX - 1, true)
    drawInMark(pointerGraphY, pointerDataX + 1, true)
    ctx.restore()
  }

  return {draw, animated}
}

function _drawTick(
  ctx: CanvasRenderingContext2D,
  derivedState: ChartStateDerived,
  forPointer: boolean,
  includeLine: boolean,
  displayX: (x: number) => number,
  dataX: number,
  format: (x: number) => string,
  drawLabelForSmall: boolean
) {
  const {scale, plotArea} = derivedState
  const label = format(dataX)
  const type = _tickType(displayX, dataX)
  const tickInfo = _tickInfo(type)
  const graphY = plotArea.bottom
  const graphX = scale.x(dataX)
  const labelGraphY = graphY + tickInfo.length + 2

  // Draw the tick line.
  if (includeLine) {
    ctx.beginPath()
    ctx.moveTo(graphX, graphY)
    ctx.lineTo(graphX, graphY + tickInfo.length)
    ctx.strokeStyle = forPointer ? ChartUtils.color.gray[100] : tickInfo.style
    ctx.lineWidth = 1
    ctx.stroke()
  }

  // Draw the label.
  if (type !== 'small' || drawLabelForSmall || forPointer) {
    ctx.font = tickInfo.font
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillStyle = forPointer ? ChartUtils.color.gray[100] : tickInfo.style
    ctx.fillText(label, graphX, labelGraphY)
  }
}

const _labelBox = (
  ctx: CanvasRenderingContext2D,
  state: {plotArea: RectExt},
  format: (x: number) => string,
  dataX: number,
  graphX: number,
  includeTick: boolean,
  displayX: (x: number) => number
) => {
  const {plotArea} = state
  const type = _tickType(displayX, dataX)
  const tickInfo = _tickInfo(type)
  ctx.font = tickInfo.font
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  const pad = {top: includeTick ? 0 : 2, bottom: 2, horz: 4}
  const measure = ctx.measureText(format(dataX))
  const width = measure.width + pad.horz * 2
  const labelGraphY = plotArea.bottom + tickInfo.length + 2
  const y = includeTick
    ? plotArea.bottom
    : labelGraphY - measure.actualBoundingBoxAscent - pad.top
  const bottom = labelGraphY + measure.actualBoundingBoxDescent + pad.bottom
  return rectExt({
    x: graphX - width / 2,
    y,
    width,
    height: bottom - y,
  })
}

const _tickType = (displayX: (x: number) => number, xIn: number) => {
  const x = displayX(xIn)
  return x % 10 === 0
    ? ('large' as const)
    : x % 5 === 0
    ? ('medium' as const)
    : ('small' as const)
}

const _tickInfo = (type: 'large' | 'medium' | 'small') => {
  const result = (length: number, style: string, font: string) => ({
    length,
    style,
    font,
  })
  return type === 'large'
    ? result(15, ChartUtils.color.gray['700'], ChartUtils.getFont(12))
    : type === 'medium'
    ? result(9, ChartUtils.color.gray['500'], ChartUtils.getFont(10))
    : result(3, ChartUtils.color.gray['400'], ChartUtils.getFont(8))
}
