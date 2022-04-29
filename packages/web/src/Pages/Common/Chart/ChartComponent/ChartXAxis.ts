import _ from 'lodash'
import {Rect, rectExt, RectExt} from '../../../../Utils/Geometry'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {assert} from '../../../../Utils/Utils'
import {ChartStateDerived} from '../Chart'
import {chartDataTransitionCurrObj} from '../ChartUtils/ChartDataTransition'
import {ChartUtils} from '../ChartUtils/ChartUtils'
import {
  ChartPointerComponent,
  ChartPointerContext,
  ChartPointerState,
} from './ChartPointer'

const minPixelsPerTickForSmall = 15
export class ChartXAxis<Data> implements ChartPointerComponent<Data> {
  private _stateTransition: {prev: Rect; target: Rect} | null = null

  constructor(
    private transform: (data: Data, x: number) => number,
    private format: (data: Data, x: number) => string,
    private markFn: (data: Data) => number
  ) {}

  setState(
    ctx: CanvasRenderingContext2D,
    data: Data,
    chartStateDerived: ChartStateDerived,
    pointerState: ChartPointerState,
    pointerTransition: number
  ) {
    const {scale, plotArea} = chartStateDerived
    const target = _labelBox(
      ctx,
      {plotArea},
      x => this.format(data, this.transform(data, x)),
      pointerState.dataX,
      scale.x(pointerState.dataX),
      true,
      x => this.transform(data, x)
    )
    const prev = this._stateTransition
      ? chartDataTransitionCurrObj(
          {...this._stateTransition, transition: pointerTransition},
          x => x
        )
      : target

    this._stateTransition = {prev, target}
  }

  destroy() {}
  draw(context: ChartPointerContext<Data>) {
    const {
      canvasContext: ctx,
      pointerTransition,
      dataTransition,
      derivedState,
    } = context
    const {scale, plotArea} = derivedState.curr
    const transform = (x: number) => this.transform(dataTransition.target, x)
    const format = (x: number) =>
      this.format(dataTransition.target, transform(x))

    const pixelsPerTick = scale.x(1) - scale.x(0)
    assert(this._stateTransition)

    // Extend plotArea by 1 to help with floating point math.
    const min = Math.ceil(scale.x.inverse(plotArea.x - 1))
    const max = Math.floor(scale.x.inverse(plotArea.x + plotArea.width + 1)) + 1
    const ticks = new Map<number, _Tick<Data>>()
    _.range(min, max).forEach(dataX =>
      ticks.set(dataX, new _Tick(this.transform, dataX, this.format))
    )

    // Draw the ticks.
    ticks.forEach(x => {
      ctx.save()
      x.draw(context, false, true)
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
      ticks.get(dataX)?.draw(context, true, includeLine)
    }

    // ---- MARK BOX ----
    const pointerBox = chartDataTransitionCurrObj(
      {...this._stateTransition, transition: pointerTransition.transition},
      x => x
    )

    const pointerGraphY = pointerBox.x + pointerBox.width / 2
    const markBox = chartDataTransitionCurrObj(dataTransition, x => {
      const markDataX = this.markFn(x)
      return _labelBox(
        ctx,
        {plotArea},
        format,
        Math.round(markDataX),
        scale.x(markDataX),
        false,
        transform
      )
    })
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
    const markDataX = Math.round(scale.x.inverse(markGraphY))
    drawInMark(markGraphY, markDataX, false)
    drawInMark(markGraphY, markDataX - 1, false)
    drawInMark(markGraphY, markDataX + 1, false)
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
}

class _Tick<Data> {
  constructor(
    private transform: (data: Data, x: number) => number,
    private dataX: number,
    private format: (data: Data, x: number) => string
  ) {}

  draw(
    context: ChartPointerContext<Data>,
    forPointer: boolean,
    includeLine: boolean
  ) {
    const {canvasContext: ctx, dataTransition, derivedState} = context
    const {scale, plotArea} = derivedState.curr
    const transform = (x: number) => this.transform(dataTransition.target, x)
    const format = (x: number) =>
      this.format(dataTransition.target, transform(x))
    const label = format(this.dataX)
    const type = _tickType(transform, this.dataX)
    const tickInfo = _tickInfo(type)
    const graphY = plotArea.bottom
    const {scale: targetScale} = derivedState.target
    const pixelsPerTickAtTarget = targetScale.x(1) - targetScale.x(0)
    const graphX = scale.x(this.dataX)
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
    if (
      type !== 'small' ||
      pixelsPerTickAtTarget > minPixelsPerTickForSmall ||
      forPointer
    ) {
      ctx.font = tickInfo.font
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = forPointer ? ChartUtils.color.gray[100] : tickInfo.style
      ctx.fillText(label, graphX, labelGraphY)
    }
  }
}

const _labelBox = (
  ctx: CanvasRenderingContext2D,
  state: {plotArea: RectExt},
  format: (x: number) => string,
  dataX: number,
  graphX: number,
  includeTick: boolean,
  transform: (x: number) => number
) => {
  const {plotArea} = state
  const type = _tickType(transform, dataX)
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

const _tickType = (transform: (x: number) => number, xIn: number) => {
  const x = transform(xIn)
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
