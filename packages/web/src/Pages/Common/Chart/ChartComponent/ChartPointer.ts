import {gsap} from 'gsap'
import _ from 'lodash'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {assert, fGet, noCase} from '../../../../Utils/Utils'
import {ChartStateDerived} from '../Chart'
import {ChartContext} from '../ChartContext'
import {
  ChartDataTransition,
  chartDataTransitionCurrNumArr,
  chartDataTransitionCurrObj,
} from '../ChartUtils/ChartDataTransition'
import {ChartUtils} from '../ChartUtils/ChartUtils'
import {ChartComponent, ChartRegisterAnimation} from './ChartComponent'

const pad = {
  vert: {edge: 6, between: 6},
  horz: {edge: 8, between: 20},
}
const duration = 1

const idealBoxXOffset = 35
const lineDash = [10, 5]
const xLabelFontSize = 14

type _DataFn<Data> = (
  data: Data
) => {line: (x: number) => number; label: string}[]

export type ChartPointerState = {dataX: number}
export type ChartPointerContext<Data> = ChartContext<Data> & {
  pointerTransition: ChartDataTransition<ChartPointerState>
}
export interface ChartPointerComponent<Data> {
  setState: (
    ctx: CanvasRenderingContext2D,
    data: Data,
    chartStateDerived: ChartStateDerived,
    pointerState: ChartPointerState,
    pointerTransition: number,
    registerAnimation: ChartRegisterAnimation
  ) => void
  draw: (context: ChartPointerContext<Data>) => void
  destroy: () => void
}

export class ChartPointer<Data> implements ChartComponent<Data> {
  private _state: {
    notAnimatedBoxInfo: _NotAnimatedBoxInfo
    transition: ChartDataTransition<{dataX: number} & _AnimatedBoxInfo>
  } | null = null
  private _animation: gsap.core.Tween | null = null

  private _components: readonly ChartPointerComponent<Data>[]

  constructor(
    public dataFn: _DataFn<Data>,
    public formatX: (data: Data, x: number) => string,
    public formatY: (x: number) => string,
    components: readonly ChartPointerComponent<Data>[]
  ) {
    this._components = components
  }

  getComponents() {
    return this._components
  }

  destroy() {
    this._animation?.kill()
    this._components.forEach(x => x.destroy())
  }

  update(
    change: 'init' | 'pointer' | 'state' | 'size',
    context: ChartContext<Data>,
    registerAnimation: ChartRegisterAnimation
  ) {
    const {canvasContext: ctx, pointer, derivedState, dataTransition} = context
    const dataX = (() => {
      switch (change) {
        case 'init':
        case 'pointer': {
          const {scale, plotArea} = derivedState.target
          return Math.round(
            _.clamp(
              scale.x.inverse(pointer.x),
              scale.x.inverse(plotArea.x),
              scale.x.inverse(plotArea.right)
            )
          )
        }
        case 'state': {
          assert(this._state)
          const {scale, plotArea} = derivedState.target
          return _.clamp(
            this._state.transition.target.dataX,
            Math.ceil(scale.x.inverse(plotArea.x)),
            Math.floor(scale.x.inverse(plotArea.right))
          )
        }

        case 'size':
          assert(this._state)
          return this._state.transition.target.dataX
        default:
          noCase(change)
      }
    })()

    const targetBoxInfo = _boxInfo(
      dataTransition.target,
      derivedState.target,
      dataX,
      this.dataFn,
      this.formatY,
      this.formatX,
      ctx
    )

    const jumpToEnd = change === 'init' || change === 'size'

    this._animation?.kill()
    const oldTransition = this._state?.transition.transition ?? 0
    this._state = {
      notAnimatedBoxInfo: targetBoxInfo.notAnimatedBoxInfo,
      transition: jumpToEnd
        ? {
            prev: {dataX, ...targetBoxInfo.animatedBoxInfo},
            target: {dataX, ...targetBoxInfo.animatedBoxInfo},
            transition: 1,
          }
        : {
            prev: chartDataTransitionCurrObj(
              fGet(this._state).transition,
              x => x
            ),
            target: {dataX, ...targetBoxInfo.animatedBoxInfo},
            transition: 0,
          },
    }

    if (!jumpToEnd) {
      this._animation = registerAnimation(
        gsap.to(this._state.transition, {
          transition: 1,
          duration,
          ease: 'power4',
        })
      )
    }

    for (const component of this._components) {
      ctx.save()
      component.setState(
        ctx,
        dataTransition.target,
        derivedState.target,
        {dataX},
        oldTransition,
        registerAnimation
      )
      ctx.restore()
    }
  }

  draw(context: ChartContext<Data>) {
    const {canvasContext: ctx, dataTransition, derivedState} = context
    const {scale, plotArea} = derivedState.curr

    assert(this._state)

    const {dataX, boxSide, boxY, boxW, boxXOffset} = chartDataTransitionCurrObj(
      this._state.transition,
      x => x
    )

    const pointerTransition = {
      prev: {dataX: this._state.transition.prev.dataX},
      target: {dataX: this._state.transition.target.dataX},
      transition: this._state.transition.transition,
    }
    this._components.forEach(x => {
      ctx.save()
      x.draw({...context, pointerTransition})
      ctx.restore()
    })

    const graphX = scale.x(dataX)
    const dataYs = chartDataTransitionCurrNumArr(dataTransition, x =>
      this.dataFn(x).map(({line}) => line(dataX))
    )
    const graphYs = dataYs.map(y => scale.y(y))

    // Draw the vertical line.
    ctx.globalAlpha = 0.3
    ctx.setLineDash(lineDash)
    ctx.lineDashOffset = lineDash[0]
    ctx.beginPath()
    ctx.moveTo(graphX, plotArea.y + plotArea.height)
    // ctx.lineTo(graphX, viewport.y)
    ctx.lineTo(graphX, Math.min(...graphYs))
    ctx.lineWidth = 1.5
    ctx.strokeStyle = ChartUtils.color.gray[500]
    ctx.stroke()
    ctx.setLineDash([])

    // Draw the target.
    ctx.globalAlpha = 0.7
    graphYs.forEach((graphY, i) => {
      ctx.beginPath()
      ctx.ellipse(graphX, graphY, 4, 4, 0, 0, Math.PI * 4)
      ctx.lineWidth = 2
      ctx.fillStyle = ChartUtils.color.gray[900]
      ctx.fill()
    })

    // Draw the box.
    const boxPosition = {x: graphX, boxY, boxW, boxSide, boxXOffset}
    const textCenterGraphYs = _drawBox(
      this._state.notAnimatedBoxInfo,
      boxPosition,
      graphYs,
      ctx
    )

    // Draw the lines.
    const boxXOffsetSided = boxSide * boxXOffset
    ctx.globalAlpha = 1
    ctx.lineWidth = 1
    graphYs.forEach((graphY, i) => {
      const textCenterGraphY = textCenterGraphYs[i]
      ctx.beginPath()
      const line = [
        {x: graphX, y: graphY},
        {x: graphX + boxXOffsetSided * 0.6, y: textCenterGraphY},
        {x: graphX + boxXOffsetSided, y: textCenterGraphY},
      ]
      ChartUtils.roundedLine(ctx, line, 10)
      ctx.strokeStyle = ChartUtils.color.gray[700]
      ctx.stroke()
    })
  }
}

type _NotAnimatedBoxInfo = ReturnType<typeof _boxInfo>['notAnimatedBoxInfo']
type _AnimatedBoxInfo = ReturnType<typeof _boxInfo>['animatedBoxInfo']
const _boxInfo = <Data>(
  data: Data,
  derivedState: ChartStateDerived,
  dataX: number,
  dataFn: _DataFn<Data>,
  formatY: (x: number) => string,
  formatX: (data: Data, x: number) => string,
  ctx: CanvasRenderingContext2D
) => {
  const {plotArea, viewport, scale} = derivedState
  const graphX = Math.round(scale.x(dataX))
  const dataYs = dataFn(data).map(({line}) => line(dataX))
  const graphYs = dataYs.map(y => scale.y(y))

  const yLabels = dataFn(data).map(x => x.label)
  const yTexts = dataYs.map((y, i) => `${formatY(y)}`)

  ctx.font = ChartUtils.getMonoFont(11)
  const yTextMeasures = yTexts.map(y => ctx.measureText(y))
  const yLabelMeasures = yLabels.map(y => ctx.measureText(y))
  const yTextMeasureMaxWidth = Math.max(...yTextMeasures.map(x => x.width))
  const yLabelMeasureMaxWidth = Math.max(...yLabelMeasures.map(x => x.width))
  const xText = formatX(data, dataX)

  ctx.font = ChartUtils.getFont(xLabelFontSize, 'bold')
  const xTextMeasure = ctx.measureText(xText)
  const xTextHeight = xTextMeasure.actualBoundingBoxAscent
  const xTextWidth = xTextMeasure.width

  // Assume there is nothing below the line.
  const textH = yTextMeasures[0].actualBoundingBoxAscent

  const boxH =
    pad.vert.edge * 2 +
    xTextHeight +
    pad.vert.between +
    yTexts.length * (textH + pad.vert.between) -
    pad.vert.between

  const boxY = _.clamp(
    _.mean(graphYs) - boxH * 0.45,
    viewport.y,
    plotArea.y + plotArea.height - boxH - 10
  )
  const boxW =
    Math.max(
      xTextWidth,
      yLabelMeasureMaxWidth + yTextMeasureMaxWidth + pad.horz.between
    ) +
    pad.horz.edge * 2

  // const boxXOffset = Math.min(idealBoxXOffset, plotArea.width / 2 - boxW)
  // const boxRight = graphX + boxXOffset + boxW
  let boxXOffset =
    idealBoxXOffset *
    (viewport.width < 200 ? 0.3 : viewport.width < 400 ? 0.5 : 1)
  const boxPad = viewport.width > 500 ? 5 : -10
  const boxSide = graphX + boxXOffset + boxW + boxPad < plotArea.right ? 1 : -1
  boxXOffset = Math.min(
    boxXOffset,
    boxSide === 1
      ? viewport.right - 1 - graphX - boxW
      : graphX - (viewport.x + 1) - boxW
  )

  return {
    notAnimatedBoxInfo: {
      boxH,
      yTexts,
      textH,
      yLabelMeasureMaxWidth,
      yLabels,
      xText,
      xTextHeight,
    },
    animatedBoxInfo: {
      boxY,
      boxW,
      boxXOffset,
      boxSide,
    },
  }
}

const _drawBox = (
  {
    boxH,
    yTexts,
    textH,
    yLabelMeasureMaxWidth,
    yLabels,
    xText,
    xTextHeight,
  }: _NotAnimatedBoxInfo,
  {
    x: graphX,
    boxY,
    boxSide,
    boxXOffset,
    boxW,
  }: {
    x: number
    boxY: number
    boxSide: number
    boxXOffset: number
    boxW: number
  },
  graphYs: number[],
  ctx: CanvasRenderingContext2D
) => {
  const boxXOffsetSided = linearFnFomPoints(
    -1,
    -boxXOffset - boxW,
    1,
    boxXOffset
  )(boxSide)
  const box = {width: boxW, height: boxH, x: graphX + boxXOffsetSided, y: boxY}

  ctx.globalAlpha = 1

  // Draw the box.
  ctx.beginPath()
  ChartUtils.roundRect(ctx, box, 10)
  ctx.fillStyle = ChartUtils.color.gray[700]
  ctx.fill()

  // Draw the Label.
  ctx.font = ChartUtils.getFont(xLabelFontSize, 'bold')
  ctx.textAlign = 'left'
  ctx.fillStyle = ChartUtils.color.theme1Light
  ctx.fillText(
    xText,
    box.x + pad.horz.edge,
    box.y + pad.vert.edge + xTextHeight
  )

  // Draw the text.
  const textCenterGraphYs = yTexts.map((yText, i) => {
    ctx.font = ChartUtils.getMonoFont(11)
    ctx.fillStyle = ChartUtils.color.gray[200]
    const textGraphY =
      box.y + box.height - pad.vert.edge - (textH + pad.vert.between) * i
    const labelX = box.x + pad.horz.edge + yLabelMeasureMaxWidth

    ctx.textAlign = 'right'
    ctx.fillText(yLabels[i], labelX, textGraphY)
    ctx.fillText(yText, box.x + box.width - pad.horz.edge, textGraphY)
    ctx.textAlign = 'left'

    ctx.font = ChartUtils.getMonoFont(7)
    ctx.fillText('th', labelX, textGraphY - 4)

    return textGraphY - textH / 2
  })

  return textCenterGraphYs
}
