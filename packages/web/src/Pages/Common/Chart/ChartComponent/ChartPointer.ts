import {gsap} from 'gsap'
import _ from 'lodash'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {assert} from '../../../../Utils/Utils'
import {ChartFullState} from '../Chart'
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

export type ChartPointerState = {x: number}
export type ChartPointerContext<Data> = ChartContext<Data> & {
  pointerTransition: ChartDataTransition<ChartPointerState>
}
export interface ChartPointerComponent<Data> {
  setState: (
    ctx: CanvasRenderingContext2D,
    state: ChartFullState<Data> & {pointerState: ChartPointerState},
    pointerTransition: number,
    registerAnimation: ChartRegisterAnimation
  ) => void
  draw: (
    context: ChartPointerContext<Data>,
    registerAnimation: ChartRegisterAnimation
  ) => void
  destroy: () => void
}

export class ChartPointer<Data> implements ChartComponent<Data> {
  private _state: {
    boxInfo: _BoxInfo
    transition: ChartDataTransition<{
      positionX: number
      boxW: number
      boxY: number
      boxXOffset: number
      boxSide: number
    }>
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

  setPointerPosition(
    position: {x: number} | null,
    state: ChartFullState<Data>,
    ctx: CanvasRenderingContext2D,
    registerAnimation: ChartRegisterAnimation,
    inside = position !== null
  ) {
    if (position) {
      const positionX = state.scale.x(
        _.clamp(
          Math.round(state.scale.x.inverse(position.x)),
          Math.ceil(state.xyRange.x.start),
          Math.floor(state.xyRange.x.end)
        )
      )

      const {boxInfo, boxY, boxW, boxXOffset, boxSide} = _boxInfo(
        state,
        positionX,
        this.dataFn,
        this.formatY,
        this.formatX,
        ctx
      )
      const target = {positionX, boxY, boxW, boxXOffset, boxSide}
      const prev = this._state
        ? chartDataTransitionCurrObj(this._state.transition, x => x)
        : {positionX: position.x, boxY, boxW, boxXOffset, boxSide}

      for (const component of this._components) {
        ctx.save()
        component.setState(
          ctx,
          {
            ...state,
            pointerState: {x: target.positionX},
          },
          this._state?.transition.transition ?? 0,
          registerAnimation
        )
        ctx.restore()
      }
      this._state = {
        boxInfo,
        transition: {target, prev, transition: 0},
      }

      this._animation?.kill()
      this._animation = registerAnimation(
        gsap.to(this._state.transition, {
          transition: 1,
          duration,
          ease: 'power4',
        })
      )
    }
  }

  setState(
    state: ChartFullState<Data>,
    context: CanvasRenderingContext2D,
    registerAnimation: ChartRegisterAnimation
  ) {
    if (!this._state) return
    this.setPointerPosition(
      {x: this._state.transition.target.positionX},
      state,
      context,
      registerAnimation
    )
  }

  draw(
    chartContext: ChartContext<Data>,
    registerAnimation: ChartRegisterAnimation
  ) {
    const {ctx, stateTransition, currState} = chartContext
    const {scale, plotArea} = currState
    if (!this._state) {
      this.setPointerPosition(
        {x: plotArea.x + plotArea.width * 0.25},
        stateTransition.target,
        ctx,
        registerAnimation,
        false
      )
    }
    assert(this._state)

    const {
      positionX: graphX,
      boxSide,
      boxY,
      boxW,
      boxXOffset,
    } = chartDataTransitionCurrObj(this._state.transition, x => x)
    ctx.restore()
    const pointerTransition = {
      prev: {x: this._state.transition.prev.positionX},
      target: {x: this._state.transition.target.positionX},
      transition: this._state.transition.transition,
    }
    this._components.forEach(x => {
      ctx.save()
      x.draw({...chartContext, pointerTransition}, registerAnimation)
      ctx.restore()
    })

    const dataX = scale.x.inverse(graphX)
    const dataYs = chartDataTransitionCurrNumArr(stateTransition, x =>
      this.dataFn(x.data).map(({line}) => line(dataX))
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
      this._state.boxInfo,
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

type _BoxInfo = ReturnType<typeof _boxInfo>['boxInfo']
const _boxInfo = <Data>(
  state: ChartFullState<Data>,
  graphX: number,
  dataFn: _DataFn<Data>,
  formatY: (x: number) => string,
  formatX: (data: Data, x: number) => string,
  ctx: CanvasRenderingContext2D
) => {
  const {plotArea, viewport, scale, data} = state
  const dataX = Math.round(scale.x.inverse(graphX))
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
    plotArea.y,
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
    (viewport.width < 200 ? 0.3 : viewport.width < 500 ? 0.5 : 1)
  const boxPad = viewport.width > 500 ? 5 : -10
  const boxSide = graphX + boxXOffset + boxW + boxPad < plotArea.right ? 1 : -1
  boxXOffset = Math.min(
    boxXOffset,
    boxSide === 1
      ? viewport.right - 1 - graphX - boxW
      : graphX - (viewport.x + 1) - boxW
  )

  return {
    boxInfo: {
      boxH,
      yTexts,
      textH,
      yLabelMeasureMaxWidth,
      yLabels,
      xText,
      xTextHeight,
    },
    boxY,
    boxW,
    boxXOffset,
    boxSide,
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
  }: _BoxInfo,
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
