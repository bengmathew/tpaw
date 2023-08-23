import { fGet, linearFnFomPoints } from '@tpaw/common'
import _ from 'lodash'
import { Padding, rectExt } from '../../../../Utils/Geometry'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { ChartContext } from '../ChartContext'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent } from './ChartComponent'

export type ChartXAxisTickType = 'large' | 'medium' | 'small' | 'none'
export type ChartXAxisTickStyle = {
  length: number
  gap: number
  color: string
  font: string
}

export type ChartXAxisOpts = {
  type: (transformedDataX: number) => ChartXAxisTickType
  tickStyle: (type: Exclude<ChartXAxisTickType, 'none'>) => ChartXAxisTickStyle
  padding: Padding
  style: {
    background: { retired: string; notRetired: string }
  }
  dataXTransform: (dataX: number) => number
  formatLabel: (transformedDataX: number) => string
  pointerStyle: {
    color: {
      fill: string
    }
    height: number
  }
  yOffset: number
  shouldLabel: (
    pixelsPerTick: number,
    type: Exclude<ChartXAxisTickType, 'none'>,
  ) => boolean
  visible: boolean
  maxDataX: number
  retirementDataX: number
  label: string
  height: number
  hidePointer: boolean
  labelStyle: {
    font: string
    color: string
    padding: { left: number; bottom: number }
  }
}

export class ChartXAxis<Data> implements ChartComponent<Data> {
  constructor(
    private readonly _opts: (
      chartContext: ChartContext<Data>,
    ) => ChartXAxisOpts,
  ) {}

  public draw(chartContext: ChartContext<Data>) {
    const { canvasContext, derivedState, currPointerInDataCoord } = chartContext
    const opts = this._opts(chartContext)
    const {
      yOffset,
      pointerStyle,
      visible,
      maxDataX,
      padding,
      retirementDataX,
      style,
      hidePointer,
      label,
      height,
      type: getType,
      labelStyle,
    } = opts
    if (!visible) return
    const { scale, plotArea, viewport } = derivedState.curr
    const pixelRegion = rectExt({
      x: plotArea.x,
      y: plotArea.bottom + yOffset,
      right:
        Math.min(plotArea.right, scale.x(maxDataX)) +
        padding.left +
        padding.right,
      height,
    })
    // Extend plotArea by 1 to help with floating point math.
    const dataXs = _.range(
      Math.ceil(scale.x.inverse(plotArea.x - 1)),
      Math.min(Math.floor(scale.x.inverse(plotArea.right + 1)), maxDataX) + 1, // +1 because range is not inclusive.
    )
    const retirementPixelX = scale.x(retirementDataX)

    // ---- BACKGROUND ----
    {
      canvasContext.save()
      canvasContext.beginPath()
      ChartUtils.roundRect(canvasContext, pixelRegion, 5)
      canvasContext.clip()

      canvasContext.fillStyle = style.background.notRetired
      canvasContext.fillRect(
        pixelRegion.x,
        pixelRegion.y,
        pixelRegion.width,
        pixelRegion.height,
      )

      canvasContext.fillStyle = style.background.retired
      canvasContext.fillRect(
        Math.max(pixelRegion.x, retirementPixelX),
        pixelRegion.y,
        pixelRegion.width,
        pixelRegion.height,
      )

      canvasContext.restore()
    }

    // ---- LABEL ----
    canvasContext.font = labelStyle.font
    const labelPixelX = plotArea.x + labelStyle.padding.left
    const labelWidth = canvasContext.measureText(label).width
    {
      canvasContext.fillStyle = labelStyle.color
      canvasContext.textBaseline = 'bottom'
      canvasContext.fillText(
        label,
        labelPixelX,
        pixelRegion.bottom - labelStyle.padding.bottom,
      )
    }

    // ---- POINTER  ----
    if (!hidePointer) {
      canvasContext.save()
      // const pointerPixelS = tickStyle('large').length + padding.top
      const pixelX = scale.x(currPointerInDataCoord.x)
      const { height, color } = pointerStyle
      const overshotPixelX =
        scale.x(currPointerInDataCoord.x) - scale.x(maxDataX)
      canvasContext.globalAlpha = _.clamp(
        linearFnFomPoints(0, 1, 15, 0)(overshotPixelX),
        0,
        1,
      )
      canvasContext.beginPath()
      ChartUtils.roundedPolygon(
        canvasContext,
        [
          { x: pixelX, y: pixelRegion.y },
          { x: pixelX + height, y: pixelRegion.y + height },
          { x: pixelX - height, y: pixelRegion.y + height },
        ],
        2,
      )
      canvasContext.fillStyle = color.fill
      canvasContext.fill()

      canvasContext.restore()
    }

    // ---- TICKS ----
    {
      const pixelRight = scale.x(fGet(_.last(dataXs)))
      const noLabelPixelZones = [
        { start: plotArea.x, end: plotArea.x + labelPixelX + labelWidth + 5 },
        { start: pixelRight - 25, end: pixelRight },
      ]
      dataXs.forEach((dataX) => {
        _drawTick(
          dataX,
          noLabelPixelZones,
          null, // special
          chartContext,
          opts,
        )
      })
    }
    // ---- MIN/MAX ----
    {
      const _draw = (dataX: number, asDataX?: number) =>
        _drawTick(
          dataX,
          [],
          { forceLabel: true, asDataX: asDataX ?? null },
          chartContext,
          opts,
        )

      const findFirstTick = (dataX: number): number =>
        getType(opts.dataXTransform(dataX)) !== 'none'
          ? dataX
          : findFirstTick(dataX - 1)
      let firstPhantomTick = findFirstTick(dataXs[0])
      _draw(dataXs[0], firstPhantomTick)

      const lastVisibleTick = _.reverse([...dataXs]).find(
        (x) => getType(opts.dataXTransform(x)) !== 'none',
      )
      if (lastVisibleTick !== undefined) _draw(lastVisibleTick)
    }
  }
}

function _drawTick<Data>(
  dataX: number,
  noLabelPixelZones: SimpleRange[],
  special: { forceLabel: boolean; asDataX: number | null } | null,
  { derivedState, canvasContext }: ChartContext<Data>,
  {
    dataXTransform,
    formatLabel,
    tickStyle,
    type: getType,
    yOffset,
    shouldLabel,
    maxDataX,
    padding,
  }: ChartXAxisOpts,
) {
  if (dataX > maxDataX) return
  const transformedDataX = dataXTransform(special?.asDataX ?? dataX)
  const { scale, plotArea } = derivedState.curr
  const label = formatLabel(transformedDataX)
  const type = getType(transformedDataX)
  if (type === 'none') return
  const { length, color, font, gap } = tickStyle(type)
  const pixelY = plotArea.bottom + yOffset + padding.top
  const pixelX = scale.x(dataX)
  const labelPixelY = pixelY + length + gap

  // Draw the tick line.
  if (!special || special.asDataX === null || special.asDataX === dataX) {
    canvasContext.save()
    canvasContext.beginPath()
    canvasContext.moveTo(pixelX, pixelY)
    canvasContext.lineTo(pixelX, pixelY + length)
    canvasContext.strokeStyle = color
    canvasContext.lineWidth = 1
    canvasContext.stroke()
  }

  // Draw the label.
  const targetScale = derivedState.target.scale
  const pixelsPerTickAtTarget = targetScale.x(1) - targetScale.x(0)
  if (
    special?.forceLabel ||
    (shouldLabel(pixelsPerTickAtTarget, type) &&
      noLabelPixelZones.every(
        ({ start, end }) => pixelX < start || pixelX > end,
      ))
  ) {
    canvasContext.font = font
    const width = canvasContext.measureText(label).width
    const adjustedPixelX = _.clamp(
      pixelX,
      plotArea.x + width / 2,
      plotArea.right - width / 2,
    )
    canvasContext.textAlign = 'center'
    canvasContext.textBaseline = 'top'
    canvasContext.fillStyle = color
    canvasContext.fillText(label, adjustedPixelX, labelPixelY)
  }
  canvasContext.restore()
}
