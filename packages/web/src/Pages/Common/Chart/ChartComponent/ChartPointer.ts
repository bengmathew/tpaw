import { assert, fGet, noCase } from '@tpaw/common'
import _ from 'lodash'
import { Rect, rectExt } from '../../../../Utils/Geometry'
import { interpolate } from '../../../../Utils/Interpolate'
import { transitionTransform } from '../../../../Utils/Transition'
import { ChartStateDerived } from '../Chart'
import { ChartContext } from '../ChartContext'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent } from './ChartComponent'

export type ChartPointerOpts = {
  formatX: (dataX: number) => { text: string; color: string | null }[]
  formatY: (dataY: number) => string
  showTh: boolean
  pad: {
    vert: { top: number; between: number; bottom: number }
    horz: {
      edge: number
      between: number
      outside: { lineLength: number; margin: number }
    }
  }
}
const verticalLineDash = [10, 5]
type _DataFn<Data> = (
  data: Data,
) => ({ line: (x: number) => number; label: string } | null)[]

export class ChartPointer<Data> implements ChartComponent<Data> {
  private _boxInfoTransition: {
    from: { region: Rect; xLineTarget: number }
    curr: { region: Rect; xLineTarget: number }
  } | null = null
  constructor(
    private readonly _dataFn: _DataFn<Data>,
    private readonly _opts: (
      chartContext: ChartContext<Data>,
    ) => ChartPointerOpts,
  ) {}

  public update(change: 'init' | 'pointer' | 'stateAndPointer' | 'sizing') {
    this._boxInfoTransition = (() => {
      switch (change) {
        case 'pointer':
        case 'stateAndPointer':
          const curr = fGet(this._boxInfoTransition).curr
          return { from: curr, curr: curr }
        case 'init':
        case 'sizing':
          return this._boxInfoTransition
        default:
          noCase(change)
      }
    })()
  }

  public draw(chartContext: ChartContext<Data>) {
    const {
      canvasContext,
      currPointerInDataCoord,
      derivedState,
      dataTransition,
      pointerInDataCoordTransition,
    } = chartContext
    const { scale, plotArea } = derivedState.curr
    const opts = this._opts(chartContext)

    const filteredDataTransition = (() => {
      const unfiltered = transitionTransform(dataTransition, this._dataFn)
      const selector = unfiltered.target.map((x) => x !== null)
      return transitionTransform(unfiltered, (x) =>
        x.filter((_, i) => selector[i]),
      )
    })()

    const currDataYs = _.reverse(
      filteredDataTransition.target.map((target, i) => {
        assert(target)
        const from = filteredDataTransition.from[i]
        if (from === null) return target.line(currPointerInDataCoord.x)
        return interpolate(
          from.line(currPointerInDataCoord.x),
          target.line(currPointerInDataCoord.x),
          filteredDataTransition.progress,
        )
      }),
    )

    canvasContext.save()
    const { drawBox, targetBoxInfo } = _calculateBox(
      _.reverse(
        filteredDataTransition.target
          .map((x) => fGet(x))
          .map((x) => ({
            dataY: x.line(pointerInDataCoordTransition.target.x),
            label: x.label,
          })),
      ),
      pointerInDataCoordTransition.target.x,
      canvasContext,
      derivedState.curr,
      opts,
    )
    canvasContext.restore()
    if (!this._boxInfoTransition) {
      this._boxInfoTransition = { from: targetBoxInfo, curr: targetBoxInfo }
    }
    this._boxInfoTransition.curr = interpolate(
      this._boxInfoTransition.from,
      targetBoxInfo,
      pointerInDataCoordTransition.progress,
    )

    const pixelX = scale.x(currPointerInDataCoord.x)
    const pixelYs = currDataYs.map(scale.y)

    canvasContext.save()
    drawBox(this._boxInfoTransition.curr, pixelX, pixelYs)
    canvasContext.restore()

    // Draw the vertical line.
    canvasContext.globalAlpha = 0.3
    canvasContext.lineWidth = 1.5
    canvasContext.strokeStyle = ChartUtils.color.gray[500]
    canvasContext.setLineDash(verticalLineDash)
    canvasContext.lineDashOffset = verticalLineDash[0]
    canvasContext.beginPath()
    canvasContext.moveTo(pixelX, plotArea.bottom) // +20 is a hack to show line when person1 xaxis is cut short.
    canvasContext.lineTo(pixelX, Math.min(...pixelYs))
    canvasContext.stroke()
    canvasContext.setLineDash([])

    // Draw the target on the data line.
    canvasContext.globalAlpha = 0.7
    canvasContext.lineWidth = 2
    canvasContext.fillStyle = ChartUtils.color.gray[900]
    pixelYs.forEach((pixelY) => {
      canvasContext.beginPath()
      canvasContext.ellipse(pixelX, pixelY, 4, 4, 0, 0, Math.PI * 4)
      canvasContext.fill()
    })
  }
}

const _measureText =
  (canvasContext: CanvasRenderingContext2D) =>
  (text: string, fontSize: number, style?: 'bold' | '') => {
    const font = ChartUtils.getMonoFont(fontSize, style)
    canvasContext.font = font
    const { width, actualBoundingBoxAscent, actualBoundingBoxDescent } =
      canvasContext.measureText(text)
    return {
      text,
      font,
      width,
      height: actualBoundingBoxAscent + actualBoundingBoxDescent,
    }
  }

const _calculateBox = (
  labeledDataYsAtTarget: { dataY: number; label: string }[],
  dataXAtTarget: number,
  canvasContext: CanvasRenderingContext2D,
  { scale, viewport, plotArea }: ChartStateDerived,
  { formatX, formatY, showTh, pad }: ChartPointerOpts,
) => {
  const headerFormatted = formatX(dataXAtTarget)
  const mt = _measureText(canvasContext)
  const textInfos = {
    header: headerFormatted.map(({ text }) => mt(text, 13, 'bold')),
    lines: labeledDataYsAtTarget.map(({ dataY, label }, i) => ({
      label: mt(label, 11),
      dataY: mt(formatY(dataY), 11),
    })),
    th: showTh ? mt('th', 7) : null,
  }

  const headerRelativePosition = { x: pad.horz.edge, y: pad.vert.top }

  let labelRelativePixelY =
    headerRelativePosition.y +
    Math.max(...textInfos.header.map((x) => x.height)) +
    pad.vert.between

  const labelRelativePixelYs = textInfos.lines.map(({ label }) => {
    const result = labelRelativePixelY
    labelRelativePixelY += label.height + pad.vert.between
    return result
  })
  const height = labelRelativePixelY - pad.vert.between + pad.vert.bottom

  const labelsMaxWidth = Math.max(...textInfos.lines.map((x) => x.label.width))
  const labelRight = pad.horz.edge + labelsMaxWidth

  const dataYsMaxWidth = Math.max(...textInfos.lines.map((x) => x.dataY.width))

  const width = Math.max(
    labelRight +
      (textInfos.th?.width ?? 0) +
      pad.horz.between +
      dataYsMaxWidth +
      pad.horz.edge,
    pad.horz.edge * 2 + _.sum(textInfos.header.map((x) => x.width)),
  )

  const pixelYsAtTarget = labeledDataYsAtTarget.map(({ dataY }) =>
    scale.y(dataY),
  )
  const pixelXAtTarget = scale.x(dataXAtTarget)

  const y = _.clamp(
    _.mean(pixelYsAtTarget) - height * 0.45,
    viewport.y,
    plotArea.y + plotArea.height - height - 10,
  )

  const { lineLength } = pad.horz.outside
  const side =
    pixelXAtTarget + lineLength + width + pad.horz.outside.margin <
    viewport.right
      ? 1
      : -1
  const x = pixelXAtTarget + (side === -1 ? -lineLength - width : lineLength)
  const targetXLineTarget = side === -1 ? x + width : x

  const targetBoxRegion = rectExt({ x, y, height, width })
  const targetBoxInfo = {
    region: targetBoxRegion,
    xLineTarget: targetXLineTarget,
  }
  const drawBox = (
    {
      region,
      xLineTarget,
    }: {
      region: Rect
      xLineTarget: number
    },
    pixelX: number,
    pixelYs: number[],
  ) => {
    const regionExt = rectExt(region)
    const { x, y, right } = regionExt

    canvasContext.textBaseline = 'top'

    // Draw the lines.
    canvasContext.globalAlpha = 1
    canvasContext.lineWidth = 1
    canvasContext.strokeStyle = ChartUtils.color.gray[700]

    // Math.max, because it it goes to far out, the lines won't draw.
    pixelYs
      .map((x) => Math.max(x, viewport.x - 10000000)) 
      .forEach((pixelY, i) => {
        const graphYOnBox =
          y + labelRelativePixelYs[i] + textInfos.lines[i].label.height / 2
        canvasContext.beginPath()

        const line = [
          { x: pixelX, y: pixelY },
          { x: pixelX + (xLineTarget - pixelX) * 0.6, y: graphYOnBox },
          { x: xLineTarget, y: graphYOnBox },
        ]
        console.dir(line)
        ChartUtils.roundedLine(canvasContext, line, 10)
        canvasContext.stroke()
      })

    // Clip the box.
    canvasContext.beginPath()
    ChartUtils.roundRect(canvasContext, regionExt, 10)
    canvasContext.clip()

    // Draw the box.
    canvasContext.beginPath()
    ChartUtils.roundRect(canvasContext, regionExt, 10)
    canvasContext.fillStyle = ChartUtils.color.gray[700]
    canvasContext.fill()

    // Draw the header.
    let headerPixelX = x + headerRelativePosition.x
    textInfos.header.forEach(({ font, width, text }, i) => {
      canvasContext.font = font
      canvasContext.textAlign = 'left'
      canvasContext.fillStyle =
        headerFormatted[i].color ?? ChartUtils.color.gray[200]
      canvasContext.fillText(text, headerPixelX, y + headerRelativePosition.y)
      headerPixelX += width
    })

    // Draw the text lines.
    canvasContext.fillStyle = ChartUtils.color.gray[200]
    textInfos.lines.forEach(({ label, dataY }, i) => {
      const currY = y + labelRelativePixelYs[i]

      // Draw the label.
      canvasContext.font = label.font
      canvasContext.textAlign = 'right'
      canvasContext.fillText(label.text, x + labelRight, currY)

      // Draw the th.]
      if (textInfos.th) {
        canvasContext.font = textInfos.th.font
        canvasContext.textAlign = 'left'
        canvasContext.fillText(textInfos.th.text, x + labelRight, currY - 2)
      }
      // Draw the dataY.
      canvasContext.font = dataY.font
      canvasContext.textAlign = 'right'
      canvasContext.fillText(dataY.text, right - pad.horz.edge, currY)
    })
  }

  return { drawBox, targetBoxInfo }
}
