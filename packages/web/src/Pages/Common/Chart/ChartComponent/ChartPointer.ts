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
  subHeading: string
  formatX: (dataX: number) => string
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

const _calculateBox = (
  labeledDataYsAtTarget: { dataY: number; label: string }[],
  dataXAtTarget: number,
  canvasContext: CanvasRenderingContext2D,
  { scale, viewport, plotArea }: ChartStateDerived,
  { subHeading, formatX, formatY, showTh, pad }: ChartPointerOpts,
) => {
  const getTextInfo = (
    text: string,
    font: string,
    color: string,
    textAlign: 'left' | 'right',
  ) => {
    return {
      ..._measureText(canvasContext, text, font),
      draw: (x: number, y: number) => {
        canvasContext.textBaseline = 'top'
        canvasContext.textAlign = textAlign
        canvasContext.font = font
        canvasContext.fillStyle = color
        canvasContext.fillText(text, x, y)
      },
    }
  }
  const defaultColor = ChartUtils.color.gray[200]
  const lineInfos = {
    heading: getTextInfo(
      formatX(dataXAtTarget),
      ChartUtils.getMonoFont(13, 'bold'),
      ChartUtils.color.teal[500],
      'left',
    ),
    subHeading: getTextInfo(
      subHeading,
      ChartUtils.getMonoFont(12),
      defaultColor,
      'left',
    ),
    dataLines: (() => {
      const th = showTh
        ? getTextInfo('th', ChartUtils.getMonoFont(7), defaultColor, 'left')
        : null
      return labeledDataYsAtTarget.map(({ dataY, label }, i) => {
        const font = ChartUtils.getMonoFont(11)
        return {
          label: getTextInfo(label, font, defaultColor, 'right'),
          th,
          dataY: getTextInfo(formatY(dataY), font, defaultColor, 'right'),
        }
      })
    })(),
  }

  const headingRelativePosition = { x: pad.horz.edge, y: pad.vert.top }

  let labelRelativePixelY =
    headingRelativePosition.y +
    lineInfos.heading.height +
    pad.vert.between * .75 +
    lineInfos.subHeading.height +
    pad.vert.between +
    pad.vert.between 

  const labelRelativePixelYs = lineInfos.dataLines.map(({ label }) => {
    const result = labelRelativePixelY
    labelRelativePixelY += label.height + pad.vert.between
    return result
  })
  const height = labelRelativePixelY - pad.vert.between + pad.vert.bottom

  const labelsMaxWidth = Math.max(
    ...lineInfos.dataLines.map((x) => x.label.width),
  )
  const labelRight = pad.horz.edge + labelsMaxWidth

  const dataYsMaxWidth = Math.max(
    ...lineInfos.dataLines.map((x) => x.dataY.width),
  )

  const width = Math.max(
    labelRight +
      (lineInfos.dataLines[0].th?.width ?? 0) +
      pad.horz.between +
      dataYsMaxWidth +
      pad.horz.edge,
    pad.horz.edge * 2 + lineInfos.heading.width,
    pad.horz.edge * 2 + lineInfos.subHeading.width,
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
          y + labelRelativePixelYs[i] + lineInfos.dataLines[i].label.height / 2
        canvasContext.beginPath()

        const line = [
          { x: pixelX, y: pixelY },
          { x: pixelX + (xLineTarget - pixelX) * 0.6, y: graphYOnBox },
          { x: xLineTarget, y: graphYOnBox },
        ]
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

    let currY = y + headingRelativePosition.y

    // Draw the heading.
    lineInfos.heading.draw(x + headingRelativePosition.x, currY)
    currY += lineInfos.heading.height + pad.vert.between * .75

    // Draw the subHeading.
    lineInfos.subHeading.draw(x + headingRelativePosition.x, currY)
    currY += lineInfos.subHeading.height + pad.vert.between
    canvasContext.beginPath()
    canvasContext.moveTo(x + pad.horz.edge, currY)
    canvasContext.lineTo(regionExt.right - pad.horz.edge , currY)
    canvasContext.lineCap = 'round'
    canvasContext.lineWidth = .5
    canvasContext.strokeStyle = ChartUtils.color.gray[400]
    canvasContext.stroke()
    currY += pad.vert.between 
    
    // canvasContext.beginPath()
    // ChartUtils.roundRect(
    //   canvasContext,
    //   rectExt({ x, y: currY, width: region.width, height: region.height }),
    //   0,
    //   )
    //   canvasContext.fillStyle=ChartUtils.color.gray[500]
    // canvasContext.fill()

    // Draw the text lines.
    lineInfos.dataLines.forEach(({ label, th, dataY }, i) => {
      label.draw(x + labelRight, currY)
      th?.draw(x + labelRight, currY - 2)
      dataY.draw(right - pad.horz.edge, currY)
      currY += label.height + pad.vert.between
    })
  }

  return { drawBox, targetBoxInfo }
}

const _measureText = (
  canvasContext: CanvasRenderingContext2D,
  text: string,
  font: string,
) => {
  canvasContext.font = font
  const { width, actualBoundingBoxAscent, actualBoundingBoxDescent } =
    canvasContext.measureText(text)
  return {
    width,
    height: actualBoundingBoxAscent + actualBoundingBoxDescent,
  }
}
