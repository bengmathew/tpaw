import _ from 'lodash'
import { ChartContext } from '../ChartContext'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent } from './ChartComponent'

export class ChartYAxis<Data> implements ChartComponent<Data> {
  constructor(
    public format: (data: Data, x: number) => string,
    public getMaxDataY: (data: Data) => number,
    public colors: { text: string; line: string },
  ) {}

  public draw(context: ChartContext<Data>) {
    const { canvasContext: ctx, dataTransition, derivedState } = context
    const { scale, plotArea } = derivedState.curr

    const maxDataY = this.getMaxDataY(dataTransition.target)
    const { gridDataSize, numLines } = getYAxisGridSizeInfo(maxDataY)

    ctx.font = ChartUtils.getMonoFont(11)
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'

    _.range(1, numLines)
      .map((i) => i * gridDataSize)
      .forEach((gridDataY) => {
        const gridGraphY = scale.y(gridDataY)
        if (gridGraphY < plotArea.y) return
        ctx.strokeStyle = this.colors.line
        ctx.beginPath()
        ctx.moveTo(plotArea.x, gridGraphY)
        ctx.lineTo(plotArea.right, gridGraphY)
        ctx.stroke()
        ctx.fillStyle = this.colors.text
        ctx.fillText(
          this.format(dataTransition.target, gridDataY),
          plotArea.x - 10,
          gridGraphY,
        )
      })
  }
}

// Hack: displayMax, in combination with the gridDataSize, coincidentally
// maxes the percentage graphs to have the largest gridline not exceed 100%,
// which relieves us from having to engineer that explicitly. If changing
// these values, make sure to solve the maxing of percentage graphs.
export const getYAxisGridSizeInfo = (max: number) => {
  console.dir('------')
  console.dir(`max: ${max}`)
  const gridDataSize = (() => {
    const base = Math.pow(10, Math.floor(Math.log10(max)) + 1 - 1)
    const normMax = max / base
    const normGridSize = normMax < 1.5 ? 0.25 : normMax < 4 ? 0.5 : 1

    console.dir(`base: ${base}`)
    console.dir(`normMax: ${normMax}`)
    console.dir(`normGridSize: ${normGridSize}`)
    return normGridSize * base
  })()

  const displayMax = max * 1.2
  const numLines = Math.ceil(displayMax / gridDataSize) + 1
  console.dir(`displayMaX: ${displayMax}`)
  console.dir(`gridDataSize: ${gridDataSize}`)
  console.dir(`numLines: ${numLines}`)
  return { gridDataSize, numLines, displayMax }
}
