import _ from 'lodash'
import { ChartContext } from '../ChartContext'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent } from './ChartComponent'

export class ChartYAxis<Data> implements ChartComponent<Data> {
  constructor(
    public format: (data: Data, x: number) => string,
    public getGridInfo: (data: Data) => ReturnType<typeof getYAxisGridInfo>,
    public colors: { text: string; line: string },
  ) {}

  public draw(context: ChartContext<Data>) {
    const { canvasContext: ctx, dataTransition, derivedState } = context
    const { scale, plotArea } = derivedState.curr

    const { gridDataSize, numLines } = this.getGridInfo(dataTransition.target)

    ctx.font = ChartUtils.getMonoFont(11)
    ctx.textAlign = 'right'

    _.range(0, numLines)
      .map((i) => i * gridDataSize)
      .forEach((gridDataY, i) => {
        const gridGraphY = scale.y(gridDataY)
        if (gridGraphY < plotArea.y) return
        ctx.strokeStyle = this.colors.line
        ctx.beginPath()
        ctx.moveTo(plotArea.x, gridGraphY)
        ctx.lineTo(plotArea.right, gridGraphY)
        ctx.stroke()
        ctx.fillStyle = this.colors.text
        ctx.textBaseline = i === 0 ? 'bottom' : 'middle'
        ctx.fillText(
          this.format(dataTransition.target, gridDataY),
          plotArea.x - 10,
          gridGraphY,
        )
      })
  }
}

export const getYAxisGridInfo = ({
  max,
  maxGridLine,
}: {
  max: number
  maxGridLine: number | null
}) => {
  const gridDataSize = (() => {
    const base = Math.pow(10, Math.floor(Math.log10(max)) + 1 - 1)
    const normMax = max / base
    const normGridSize = normMax < 1.5 ? 0.25 : normMax < 4 ? 0.5 : 1
    return normGridSize * base
  })()

  const displayMax = max * 1.2
  const lines = _.range(0, Math.ceil(displayMax / gridDataSize) + 1)
    .map((x) => x * gridDataSize)
    .filter((x) => (maxGridLine === null ? true : x <= maxGridLine))
  const numLines = lines.length
  return { gridDataSize, numLines, displayMax }
}
