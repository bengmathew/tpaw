import _ from 'lodash'
import { RGB } from '../../../../Utils/ColorUtils'
import { ChartContext } from '../ChartContext'
import { ChartUtils } from '../ChartUtils/ChartUtils'
import { ChartComponent } from './ChartComponent'

export type ChartYAxisProps = {
  format: (dataY: number) => string
  gridInfo: ReturnType<typeof getYAxisGridInfo>
  style: {
    colors: { text: RGB; line: RGB }
  }
}

export class ChartYAxis<Params> implements ChartComponent<Params> {
  constructor(
    private _propsFn: (
      state: ChartContext<Params>['stateTransition']['from'],
    ) => ChartYAxisProps,
  ) {}

  public draw(context: ChartContext<Params>) {
    const { canvasContext: ctx, stateTransition } = context
    const { derivedState } = stateTransition.target
    const { scale, plotArea } = derivedState
    const { format, gridInfo, style } = this._propsFn(stateTransition.target)
    const { gridDataSize, numLines } = gridInfo

    ctx.font = ChartUtils.getMonoFont(11)
    ctx.textAlign = 'right'

    _.range(0, numLines)
      .map((i) => i * gridDataSize)
      .forEach((gridDataY, i) => {
        const gridGraphY = scale.y(gridDataY)
        if (gridGraphY < plotArea.y) return
        if (i !== 0) {
          ctx.strokeStyle = RGB.toHex(style.colors.line)
          ctx.beginPath()
          ctx.moveTo(plotArea.x, gridGraphY)
          ctx.lineTo(plotArea.right, gridGraphY)
          ctx.stroke()
        }
        ctx.fillStyle = RGB.toHex(style.colors.text)
        ctx.textBaseline = i === 0 ? 'middle' : 'middle'
        ctx.fillText(format(gridDataY), plotArea.x - 10, gridGraphY)
      })
    return null
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
