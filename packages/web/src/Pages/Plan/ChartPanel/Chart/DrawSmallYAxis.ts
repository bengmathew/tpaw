import {formatCurrency} from '../../../../Utils/FormatCurrency'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {ChartContext} from './ChartContext'
import {ChartUtils} from './ChartUtils'

export function drawSmallYAxis(context: ChartContext) {
  const {ctx, tpawResult, percentileInfo, plotArea, scale} = context

  const byAge = (age: number) => {
    const x = scale.curr.x(age)
    const minData = linearFnFomPoints(
      0,
      tpawResult.prevData[0](age),
      1,
      tpawResult.data[0](age)
    )(tpawResult.dataTransition)
    const minY = scale.curr.y(minData)
    const n = tpawResult.data.length
    const maxData = linearFnFomPoints(
      0,
      tpawResult.prevData[n - 1](age),
      1,
      tpawResult.data[n - 1](age)
    )(tpawResult.dataTransition)

    const maxY = scale.curr.y(maxData)
    return {
      x,
      minDataStr: `${formatCurrency(minData)}`,
      maxDataStr: `${formatCurrency(maxData)}`,
      minY,
      maxY,
    }
  }

  ctx.save()

  ctx.globalAlpha = 1
  ctx.fillStyle = ChartUtils.color.gray[800]
  ctx.font = ChartUtils.getMonoFont(11)
  const measure = ctx.measureText('$1')

  const drawPair = ({
    x,
    minDataStr,
    maxDataStr,
    minY,
    maxY,
  }: ReturnType<typeof byAge>) => {
    if (minDataStr !== maxDataStr) {
      const adjustedMinY = Math.min(
        plotArea.y + plotArea.height - 4,
        minY + measure.actualBoundingBoxAscent + 4
      )
      ctx.fillText(minDataStr, x, adjustedMinY)

      const adjustedMaxY = Math.min(
        maxY - 8,
        adjustedMinY - measure.actualBoundingBoxAscent - 4
      )
      ctx.fillText(maxDataStr, x, adjustedMaxY)
    } else {
      ctx.fillText(maxDataStr, x, maxY - 8)
    }
  }
  ctx.textAlign = 'left'
  drawPair(byAge(tpawResult.age.start))
  ctx.textAlign = 'right'
  drawPair(byAge(tpawResult.age.end))

  ctx.globalAlpha = 1

  ctx.restore()
}
