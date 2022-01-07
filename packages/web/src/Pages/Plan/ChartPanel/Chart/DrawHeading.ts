import { ChartContext } from './ChartContext'
import { ChartUtils } from './ChartUtils'

export function drawHeading(context: ChartContext) {
  const {ctx, tpawResult, scale} = context
  ctx.font = ChartUtils.getFont(20, 'bold')
  ctx.fillStyle = 'black'
  ctx.fillText(`Spending During Retirement`, 20, 30)

  ctx.font = ChartUtils.getFont(14)
  ctx.fillStyle = ChartUtils.color.gray['400']
  ctx.fillText(
    `Results from simulating your retirement ${tpawResult.src.args.numRuns} times.`,
    20,
    50
  )
}
