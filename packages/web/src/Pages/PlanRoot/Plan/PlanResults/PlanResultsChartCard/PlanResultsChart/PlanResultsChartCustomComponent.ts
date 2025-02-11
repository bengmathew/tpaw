import { rectExt } from '../../../../../../Utils/Geometry'
import { ChartComponent } from '../../../../../Common/Chart/ChartComponent/ChartComponent'
import { ChartContext } from '../../../../../Common/Chart/ChartContext'
import { PlanResultsChartDataForPDF } from './PlanResultsChartData'

export class PlanResultsChartCustomComponent
  implements
    ChartComponent<{
      data: PlanResultsChartDataForPDF
    }>
{
  constructor(private opts: { gapToLine: number; lineWidth: number }) {}

  draw({
    canvasContext: ctx,
    currState,
    stateTransition,
  }: ChartContext<{
    data: PlanResultsChartDataForPDF
  }>) {
    const { plotArea, viewport } = currState.derivedState
    const { planColors } = stateTransition.target.params.data

    const region = rectExt({
      x: plotArea.x,
      y: plotArea.bottom + this.opts.gapToLine + this.opts.lineWidth / 2,
      width: plotArea.width,
      height: viewport.bottom,
    })

    ctx.beginPath()
    ctx.moveTo(region.x, region.y)
    ctx.lineTo(region.right, region.y)
    ctx.strokeStyle = planColors.shades.main[10].hex
    ctx.lineWidth = this.opts.lineWidth 
    ctx.stroke()
    return null
  }
}
