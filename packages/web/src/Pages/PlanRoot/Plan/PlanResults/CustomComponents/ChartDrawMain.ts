import { rectExt } from '../../../../../Utils/Geometry'
import { ChartComponent } from '../../../../Common/Chart/ChartComponent/ChartComponent'
import { ChartContext } from '../../../../Common/Chart/ChartContext'
import { TPAWChartDataMain } from '../TPAWChart/TPAWChartDataMain'

export class ChartDrawMain implements ChartComponent<TPAWChartDataMain> {
  constructor(private opts: { xAxisLineColor: string; lineWidth: number }) {}

  draw({ canvasContext, derivedState }: ChartContext<TPAWChartDataMain>) {
    const { plotArea, viewport } = derivedState.curr

    const region = rectExt({
      x: viewport.x,
      y:
        plotArea.bottom +
        5 + // gap at bottom of plotArea
        this.opts.lineWidth / 2,
      width: viewport.width,
      height: viewport.bottom,
    })

    canvasContext.beginPath()
    canvasContext.moveTo(region.x, region.y)
    canvasContext.lineTo(region.right, region.y)
    canvasContext.strokeStyle = this.opts.xAxisLineColor
    canvasContext.lineWidth = this.opts.lineWidth
    canvasContext.stroke()
  }
}
