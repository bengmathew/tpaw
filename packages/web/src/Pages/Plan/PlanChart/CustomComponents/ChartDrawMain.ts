import { rectExt } from '../../../../Utils/Geometry'
import { ChartComponent } from '../../../Common/Chart/ChartComponent/ChartComponent'
import { ChartContext } from '../../../Common/Chart/ChartContext'
import { ChartUtils } from '../../../Common/Chart/ChartUtils/ChartUtils'
import { TPAWChartDataMain } from '../TPAWChart/TPAWChartDataMain'

export class ChartDrawMain implements ChartComponent<TPAWChartDataMain> {
  constructor() {}

  draw({ canvasContext, derivedState }: ChartContext<TPAWChartDataMain>) {
    const { plotArea, viewport } = derivedState.curr

    const lineWidth = 4
    const region = rectExt({
      x: viewport.x,
      y:
        plotArea.bottom +
        2 + // gap at bottom of plotArea
        lineWidth / 2,
      width: viewport.width,
      height: viewport.bottom,
    })

    canvasContext.beginPath()
    canvasContext.moveTo(region.x, region.y)
    canvasContext.lineTo(region.right, region.y)
    canvasContext.strokeStyle = ChartUtils.color.gray[500]
    canvasContext.lineWidth = lineWidth
    canvasContext.stroke()

  }
}
