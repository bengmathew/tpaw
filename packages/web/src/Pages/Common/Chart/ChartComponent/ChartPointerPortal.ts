import { Rect } from '../../../../Utils/Geometry'
import { ChartContext } from '../ChartContext'
import { ChartComponent } from './ChartComponent'

export class ChartPointerPortal implements ChartComponent<null> {
  public data: { ctx: CanvasRenderingContext2D; viewport: Rect } | null = null
  draw({ canvasContext, currState }: ChartContext<null>) {
    this.data = {
      ctx: canvasContext,
      viewport: currState.derivedState.viewport,
    }
    return null
  }

  destroy() {
    this.data = null
  }
}
