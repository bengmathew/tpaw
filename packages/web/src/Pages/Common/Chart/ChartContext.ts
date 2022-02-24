import {ChartState} from './Chart'
import {ChartDataTransition} from './ChartUtils/ChartDataTransition'

export type ChartContext<Data> = {
  ctx: CanvasRenderingContext2D
  stateTransition: ChartDataTransition<ChartState<Data>>
  currState: Omit<ChartState<Data>, 'data'>
}
