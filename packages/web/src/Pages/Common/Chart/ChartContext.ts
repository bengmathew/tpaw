import {ChartFullState} from './Chart'
import {ChartDataTransition} from './ChartUtils/ChartDataTransition'

export type ChartContext<Data> = {
  ctx: CanvasRenderingContext2D
  stateTransition: ChartDataTransition<ChartFullState<Data>>
  currState: Omit<ChartFullState<Data>, 'data'>
}
