import { Size } from '../../../Utils/Geometry'
import {ChartState, ChartStateDerived} from './Chart'
import {ChartDataTransition} from './ChartUtils/ChartDataTransition'

export type ChartContext<Data> = {
  chartSize: Size
  pointer: {x: number; y: number}
  dataTransition: ChartDataTransition<Data>
  stateTransition: ChartDataTransition<ChartState>
  currState: ChartState
  derivedState: {
    prev: ChartStateDerived
    target: ChartStateDerived
    curr: ChartStateDerived
  }
  canvasContext: CanvasRenderingContext2D
}
