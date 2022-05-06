import {ChartSizing, ChartState, ChartStateDerived} from './Chart'
import {ChartDataTransition} from './ChartUtils/ChartDataTransition'

export type ChartContext<Data> = {
  canvasContext: CanvasRenderingContext2D

  // Current chart state.
  sizing: ChartSizing
  pointer: {x: number; y: number}
  dataTransition: ChartDataTransition<Data>
  stateTransition: ChartDataTransition<ChartState>

  // Computed chart state.
  currState: ChartState
  derivedState: {
    prev: ChartStateDerived
    target: ChartStateDerived
    curr: ChartStateDerived
  }
}
