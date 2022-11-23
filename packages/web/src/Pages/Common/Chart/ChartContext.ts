import { XY } from '../../../Utils/Geometry'
import { ChartSizing, ChartState, ChartStateDerived } from './Chart'
import { Transition } from '../../../Utils/Transition'

export type ChartContext<Data> = {
  canvasContext: CanvasRenderingContext2D

  // Current chart state.
  sizing: ChartSizing

  dataTransition: Transition<Data>
  stateTransition: Transition<ChartState>
  pointerInDataCoordTransition: Transition<XY>

  // Computed chart state.
  currState: ChartState
  currPointerInDataCoord: XY
  derivedState: {
    prev: ChartStateDerived
    target: ChartStateDerived
    curr: ChartStateDerived
  }
}
