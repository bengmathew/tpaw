import { XY } from '../../../Utils/Geometry'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { Transition } from '../../../Utils/Transition'
import { ChartDataRange, ChartSizing, ChartStateDerived } from './Chart'

export type ChartContext<Data> = {
  canvasContext: CanvasRenderingContext2D
  stateTransition: Transition<{
    params: Data
    dataRange: ChartDataRange
    derivedState: ChartStateDerived
  }>
  pointerTransition: {
    visualPosition: Transition<XY>
    hover: Transition<0 | 1>
    // press: Transition<0 | 1>
  }
  sizing: ChartSizing

  currState: {
    dataRangeUnion: ChartDataRange
    dataXRangeCurrentlyVisible: SimpleRange
    derivedState: ChartStateDerived
    pointer: {
      position: { visual: XY; dataNotRounded: { x: number } }
      hover: number
      // press: number
    }
  }
}
