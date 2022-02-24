import {ChartState} from '../Chart'
import {ChartContext} from '../ChartContext'

export type ChartRegisterAnimation = <
  T extends gsap.core.Tween | gsap.core.Timeline
>(
  x: T
) => T

export interface ChartComponent<Data> {
  draw: (
    ctx: ChartContext<Data>,
    registerAnimation: ChartRegisterAnimation
  ) => void
  destroy?: () => void
  setPointerPosition?: (
    position: {x: number; y: number} | null,
    state: ChartState<Data>,
    context: CanvasRenderingContext2D,
    registerAnimation: ChartRegisterAnimation
  ) => void
  setState?: (
    state: ChartState<Data>,
    context: CanvasRenderingContext2D,
    registerAnimation: ChartRegisterAnimation
  ) => void
}
