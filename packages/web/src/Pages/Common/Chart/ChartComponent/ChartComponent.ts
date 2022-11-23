import {ChartContext} from '../ChartContext'

export type ChartRegisterAnimation = <
  T extends gsap.core.Tween | gsap.core.Timeline
>(
  x: T
) => T

export interface ChartComponent<Data> {
  draw: (ctx: ChartContext<Data>) => void
  destroy?: () => void
  update?: (
    change: 'init' | 'pointer' | 'stateAndPointer' | 'sizing',
    ctx: ChartContext<Data>,
    registerAnimation: ChartRegisterAnimation
  ) => void
}
