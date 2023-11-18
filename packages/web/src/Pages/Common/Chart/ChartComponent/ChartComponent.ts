import { Transition } from '../../../../Utils/Transition'
import { ChartContext } from '../ChartContext'

export type ChartRegisterAnimation = <
  T extends gsap.core.Tween | gsap.core.Timeline,
>(
  x: T,
) => T

export type ChartComponentPointerTargetY = (dataX: number) => {
  id: string
  pixelYTransition: { from: number; target: number }
  hoverTransition: Transition<0 | 1> | null
}[]

export interface ChartComponent<Data> {
  draw: (
    ctx: ChartContext<Data>,
    reason: 'draw' | 'init' | 'pointer' | 'stateAndPointer' | 'sizingAndPointer',
    registerAnimation: ChartRegisterAnimation,
    pointerTargetY: ChartComponentPointerTargetY,
  ) => ChartComponentPointerTargetY | null
  destroy?: () => void
}
