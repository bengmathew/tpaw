import _ from 'lodash'
import { zeroOneInterpolate } from '../../../../Utils/ZeroOneInterpolate'

// TODO: Deprecate in favor or Interpolatable.

// Thanks: https://stackoverflow.com/a/47842314
type Indirect<X> = Record<string, number | X>
// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface TransitionObject extends Indirect<TransitionObject> {}

export const planSizingInterpolate =
  <State extends string, Obj extends TransitionObject>(
    states: Record<State, Obj>,
  ): ((t: { prev: State; target: State; progress: number }) => {
    [P in keyof Obj]: Obj[P]
  }) =>
  (t) =>
    interpolateObj(states[t.prev], states[t.target], t.progress)

export const interpolateObj = <Obj extends TransitionObject>(
  prev: Obj,
  target: Obj,
  progress: number,
): { [P in keyof Obj]: Obj[P] } => {
  if (progress === 0) return prev
  if (progress === 1) return target
  return _.mapValues(target, (value, key) =>
    typeof value === 'number'
      ? zeroOneInterpolate(prev[key] as number, target[key] as number, progress)
      : interpolateObj(prev[key] as any, target[key] as any, progress),
  ) as any
}
