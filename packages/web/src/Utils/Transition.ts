/* eslint-disable @typescript-eslint/no-unsafe-argument */
import { assert } from '@tpaw/common'
import _ from 'lodash'
import { zeroOneInterpolate } from './ZeroOneInterpolate'

export type Transition<Data, Target = Data> = {
  from: Data
  target: Target
  progress: number
}

export const transitionTransform = <T, U>(
  x: Transition<T>,
  transform: (x: T, other: T, src: 'from' | 'target') => U,
): Transition<U> => ({
  from: transform(x.from, x.target, 'from'),
  target: transform(x.target, x.from, 'target'),
  progress: x.progress,
})

export const transitionArrayDecompose = <T>({
  from,
  target,
  progress,
}: Transition<T[]>): Transition<T>[] =>
  _.zip(from, target).map(([from, target]) => {
    assert(from !== undefined && target !== undefined)
    return { from, target, progress }
  })

// Thanks: https://stackoverflow.com/a/47842314
type Indirect<X> = Record<string, number | number[] | X>
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
interface TransitionObject extends Indirect<TransitionObject> {}

export const chartDataTransitionCurrObj = <Obj extends TransitionObject, Data>(
  dataTransition: Transition<Data>,
  dataFn: (state: Data) => Obj,
): { [P in keyof Obj]: Obj[P] } => {
  const target = dataFn(dataTransition.target)
  const prev = dataFn(dataTransition.from)
  return _.mapValues(target, (value, key) =>
    typeof value === 'number'
      ? zeroOneInterpolate(
          prev[key] as number,
          target[key] as number,
          dataTransition.progress,
        )
      : value instanceof Array
      ? chartDataTransitionCurrNumArr(
          {
            from: prev[key] as any,
            target: target[key] as any,
            progress: dataTransition.progress,
          },
          (x: any) => x,
        )
      : chartDataTransitionCurrObj(
          {
            from: prev[key] as any,
            target: target[key] as any,
            progress: dataTransition.progress,
          },
          (x: any) => x,
        ),
  ) as any
}

export function chartDataTransitionCurrNum<Data>(
  dataTransition: Transition<Data>,
  dataFn: (state: Data) => number,
): number {
  const target = dataFn(dataTransition.target)
  const prev = dataFn(dataTransition.from)
  return zeroOneInterpolate(prev, target, dataTransition.progress)
}

export function chartDataTransitionCurrNumArr<Data>(
  dataTransition: Transition<Data>,
  dataFn: (state: Data) => number[],
): number[] {
  const target = dataFn(dataTransition.target)
  const prev = dataFn(dataTransition.from)
  return target.map((t, i) =>
    zeroOneInterpolate(prev[i], t, dataTransition.progress),
  )
}
