import _ from 'lodash'
import { Transition } from './Transition'
import { zeroOneInterpolate } from './ZeroOneInterpolate'

// Thanks: https://stackoverflow.com/a/47842314
// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface InterpolatableObject extends Record<string, Interpolatable> {}
export type Interpolatable =
  | InterpolatableObject
  | number
  | (InterpolatableObject | number)[]

export function interpolate<
  T extends Interpolatable,
  U extends Interpolatable = T,
>(x: Transition<T>, transform?: (x: T) => U): U
export function interpolate<
  T extends Interpolatable,
  U extends Interpolatable = T,
>(from: T, target: T, progress: number, transform?: (x: T) => U): T
export function interpolate<
  T extends Interpolatable,
  U extends Interpolatable = T,
>(...args: any) {
  if (args.length === 3 || args.length === 4)
    return interpolate(
      { from: args[0], target: args[1], progress: args[2] },
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
      args[3],
    )
  const { from, target, progress } = args[0] as {
    from: T
    target: T
    progress: number
  }
  const transform = args[1] as (x: T) => U
  if (transform) {
    return interpolate({
      from: transform(from),
      target: transform(target),
      progress,
    })
  }
  // Optimization.
  if (progress === 0) return from
  if (progress === 1) return target
  if (typeof from === 'number') {
    return zeroOneInterpolate(from, target as number, progress) as T
  }
  if (from instanceof Array) {
    return from.map((x, i) =>
      interpolate(x as any, (target as any[])[i], progress),
    ) as T
  }
  return _.mapValues(from, (x, key) =>
    interpolate(x, (target as any)[key], progress),
  ) as T
}
