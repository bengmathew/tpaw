import { assert } from '@tpaw/common'
import _ from 'lodash'
import { Transition } from './Transition'
import { zeroOneInterpolate } from './ZeroOneInterpolate'
import { Record } from './Record'

// Thanks: https://stackoverflow.com/a/47842314
// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface InterpolatableObject extends Record<string, Interpolatable> {}
export type Interpolatable =
  | InterpolatableObject
  | number
  | string
  | Interpolatable[]
  | readonly Interpolatable[]


export function interpolate<T extends Interpolatable>({
  from,
  target,
  progress,
}: Transition<T>): T {
  // Optimization.
  if (progress === 0) return from
  if (progress === 1) return target
  if (typeof from === 'string') {
    return target as T
  }
  if (typeof from === 'number') {
    return zeroOneInterpolate(from, target as number, progress) as T
  }
  if (from instanceof Array) {
    assert(target instanceof Array)
    assert(from.length === target.length)
    return from.map((from, i) =>
      interpolate({ from, target: target[i], progress }),
    ) as T
  }
  return _.mapValues(from, (from, key) =>
    interpolate({ from, target: (target as any)[key], progress }),
  ) as T
}
