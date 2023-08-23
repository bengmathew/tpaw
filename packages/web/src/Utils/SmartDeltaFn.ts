import _ from 'lodash';
import { fGet } from './Utils';

export const smartDeltaFn = (
  breakpoints: { value: number; delta: number }[],
) => ({
  increment(x: number) {
    const round = (delta: number) => x + (delta - (x % delta))

    for (const { value, delta } of breakpoints) {
      if (x < value) return round(delta)
    }
    return round(fGet(_.last(breakpoints)).delta)
  },
  decrement(x: number) {
    const round = (delta: number) =>
      Math.max(0, x - (x % delta === 0 ? delta : x % delta))
    for (const { value, delta } of breakpoints) {
      if (x <= value) return round(delta)
    }
    return round(fGet(_.last(breakpoints)).delta)
  },
})
