import _ from 'lodash'

export type SimpleRange = { start: number; end: number }

export namespace SimpleRange {
  export namespace Closed {
    export const isIn = (x: number, range: SimpleRange | null) =>
      range && range.start <= x && x <= range.end

    export const toArr = (range: SimpleRange | null, step = 1) =>
      range ? _.range(range.start, range.end + 1, step) : []

    export const clamp = (x: number, range: SimpleRange) =>
      _.clamp(x, range.start, range.end)
  }

  export function union(a: SimpleRange, b: SimpleRange): SimpleRange
  export function union(
    a: SimpleRange | null,
    b: SimpleRange | null,
  ): SimpleRange | null
  export function union(
    a: SimpleRange | null,
    b: SimpleRange | null,
  ): SimpleRange | null {
    return !a
      ? b
      : !b
      ? a
      : { start: Math.min(a.start, b.start), end: Math.max(a.end, b.end) }
  }

  export const intersection = (
    a: SimpleRange | null,
    b: SimpleRange | null,
  ): SimpleRange | null => {
    if (!a || !b) return null
    const candidate = {
      start: Math.max(a.start, b.start),
      end: Math.min(a.end, b.end),
    }
    return candidate.start < candidate.end ? candidate : null
  }
}
