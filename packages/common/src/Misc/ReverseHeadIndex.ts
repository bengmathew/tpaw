import { assert, fGet as fg } from '../Utils'

export namespace ReverseHeadIndex {
  export const fromHeadIndex = (headIndex: number, n: number) => {
    return n - 1 - headIndex
  }

  // This is really the same function as fromHeadIndex, but keeping it
  // separate for clarity.
  export const toHeadIndex = (reverseHeadIndex: number, n: number) => {
    return n - 1 - reverseHeadIndex
  }

  export const fGet = <T>(reverseHeadIndex: number, values: T[]) =>
    fg(values[toHeadIndex(reverseHeadIndex, values.length)])

  export const sliceToInclusive = <T>(
    reverseHeadIndex: number,
    values: T[],
  ): T[] => {
    const headIndex = toHeadIndex(reverseHeadIndex, values.length)
    assert(headIndex >= 0 && headIndex <= values.length)
    return values.slice(0, headIndex)
  }

  export const sliceFromInclusive = <T>(
    reverseHeadIndex: number,
    values: T[],
  ): T[] => {
    const headIndex = toHeadIndex(reverseHeadIndex, values.length)
    assert(headIndex >= 0 && headIndex < values.length)
    return values.slice(headIndex)
  }
}
