import { assert } from '@tpaw/common'


export const chartMergeDataComponentsArrTransition = <T extends { id: string} >(
  from: T[],
  target: T[]
) => {
  const result: ({ from: T; target: T}  |
  { from: T; target: null}  |
  { from: null; target: T} )[] = []
  let fromIndex = 0
  let targetIndex = 0
  while (fromIndex < from.length || targetIndex < target.length) {
    const currFrom = fromIndex < from.length ? from[fromIndex] : null
    const currTarget = targetIndex < target.length ? target[targetIndex] : null
    if (currTarget === null) {
      assert(currFrom)
      result.push({ from: currFrom, target: null })
      fromIndex += 1
    } else if (currFrom === null) {
      assert(currTarget)
      result.push({ from: null, target: currTarget })
      targetIndex += 1
    } else {
      if (currFrom.id !== currTarget.id) {
        result.push({ from: currFrom, target: null })
        fromIndex += 1
      } else {
        result.push({ from: currFrom, target: currTarget })
        fromIndex += 1
        targetIndex += 1
      }
    }
  }
  return result
}
