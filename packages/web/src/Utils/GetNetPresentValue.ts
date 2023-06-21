import { assert } from '@tpaw/common'

export const getNetPresentValue = (r: number[], amounts: Float64Array) => {
  const n = amounts.length
  assert(r.length === n)
  const withCurrentMonth = new Float64Array(n)
  const withoutCurrentMonth = new Float64Array(n)
  for (let i = n - 1; i >= 0; i--) {
    withoutCurrentMonth[i] =
      i === n - 1 ? 0.0 : withCurrentMonth[i + 1] / (1.0 + r[i])
    withCurrentMonth[i] = amounts[i] + withoutCurrentMonth[i]
  }
  return {
    withCurrentMonth,
    withoutCurrentMonth,
  }
}
