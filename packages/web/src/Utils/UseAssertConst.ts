import React from 'react'

export function useAssertConst(curr: any[]) {
  const [saved] = React.useState(curr)
  if (saved.length !== curr.length) {
    throw new Error(
      `Dependency lengths changed from ${saved.length} to ${curr.length}`
    )
  }

  for (const [i, fromSaved] of Array.from(saved.entries())) {
    if (fromSaved !== curr[i]) {
      throw new Error(`Value at index ${i} is not const.`)
    }
  }
}
