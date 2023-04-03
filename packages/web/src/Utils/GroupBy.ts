export const groupBy = <Value, Key>(
  values: Value[],
  getKey: (x: Value) => Key,
): Map<Key, Value[]> => {
  const result = new Map<Key, Value[]>()
  values.forEach((value) => {
    const key = getKey(value)
    const group = result.get(key)
    if (group) {
      group.push(value)
    } else {
      result.set(key, [value])
    }
  })
  return result
}
