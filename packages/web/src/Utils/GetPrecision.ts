export const getPrecision = (x: number) => {
  const parts = x.toString().split('.')
  return parts.length < 2 ? 0 : parts[1].length
}
