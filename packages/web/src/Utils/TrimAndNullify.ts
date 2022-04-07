export const trimAndNullify = (x: string | null) => {
  if (!x) return x
  const trimmed = x.trim()
  return trimmed.length === 0 ? null : trimmed
}
