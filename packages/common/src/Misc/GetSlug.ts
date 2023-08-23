import slugify from 'slugify'

export const getSlug = (
  label: string | null,
  existing: string[],
  index = null as number | null,
): string => {
  const candidate = `${slugify(label ?? 'untitled', {
    lower: true,
    strict: true,
  })}${index === null ? '' : `-${index}`}`
  return !existing.includes(candidate)
    ? candidate
    : getSlug(label, existing, index === null ? 1 : index + 1)
}
