import { DateTime } from 'luxon'

type _RemoveZone<T> = T extends [arg1: infer U, arg2?: infer V]
  ? [arg1: U, arg2?: Omit<V, 'zone'>]
  : never

export const getZonedTimeFns = (ianaTimezoneName: string) => {
  const result = (...x: _RemoveZone<Parameters<typeof DateTime.fromMillis>>) =>
    DateTime.fromMillis(x[0], { ...x[1] }).setZone(ianaTimezoneName)

  result.now = () => DateTime.now().setZone(ianaTimezoneName)

  // Note. setZone() is WRONG for fromObject() because
  // it needs to be *interpreted* in the given timezone, not *converted*.
  result.fromObject = (
    ...x: _RemoveZone<Parameters<typeof DateTime.fromObject>>
  ) => DateTime.fromObject(x[0], { ...x[1], zone: ianaTimezoneName })

  return result
}

export const getNYZonedTime = getZonedTimeFns('America/New_York')
