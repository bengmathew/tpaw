import { DateTime } from 'luxon'

type _RemoveZone<T> = T extends [arg1: infer U, arg2?: infer V]
  ? [arg1: U, arg2?: Omit<V, 'zone'>]
  : never

export const getZonedTimeFns = (ianaTimezoneName: string) => {
  const setZone = (x: DateTime) => x.setZone(ianaTimezoneName)

  const result = (...x: _RemoveZone<Parameters<typeof DateTime.fromMillis>>) =>
    setZone(DateTime.fromMillis(x[0], { ...x[1] }))
  // function literal using object notation

  result.now = () => setZone(DateTime.now())
  result.fromObject = (
    ...x: _RemoveZone<Parameters<typeof DateTime.fromObject>>
  ) => setZone(DateTime.fromObject(x[0], { ...x[1] }))
  
  result.fromISO = (
    ...x: _RemoveZone<Parameters<typeof DateTime.fromISO>>
  ) => setZone(DateTime.fromISO(x[0], { ...x[1] }))
  
  return result
}

export const getNYZonedTime = getZonedTimeFns('America/New_York')
