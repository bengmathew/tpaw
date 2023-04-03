import { DateTime } from 'luxon'

type _RemoveZone<T> = T extends [arg1: infer U, arg2?: infer V]
  ? [arg1: U, arg2?: Omit<V, 'zone'>]
  : never

// TODO: find all uses of DateTime construction and replace it with this.
export const getDateTimeInSameTimeZone = (x: DateTime) => {
  const zone = x.zoneName
  return {
    now: () => DateTime.now().setZone(zone),
    fromMillis: (...x: _RemoveZone<Parameters<typeof DateTime.fromMillis>>) =>
      DateTime.fromMillis(x[0], { ...x[1], zone }),
    fromObject: (...x: _RemoveZone<Parameters<typeof DateTime.fromObject>>) =>
      DateTime.fromObject(x[0], { ...x[1], zone }),
  }
}
