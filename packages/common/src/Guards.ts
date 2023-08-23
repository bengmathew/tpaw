import { JSONGuard, chain, failure, string, success } from 'json-guard'
import { DateTime } from 'luxon'

export namespace Guards {
  export const ianaTimezoneName = chain(string, (zoneName) =>
    DateTime.now().setZone(zoneName).isValid
      ? success(zoneName)
      : failure('Invalid IANA timezone name'),
  )

  export const among =
    <T>(values: T[]): JSONGuard<T> =>
    (x: unknown) => {
      return values.includes(x as T)
        ? success(x as T)
        : failure('Not among predefined value.')
    }

  export const uuid = chain(string, (x) => {
    const UUID_REGEX =
      /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/
    return UUID_REGEX.test(x) ? success(x) : failure('Not a UUID.')
  })

  export const undef: JSONGuard<undefined> = (x: unknown) =>
    x === undefined ? success(x) : failure('Not "undefined".')
}
