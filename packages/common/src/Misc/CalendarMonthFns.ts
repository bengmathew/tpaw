import { DateTime } from 'luxon'
import { CalendarMonth } from '../main'
import { assert, letIn, assertFalse } from '../Utils'

export namespace CalendarMonthFns {
  export const addMonths = (
    calendarMonth: CalendarMonth,
    monthsToAdd: number,
  ): CalendarMonth => {
    const asMonthsSince0 = calendarMonth.year * 12 + calendarMonth.month - 1
    const newAsMonthsSince0 = asMonthsSince0 + monthsToAdd
    assert(newAsMonthsSince0 >= 0) // Mod math doesn't work with negative numbers.
    const year = Math.floor(newAsMonthsSince0 / 12)
    const month = (newAsMonthsSince0 % 12) + 1
    return { year, month }
  }

  export const getFromMFN =
    (nowAsCalendarMonth: CalendarMonth) => (mfn: number) =>
      addMonths(nowAsCalendarMonth, mfn)

  export const getToMFN =
    (nowAsCalendarMonth: CalendarMonth) =>
    (calendarMonth: CalendarMonth): number =>
      diff(calendarMonth, nowAsCalendarMonth)

  export const diff = (a: CalendarMonth, b: CalendarMonth): number =>
    (a.year - b.year) * 12 + a.month - b.month

  export const fromTimestamp = (
    timestamp: number,
    ianaTimezoneName: string,
  ): CalendarMonth =>
    letIn(
      DateTime.fromMillis(timestamp).setZone(ianaTimezoneName),
      ({ year, month }) => ({ year, month }),
    )

  const _justMonth = (
    { month }: { month: number },
    opts: { shortMonth?: boolean } = {},
  ) => {
    const shortMonth = opts?.shortMonth ?? false
    switch (month) {
      case 1:
        return shortMonth ? 'Jan' : 'January'
      case 2:
        return shortMonth ? 'Feb' : 'February'
      case 3:
        return shortMonth ? 'Mar' : 'March'
      case 4:
        return shortMonth ? 'Apr' : 'April'
      case 5:
        return shortMonth ? 'May' : 'May'
      case 6:
        return shortMonth ? 'Jun' : 'June'
      case 7:
        return shortMonth ? 'Jul' : 'July'
      case 8:
        return shortMonth ? 'Aug' : 'August'
      case 9:
        return shortMonth ? 'Sep' : 'September'
      case 10:
        return shortMonth ? 'Oct' : 'October'
      case 11:
        return shortMonth ? 'Nov' : 'November'
      case 12:
        return shortMonth ? 'Dec' : 'December'
      default:
        assertFalse()
    }
  }

  export const toStr = (
    { year, month }: CalendarMonth,
    opts: { shortMonth?: boolean } = {},
  ) => `${_justMonth({ month }, { shortMonth: opts.shortMonth })} ${year}`
  toStr.justMonth = _justMonth
}
