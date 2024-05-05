import {
  CalendarDay,
  CalendarMonth,
  CalendarMonthFns,
  getZonedTimeFns,
} from '@tpaw/common'
import { DateTime } from 'luxon'

export namespace CalendarDayFns {
  export const toCalendarMonth = ({
    year,
    month,
  }: CalendarDay): CalendarMonth => ({ year, month })
  export const toStr = (
    day: CalendarDay,
    opts: { shortMonth?: boolean; skipYear?: boolean } = {},
  ): string => {
    const skipYear = opts.skipYear ?? false
    const monthStr = CalendarMonthFns.toStr.justMonth(day, {
      shortMonth: opts.shortMonth,
    })
    return `${monthStr} ${day.day}${skipYear ? '' : `, ${day.year}`}`
  }

  export const fromTimestamp = (
    timestamp: number,
    ianaTimezoneName: string,
  ): CalendarDay => fromDateTime(getZonedTimeFns(ianaTimezoneName)(timestamp))

  export const fromDateTime = ({ year, month, day }: DateTime): CalendarDay => {
    return { year, month, day }
  }
}
