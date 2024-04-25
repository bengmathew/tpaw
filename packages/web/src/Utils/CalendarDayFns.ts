import { CalendarDay, CalendarMonthFns } from '@tpaw/common'
import { DateTime } from 'luxon'

export namespace CalendarDayFns {
  export const toStr = (
    day: CalendarDay,
    opts: { shortMonth?: boolean } = {},
  ): string => {
    const monthStr = CalendarMonthFns.toStr.justMonth(day, {
      shortMonth: opts.shortMonth,
    })
    return `${monthStr} ${day.day}, ${day.year}`
  }

  export const fromDateTime = ({ year, month, day }: DateTime): CalendarDay => {
    return { year, month, day }
  }
}
