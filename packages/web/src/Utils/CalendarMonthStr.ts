import { CalendarMonth } from '@tpaw/common'
import { DateTime } from 'luxon'

const _justMonth = ({ month }: { month: number }) =>
  DateTime.local(0, month).toFormat('LLLL')

export const calendarMonthStr = ({ year, month }: CalendarMonth) =>
  `${_justMonth({ month })} ${year}`
calendarMonthStr.justMonth = _justMonth
