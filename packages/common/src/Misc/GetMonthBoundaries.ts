import { DateTime } from 'luxon'
import { assert, fGet } from '../Utils'
import _ from 'lodash'

export const getMonthBoundaries = (start: DateTime, end: DateTime) => ({
  startOfMonths: _getStartOfMonths(start, end),
  endOfMonths: _getEndOfMonths(start, end),
})

// start is closed, end in closed.
export const _getStartOfMonths = (start: DateTime, end: DateTime) => {
  let first = start.startOf('month')
  if (first !== start) first = first.plus({ month: 1 })
  const last = end.startOf('month')
  const numMonths = fGet(last.diff(first, 'month').toObject().months) + 1
  assert(_.isInteger(numMonths))
  return _.times(numMonths).map((i) => first.plus({ month: i }))
}

// start is closed, end in closed.
const _getEndOfMonths = (start: DateTime, end: DateTime) => {
  const msDelta = (x: DateTime | number, delta: 1 | -1) =>
    DateTime.fromMillis(x.valueOf() + delta)
  return _getStartOfMonths(msDelta(start, 1), msDelta(end, 1)).map((x) =>
    msDelta(x, -1),
  )
}
