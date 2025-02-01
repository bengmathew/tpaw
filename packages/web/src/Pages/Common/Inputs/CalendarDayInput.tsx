import { faCaretLeft, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, fGet } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useMemo, useState } from 'react'

export const CalendarDayInput = React.memo(
  ({
    className,
    valueInfo,
    range_memoized,
    onChange,
    shouldHighlightDay,
  }: {
    className?: string
    valueInfo:
      | { hasValue: true; value: DateTime } // Should be at start of day
      | { hasValue: false; startingMonth: DateTime } // Should be at start of month
    range_memoized: { start: DateTime; end: DateTime } // Should be at start of day
    onChange: (value: DateTime) => void
    shouldHighlightDay: false | ((day: DateTime) => boolean)
  }) => {
    assert(
      [range_memoized.start, range_memoized.end].every(
        (x) => x.startOf('day').toMillis() === x.toMillis(),
      ),
    )
    if (valueInfo.hasValue)
      assert(
        valueInfo.value.startOf('day').toMillis() ===
          valueInfo.value.toMillis(),
      )
    else
      assert(
        valueInfo.startingMonth.startOf('month').toMillis() ===
          valueInfo.startingMonth.toMillis(),
      )

    const { monthRange, yearRange } = useMemo(() => {
      return {
        monthRange: {
          start: range_memoized.start.startOf('month'),
          end: range_memoized.end.startOf('month'),
        },
        yearRange: {
          start: range_memoized.start.startOf('year'),
          end: range_memoized.end.startOf('year'),
        },
      }
    }, [range_memoized])

    const [month, setMonth] = useState(
      valueInfo.hasValue
        ? valueInfo.value.startOf('month')
        : valueInfo.startingMonth,
    )
    const { weeks, year } = useMemo(() => {
      const weeks = [] as (DateTime | null)[][]
      _.times(fGet(month.daysInMonth))
        .map((x) => month.plus({ days: x }).startOf('day'))
        .forEach((day, i) => {
          if (day.weekdayShort === 'Sun' || i === 0) weeks.push([])
          fGet(_.last(weeks)).push(day)
        })
      const n = weeks.length
      weeks[0] = [..._.times(7 - weeks[0].length, () => null), ...weeks[0]]
      weeks[n - 1] = [
        ...weeks[n - 1],
        ..._.times(7 - weeks[n - 1].length, () => null),
      ]
      return { weeks, year: month.startOf('year') }
    }, [month])
    return (
      <div className={clsx(className, '')}>
        <div className="my-2">
          <div className="flex items-center">
            <button
              className="px-4 py-2 text-xl disabled:lighten-2"
              disabled={year.toMillis() === yearRange.start.toMillis()}
              onClick={() =>
                setMonth(
                  DateTime.max(
                    monthRange.start,
                    month.minus({ years: 1 }).startOf('month'),
                  ),
                )
              }
            >
              <FontAwesomeIcon icon={faCaretLeft} />
            </button>
            <h2 className="text-lg font-semibold w-[100px] text-center">
              {month.year}
            </h2>
            <button
              className="px-4 py-2 text-xl disabled:lighten-2"
              disabled={year.toMillis() === yearRange.end.toMillis()}
              onClick={() =>
                setMonth(
                  DateTime.min(
                    monthRange.end,
                    month.plus({ years: 1 }).startOf('month'),
                  ),
                )
              }
            >
              <FontAwesomeIcon icon={faCaretRight} />
            </button>
          </div>
          <div className="flex justify-between">
            <div className="flex items-center">
              <button
                className="px-4 py-2 text-xl disabled:lighten-2"
                disabled={month.toMillis() === monthRange.start.toMillis()}
                onClick={() =>
                  setMonth(month.minus({ months: 1 }).startOf('month'))
                }
              >
                <FontAwesomeIcon icon={faCaretLeft} />
              </button>
              <h2 className="text-lg font-semibold w-[100px] text-center">
                {month.monthLong}
              </h2>
              <button
                className="px-4 py-2 text-xl disabled:lighten-2"
                disabled={month.toMillis() === monthRange.end.toMillis()}
                onClick={() =>
                  setMonth(month.plus({ months: 1 }).startOf('month'))
                }
              >
                <FontAwesomeIcon icon={faCaretRight} />
              </button>
            </div>
          </div>
        </div>
        <div
          className="inline-grid text-center  bg-gray-100 rounded-lg  "
          style={{ grid: 'auto/1fr 1fr 1fr 1fr 1fr 1fr 1fr' }}
        >
          {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((x, i) => (
            <div
              key={i}
              className={clsx(
                'w-[45px] h-[45px]  border-b border-gray-600 rounded-m flex items-center justify-center font-medium',
                x === 'S' && 'text-red-500',
              )}
            >
              {x}
            </div>
          ))}
          {weeks.map((week, i) =>
            week.map((day, j) =>
              day ? (
                <_Day
                  key={`${i}-${j}`}
                  day_memoized={day}
                  isCurrent={
                    valueInfo.hasValue &&
                    day.toMillis() === valueInfo.value.toMillis()
                  }
                  range_memoized={range_memoized}
                  shouldHighlight={
                    shouldHighlightDay && shouldHighlightDay(day)
                  }
                  onChange={onChange}
                />
              ) : (
                <div key={`${i}-${j}`}></div>
              ),
            ),
          )}
          {_.range(weeks.length, 6).map((week) => (
            <div key={week} className="h-[45px]" />
          ))}
        </div>
      </div>
    )
  },
)

const _Day = React.memo(
  ({
    className,
    day_memoized,
    shouldHighlight,
    isCurrent,
    range_memoized,
    onChange,
  }: {
    className?: string
    day_memoized: DateTime
    isCurrent: boolean
    range_memoized: { start: DateTime; end: DateTime }
    shouldHighlight: boolean
    onChange: (value: DateTime) => void
  }) => {
    const [isHover, setIsHover] = useState(false)
    const isBeforeMin =
      day_memoized.toMillis() < range_memoized.start.toMillis()
    const isAfterMax = day_memoized.toMillis() > range_memoized.end.toMillis()

    const disabled = isBeforeMin || isAfterMax || isCurrent

    return (
      <button
        className=""
        disabled={disabled}
        onClick={() => onChange(day_memoized)}
        onMouseEnter={() => setIsHover(true)}
        onMouseLeave={() => setIsHover(false)}
      >
        <div
          className={clsx(
            className,
            'w-[45px] h-[45px]   rounded-md flex items-center justify-center relative',
            isCurrent ? 'bg-gray-600 text-white' : isHover ? 'bg-gray-300' : '',
          )}
          // From: https://www.magicpattern.design/tools/css-backgrounds
          style={{
            ...(isBeforeMin || isAfterMax
              ? {
                  opacity: '0.2',
                  // backgroundSize: '8px 8px',
                  // backgroundImage: `repeating-linear-gradient(45deg, ${gray[300]} 0, #a7a7b3 0.8px, #f6f6f6 0, #f6f6f6 50%)`,
                }
              : {}),
          }}
        >
          <div
            className={clsx(
              'w-[27px] h-[27px] flex items-center justify-center ',
              shouldHighlight && 'rounded-full border',
              isCurrent ? 'border-gray-300' : 'border-gray-500',
            )}
          >
            {day_memoized.day}
          </div>
        </div>
      </button>
    )
  },
)
