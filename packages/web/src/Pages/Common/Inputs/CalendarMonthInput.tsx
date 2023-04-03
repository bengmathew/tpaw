import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import {
  faCaretDown,
  faCaretRight,
  faCheck,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { assert, CalendarMonth } from '@tpaw/common'
import _ from 'lodash'
import React, { useEffect, useMemo, useState } from 'react'
import { calendarMonthStr } from '../../../Utils/CalendarMonthStr'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { useSimulation } from '../../App/WithSimulation'
import { ModalListbox } from '../Modal/ModalListbox'

export const CalendarMonthInput = React.memo(
  ({
    className = '',
    value,
    onChange: onChangeIn,
    rangeAsMFN,
  }: {
    className?: string
    value: CalendarMonth
    onChange: (x: CalendarMonth) => void
    rangeAsMFN: SimpleRange
  }) => {
    const { paramsExt } = useSimulation()
    const { monthsFromNowToCalendarMonth, asMFN } = paramsExt

    const rangeAsCalendarMonth = useMemo(
      () => ({
        start: monthsFromNowToCalendarMonth(rangeAsMFN.start),
        end: monthsFromNowToCalendarMonth(rangeAsMFN.end),
      }),
      [monthsFromNowToCalendarMonth, rangeAsMFN.end, rangeAsMFN.start],
    )

    const monthRange = _.inRange(
      value.year,
      rangeAsCalendarMonth.start.year,
      rangeAsCalendarMonth.end.year + 1,
    )
      ? {
          start:
            rangeAsCalendarMonth.start.year === value.year
              ? rangeAsCalendarMonth.start.month
              : 1,
          end:
            rangeAsCalendarMonth.end.year === value.year
              ? rangeAsCalendarMonth.end.month
              : 12,
        }
      : null
    const monthChoices = useMemo(() => _.range(1, 13), [])

    const yearRange = {
      start: rangeAsCalendarMonth.start.year,
      end: rangeAsCalendarMonth.end.year,
    }
    const yearChoices = _.chunk(
      _.range(
        Math.min(value.year, _.floor(yearRange.start, -1)),
        Math.max(value.year, _.ceil(yearRange.end, -1)),
      ),
      10,
    )

    const getIsInRange = (x: CalendarMonth) =>
      _.inRange(asMFN(x), rangeAsMFN.start, rangeAsMFN.end + 1)
    const isInRange = getIsInRange(value)

    const handleChange = (x: CalendarMonth) => {
      if (getIsInRange(x)) {
        onChangeIn(x)
        return false
      } else {
        onChangeIn(
          monthsFromNowToCalendarMonth(
            _.clamp(asMFN(x), rangeAsMFN.start, rangeAsMFN.end),
          ),
        )
        return true
      }
    }
    return (
      <div
        className={`${className} grid gap-y-2 gap-x-2 `}
        style={{ grid: 'auto/auto 1fr' }}
      >
        <_Listbox
          choices={yearChoices}
          value={value.year}
          onChange={(year) => handleChange({ ...value, year })}
          isDisabled={(year) =>
            !_.inRange(year, yearRange.start, yearRange.end + 1)
          }
          toStr={(x) => `${x}`}
          error={!isInRange}
          type="year"
        />

        <_Listbox
          choices={monthChoices}
          value={value.month}
          onChange={(month) => {
            assert(_.inRange(month, 0, 14))
            const newValue =
              month === 0
                ? { year: value.year - 1, month: 12 }
                : month === 13
                ? { year: value.year + 1, month: 1 }
                : { year: value.year, month }
            return handleChange(newValue)
          }}
          isDisabled={(month) =>
            monthRange
              ? !_.inRange(month, monthRange.start, monthRange.end + 1)
              : false
          }
          toStr={(month) => calendarMonthStr.justMonth({ month })}
          error={!isInRange}
          type="month"
        />
      </div>
    )
  },
)

const _Listbox = React.memo(
  ({
    choices,
    value,
    onChange,
    isDisabled,
    toStr,
    error,
    type,
  }: {
    choices: number[][] | number[]
    value: number
    onChange: (x: number) => boolean
    isDisabled: (x: number) => boolean
    toStr: (x: number) => string
    error: boolean
    type: 'year' | 'month'
  }) => {
    const [outOfBounds, setOutOfBounds] = useState(false)
    useEffect(() => {
      const timeout = window.setTimeout(() => setOutOfBounds(false), 1000)
      return () => window.clearTimeout(timeout)
    }, [outOfBounds])
    return (
      <>
        <ModalListbox
          className=""
          choices={choices}
          value={value}
          onChange={onChange}
          align={'left'}
          isDisabled={isDisabled}
        >
          <div
            className={`flex justify-between items-center gap-x-1 border rounded-lg px-2 transition-all duration-1000 w-[130px]
        ${error ? 'border-errorFG text-errorFG ' : 'border-gray-400 '}
        ${outOfBounds ? 'border-errorFG bg-errorBlockBG ' : '  '}
        `}
          >
            {toStr(value)}
            <FontAwesomeIcon icon={faCaretDown} />
          </div>
          {typeof choices[0] === 'number'
            ? null
            : ({ choices, open }) => (
                <div className="py-3 px-5 flex items-center gap-x-2 font-bold lighten text-sm ">
                  {toStr(choices[0])}
                  <FontAwesomeIcon icon={open ? faCaretDown : faCaretRight} />
                </div>
              )}
          {({ choice: value, selected, active, disabled }) => {
            const valueStr = toStr(value)
            return (
              <div className=" cursor-pointer">
                <div
                  className={`py-2  pr-5 flex items-center 
                  ${type === 'month' ? 'pl-3' : 'pl-3'}
              ${active ? 'bg-gray-300' : ''}
              ${disabled ? 'opacity-50' : ''}
              `}
                >
                  <div className="w-[10px] mr-4">
                    {selected && (
                      <FontAwesomeIcon
                        className="w-[15px] mr-4"
                        icon={faCheck}
                      />
                    )}
                  </div>
                  {type === 'month' ? (
                    valueStr
                  ) : (
                    <div className="">
                      {/* <span className="lighten text-s  mr-0.5">{valueStr.slice(0, 2)}</span> */}
                      <span className="">{valueStr.slice(0)}</span>
                    </div>
                  )}
                </div>
              </div>
            )
          }}
        </ModalListbox>
        <div className="">
          <button
            className=" pl-3 pr-3"
            onClick={() => setOutOfBounds(onChange(value + 1))}
          >
            <FontAwesomeIcon icon={faPlus} />
          </button>
          <button
            className="pl-3 pr-3"
            onClick={() => setOutOfBounds(onChange(value - 1))}
          >
            <FontAwesomeIcon icon={faMinus} />
          </button>
        </div>
      </>
    )
  },
)
