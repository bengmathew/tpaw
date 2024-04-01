import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import {
  faCaretDown,
  faCaretRight,
  faCheck,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { CalendarMonth, CalendarMonthFns, assert } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { useEffect, useMemo, useState } from 'react'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { useSimulation } from '../../../PlanRoot/PlanRootHelpers/WithSimulation'
import { ModalListbox } from '../../Modal/ModalListbox'

// TODO: Handle validRangeAsMFN = null (currently crashes)
export const CalendarMonthInput = React.memo(
  ({
    className,
    normValue: { baseValue, validRangeAsMFN },
    onChange: onChangeIn,
  }: {
    className?: string
    normValue: {
      baseValue: CalendarMonth
      validRangeAsMFN: { includingLocalConstraints: SimpleRange | null }
    }
    onChange: (x: CalendarMonth) => void
  }) => {
    const { planParamsNorm } = useSimulation()
    const { nowAs } = planParamsNorm
    const toMFN = CalendarMonthFns.getToMFN(nowAs.calendarMonth)
    const fromMFN = CalendarMonthFns.getFromMFN(nowAs.calendarMonth)
    const rangeAsCalendarMonth = validRangeAsMFN.includingLocalConstraints
      ? {
          start: fromMFN(validRangeAsMFN.includingLocalConstraints.start),
          end: fromMFN(validRangeAsMFN.includingLocalConstraints.end),
        }
      : null

    const monthRange =
      rangeAsCalendarMonth &&
      SimpleRange.Closed.isIn(baseValue.year, {
        start: rangeAsCalendarMonth.start.year,
        end: rangeAsCalendarMonth.end.year,
      })
        ? {
            start:
              rangeAsCalendarMonth.start.year === baseValue.year
                ? rangeAsCalendarMonth.start.month
                : 1,
            end:
              rangeAsCalendarMonth.end.year === baseValue.year
                ? rangeAsCalendarMonth.end.month
                : 12,
          }
        : null
    const monthChoices = useMemo(() => _.range(1, 13), [])

    const yearRange = rangeAsCalendarMonth
      ? {
          start: rangeAsCalendarMonth.start.year,
          end: rangeAsCalendarMonth.end.year,
        }
      : null
    const yearChoices = yearRange
      ? _.chunk(
          _.range(
            Math.min(baseValue.year, _.floor(yearRange.start, -1)),
            Math.max(baseValue.year, _.ceil(yearRange.end, -1)),
          ),
          10,
        )
      : []

    const isInRange = SimpleRange.Closed.isIn(
      toMFN(baseValue),
      validRangeAsMFN.includingLocalConstraints,
    )

    const handleChange = (x: CalendarMonth) => {
      assert(validRangeAsMFN.includingLocalConstraints)
      if (
        SimpleRange.Closed.isIn(
          toMFN(x),
          validRangeAsMFN.includingLocalConstraints,
        )
      ) {
        onChangeIn(x)
        return false
      } else {
        onChangeIn(
          fromMFN(
            SimpleRange.Closed.clamp(
              toMFN(x),
              validRangeAsMFN.includingLocalConstraints,
            ),
          ),
        )
        return true
      }
    }
    return (
      <div
        className={clsx(className, `grid gap-y-2 gap-x-2 `)}
        style={{ grid: 'auto/auto 1fr' }}
      >
        <_Listbox
          choices={yearChoices}
          value={baseValue.year}
          onChange={(year) => handleChange({ ...baseValue, year })}
          isDisabled={(year) => !SimpleRange.Closed.isIn(year, yearRange)}
          toStr={(x) => `${x}`}
          error={!isInRange}
          type="year"
        />

        <_Listbox
          choices={monthChoices}
          value={baseValue.month}
          onChange={(month) => {
            assert(_.inRange(month, 0, 14))
            const newValue =
              month === 0
                ? { year: baseValue.year - 1, month: 12 }
                : month === 13
                  ? { year: baseValue.year + 1, month: 1 }
                  : { year: baseValue.year, month }
            return handleChange(newValue)
          }}
          isDisabled={(month) =>
            monthRange
              ? !_.inRange(month, monthRange.start, monthRange.end + 1)
              : false
          }
          toStr={(month) => CalendarMonthFns.toStr.justMonth({ month })}
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
