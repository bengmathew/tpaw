import { Month, MonthRange, noCase } from '@tpaw/common'
import React from 'react'
import { PlanParamsExtended } from '../../UseSimulator/ExtentPlanParams'
import { calendarMonthStr } from '../../Utils/CalendarMonthStr'
import { numMonthsStr } from '../../Utils/NumMonthsStr'
import { SimpleRange } from '../../Utils/SimpleRange'
import { yourOrYourPartners } from '../../Utils/YourOrYourPartners'

export const MonthRangeDisplay = React.memo(
  ({
    className = '',
    valueClamped,
    skipLength,
    range,
    planParamsExt,
  }: {
    className?: string
    valueClamped: MonthRange | null
    range: SimpleRange | null
    skipLength: boolean
    planParamsExt: PlanParamsExtended
  }) => {
    const { monthRangeBoundsCheck } = planParamsExt
    const error =
      range &&
      valueClamped &&
      monthRangeBoundsCheck(valueClamped, range).errorMsgs.length > 0

    return (
      <div className={`${className} ${error ? 'text-errorFG' : ''}`}>
        {valueClamped ? (
          (() => {
            const monthRangeStr = monthRangeWithStartClampedToNowToString(
              valueClamped,
              planParamsExt,
            )
            return (
              <>
                <h2>{monthRangeStr.from.full}</h2>
                <h2>{monthRangeStr.until.full}</h2>
                {!skipLength && monthRangeStr.duration && (
                  <h2>{monthRangeStr.duration}</h2>
                )}
              </>
            )
          })()
        ) : (
          <>
            <h2 className="">
              <span className="bg-gray-300 px-2 py-0.5 rounded-lg text-sm">
                Completed
              </span>{' '}
              This entry is in the past.
            </h2>
          </>
        )}
      </div>
    )
  },
)

export const monthRangeWithStartClampedToNowToString = (
  value: MonthRange,
  planParamsExt: PlanParamsExtended,
) => {
  const { planParams, monthRangeLength } = planParamsExt
  const { withPartner } = planParams.people

  let fragments = { start: [] as string[], date: null as string | null }
  const combineAndResetFragments = () => {
    const { start: startFragments, date } = fragments
    fragments = { start: [], date: null }
    const start = startFragments.join(' ')
    return { start, date, full: date === null ? start : `${start} ${date}` }
  }

  let from: ReturnType<typeof combineAndResetFragments>
  let until: ReturnType<typeof combineAndResetFragments>
  let duration = '' as string | null

  const pushNamedAge = (month: Extract<Month, { type: 'namedAge' }>) => {
    switch (month.age) {
      case 'lastWorkingMonth':
        fragments.start.push(yourOrYourPartners(month))
        fragments.start.push('last working month')
        break
      case 'retirement':
        if (withPartner) fragments.start.push(yourOrYourPartners(month))
        fragments.start.push('retirement')
        break
      case 'max':
        fragments.start.push(yourOrYourPartners(month))
        fragments.start.push('max age')
        break
      default:
        noCase(month.age)
    }
  }

  const pushMonthForDuration = (month: Month) => {
    switch (month.type) {
      case 'calendarMonthAsNow':
        fragments.start.push('now')
        break
      case 'calendarMonth':
        fragments.start.push('at calendar month')
        fragments.date = calendarMonthStr(month.calendarMonth)
        break
      case 'namedAge':
        fragments.start.push('at')
        pushNamedAge(month)
        break
      case 'numericAge':
        fragments.start.push('when')
        fragments.start.push(yourOrYourPartners(month))
        fragments.start.push('age is')
        fragments.date = numMonthsStr(month.age.inMonths)
        break
      default:
        noCase(month)
    }
  }

  switch (value.type) {
    case 'startAndEnd':
      fragments.start.push('From')
      switch (value.start.type) {
        case 'calendarMonthAsNow':
          fragments.start.push('now')
          break
        case 'calendarMonth':
          fragments.start.push('calendar month')
          fragments.date = calendarMonthStr(value.start.calendarMonth)
          break
        case 'namedAge':
          pushNamedAge(value.start)
          break
        case 'numericAge':
          fragments.start.push('when')
          fragments.start.push(yourOrYourPartners(value.start))
          fragments.start.push('age is')
          fragments.date = numMonthsStr(value.start.age.inMonths)
          break
        default:
          noCase(value.start)
      }
      from = combineAndResetFragments()

      fragments.start.push('Until')
      switch (value.end.type) {
        case 'calendarMonthAsNow':
          fragments.start.push('now')
          break
        case 'calendarMonth':
          fragments.start.push('calendar month')
          fragments.date = calendarMonthStr(value.end.calendarMonth)
          break
        case 'namedAge':
          pushNamedAge(value.end)
          break
        case 'numericAge':
          fragments.start.push(yourOrYourPartners(value.end))
          fragments.start.push('age is')
          fragments.date = numMonthsStr(value.end.age.inMonths)
          break
        default:
          noCase(value.end)
      }
      until = combineAndResetFragments()
      duration = `That's ${numMonthsStr(monthRangeLength(value))}`
      break
    case 'startAndNumMonths':
      fragments.start.push('Starting')
      pushMonthForDuration(value.start)
      from = combineAndResetFragments()
      fragments.start.push('For a period of')
      fragments.date = numMonthsStr(value.numMonths)
      until = combineAndResetFragments()
      break
    case 'endAndNumMonths':
      fragments.start.push('For a period of')
      fragments.date = numMonthsStr(value.numMonths)
      from = combineAndResetFragments()

      fragments.start.push('Ending')
      pushMonthForDuration(value.end)
      until = combineAndResetFragments()

      break
    default:
      noCase(value)
  }
  return { from, until, duration }
}
