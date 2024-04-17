import { CalendarMonthFns, Month, noCase } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { NormalizedDurationNotInThePast } from '../../UseSimulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeAmountAndTimingRecurring'
import {
  NormalizedMonthInThePast,
  NormalizedMonthNotInThePast,
} from '../../UseSimulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizedMonth'
import {
  youAreOrYourPartnerIs,
  yourOrYourPartners,
} from '../../Utils/YourOrYourPartners'
import { InMonthsFns } from '../../Utils/InMonthsFns'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faTurnDownLeft } from '@fortawesome/pro-solid-svg-icons'

export const MonthOrDurationDisplay = React.memo(
  ({
    className = '',
    normValue,
  }: {
    className?: string
    // eslint-disable-next-line @typescript-eslint/no-redundant-type-constituents
    normValue:
      | NormalizedMonthNotInThePast
      | NormalizedDurationNotInThePast
      | NormalizedMonthInThePast
  }) => {
    const str = getNormalizedMonthStr(normValue).full
    return (
      <div
        className={`${className} ${normValue.errorMsg ? 'text-errorFG' : ''}`}
      >
        <h2>{str}</h2>
        {normValue.errorMsg && (
          <div className="ml-2">
            <FontAwesomeIcon
              className="fa-rotate-90 mb-1 mr-1 text-xs"
              icon={faTurnDownLeft}
            />{' '}
            {normValue.errorMsg}
          </div>
        )}
      </div>
    )
  },
)

export const getNormalizedMonthStr = (
  normValue:
    | { isInThePast: false; baseValue: Month }
    | { isInThePast: false; baseValue: { inMonths: number } }
    | { isInThePast: true },
) => {
  const result = (
    monthTypeMenuLabel: string,
    monthTypeMenuDash: 'dash' | 'noDash',
    full = monthTypeMenuLabel,
  ) => {
    return {
      forMonthTypeMenu: {
        label: monthTypeMenuLabel,
        dash: monthTypeMenuDash === 'dash',
      },
      full,
    }
  }
  if (normValue.isInThePast) {
    return result('A month in the past', 'noDash')
  }
  if ('inMonths' in normValue.baseValue) {
    return result(InMonthsFns.toStr(normValue.baseValue), 'noDash')
  }
  switch (normValue.baseValue.type) {
    case 'now':
      return result('Now', 'noDash')
    case 'calendarMonth':
      return result(
        'At calendar month',
        'dash',
        CalendarMonthFns.toStr(normValue.baseValue.calendarMonth),
      )
    case 'namedAge': {
      const prefix = `${yourOrYourPartners(normValue.baseValue.person)}`
      switch (normValue.baseValue.age) {
        case 'lastWorkingMonth':
          return result(`At ${prefix} last working month`, 'noDash')
        case 'retirement':
          return result(`At ${prefix} retirement`, 'noDash')
        case 'max':
          return result(`At ${prefix} max age`, 'noDash')
        default:
          noCase(normValue.baseValue)
      }
    }
    case 'numericAge':
      const prefix = `When ${youAreOrYourPartnerIs(normValue.baseValue.person)}`
      return result(
        prefix,
        'dash',
        `${prefix} ${InMonthsFns.toStr(normValue.baseValue.age)}`,
      )
    default:
      noCase(normValue.baseValue)
  }
}
