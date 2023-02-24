import { Month, MonthRange } from '@tpaw/common'
import React from 'react'
import { numMonthsStr } from '../../../Utils/NumMonthsStr'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { assert, noCase } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { monthRangeToString } from '../MonthRangeDisplay'
import { MonthInput, MonthType } from './MonthInput/MonthInput'

export type MonthRangeInputProps = React.ComponentProps<typeof MonthRangeInput>

export const MonthRangeInput = React.memo(
  ({
    className = '',
    value,
    setValue,
    range,
    choices,
    modalTextInputOnMobile,
  }: {
    className?: string
    value: MonthRange
    setValue: (month: MonthRange) => void
    range: SimpleRange
    choices: { start: MonthType[]; end: MonthType[] }
    modalTextInputOnMobile: boolean
  }) => {
    const { params, paramsExt } = useSimulation()
    const { people } = params
    const { monthRangeBoundsCheck, monthRangeLength, asMFN, months } = paramsExt
    const { errorMsgs } = monthRangeBoundsCheck(value, range)

    const mfnToNumericAge = (mfn: number): Month => {
      if (
        mfn >
        people.person1.ages.maxMonth - people.person1.ages.currentMonth
      ) {
        assert(people.withPartner)
        return months.person2.numericAge(mfn + people.person2.ages.currentMonth)
      } else {
        return months.person1.numericAge(mfn + people.person1.ages.currentMonth)
      }
    }
    const rangeFromStart = (
      start: Month | { numMonths: number },
    ): MonthRange => {
      if ('numMonths' in start) {
        switch (value.type) {
          case 'endAndNumMonths':
            return { ...value, numMonths: start.numMonths }
          case 'startAndEnd':
            return {
              type: 'endAndNumMonths',
              end: value.end,
              numMonths: monthRangeLength(value),
            }
          case 'startAndNumMonths':
            return {
              type: 'endAndNumMonths',
              end: mfnToNumericAge(asMFN(value).end),
              numMonths: monthRangeLength(value),
            }
          default:
            noCase(value)
        }
      } else {
        if (value.type === 'startAndNumMonths') {
          return { ...value, start }
        } else {
          return { type: 'startAndEnd', start, end: value.end }
        }
      }
    }
    const rangeFromEnd = (end: Month | { numMonths: number }): MonthRange => {
      if ('numMonths' in end) {
        switch (value.type) {
          case 'startAndNumMonths':
            return { ...value, numMonths: end.numMonths }
          case 'startAndEnd':
            return {
              type: 'startAndNumMonths',
              start: value.start,
              numMonths: monthRangeLength(value),
            }
          case 'endAndNumMonths':
            return {
              type: 'startAndNumMonths',
              start: mfnToNumericAge(asMFN(value).start),
              numMonths: monthRangeLength(value),
            }
          default:
            noCase(value)
        }
      } else {
        if (value.type === 'endAndNumMonths') {
          return { ...value, end }
        } else {
          return { type: 'startAndEnd', end, start: value.start }
        }
      }
    }
    const toString = (value: MonthRange) => monthRangeToString(value, paramsExt)
    return (
      <div className={`${className}`}>
        <MonthInput
          className=""
          classNameForNumberInput="mb-4"
          value={value.type === 'endAndNumMonths' ? value : value.start}
          onChange={(x) => setValue(rangeFromStart(x))}
          toMFN={(x) => asMFN(rangeFromStart(x)).start}
          location="rangeStart"
          range={range}
          choices={choices.start}
          modalTextInputOnMobile={modalTextInputOnMobile}
          getMonthLabel={(x) => toString(rangeFromStart(x)).from.start}
        />
        <MonthInput
          classNameForNumberInput="mb-4"
          value={value.type === 'startAndNumMonths' ? value : value.end}
          onChange={(x) => setValue(rangeFromEnd(x))}
          toMFN={(x) => asMFN(rangeFromEnd(x)).end}
          location="rangeEnd"
          range={
            value.type === 'endAndNumMonths'
              ? range
              : { start: asMFN(value.start), end: range.end }
          }
          choices={choices.end}
          modalTextInputOnMobile={modalTextInputOnMobile}
          getMonthLabel={(x) => toString(rangeFromEnd(x)).until.start}
        />
        {errorMsgs.length > 0 ? (
          <div className="bg-red-100 rounded-lg p-2 text-errorFG mt-4">
            {errorMsgs.map((x, i) => (
              <h2 key={i}>{x}</h2>
            ))}
          </div>
        ) : (
          <h2 className="mt-2 ">
            {`That's ${numMonthsStr(monthRangeLength(value))}`}
          </h2>
        )}
      </div>
    )
  },
)
