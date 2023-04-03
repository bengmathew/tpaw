import { assert, Month, MonthRange } from '@tpaw/common'
import React from 'react'
import { numMonthsStr } from '../../../Utils/NumMonthsStr'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { noCase } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { monthRangeWithStartClampedToNowToString } from '../MonthRangeDisplay'
import { MonthInput, MonthType } from './MonthInput/MonthInput'

export type MonthRangeInputProps = React.ComponentProps<typeof MonthRangeInput>

export const MonthRangeInput = React.memo(
  ({
    className = '',
    valueClamped,
    setValue,
    rangeAsMFN: rangeAsMFNIn,
    choices,
    modalTextInputOnMobile,
  }: {
    className?: string
    valueClamped: MonthRange
    setValue: (month: MonthRange) => void
    rangeAsMFN: SimpleRange
    choices: { start: MonthType[]; end: MonthType[] }
    modalTextInputOnMobile: boolean
  }) => {
    const { paramsExt } = useSimulation()
    const {
      monthRangeBoundsCheck,
      monthRangeLength,
      asMFN,
      monthsFromNowToNumericAge,
      getIsMonthPastMaxAge,
    } = paramsExt
    const rangeAsMFN = {
      start: Math.max(0, rangeAsMFNIn.start),
      end: rangeAsMFNIn.end,
    }
    assert(rangeAsMFN.end >= rangeAsMFN.start)

    const { errorMsgs } = monthRangeBoundsCheck(valueClamped, rangeAsMFN)

    const mfnToNumericAge = (mfn: number): Month => {
      const asPerson1 = monthsFromNowToNumericAge(mfn, 'person1')
      return getIsMonthPastMaxAge(asPerson1)
        ? monthsFromNowToNumericAge(mfn, 'person2')
        : asPerson1
    }

    const rangeFromStart = (
      start: Month | { numMonths: number },
    ): MonthRange => {
      if ('numMonths' in start) {
        switch (valueClamped.type) {
          case 'endAndNumMonths':
            return { ...valueClamped, numMonths: start.numMonths }
          case 'startAndEnd':
            return {
              type: 'endAndNumMonths',
              end: valueClamped.end,
              numMonths: monthRangeLength(valueClamped),
            }
          case 'startAndNumMonths':
            return {
              type: 'endAndNumMonths',
              end: mfnToNumericAge(asMFN(valueClamped).end),
              numMonths: monthRangeLength(valueClamped),
            }
          default:
            noCase(valueClamped)
        }
      } else {
        if (valueClamped.type === 'startAndNumMonths') {
          return { ...valueClamped, start }
        } else {
          return { type: 'startAndEnd', start, end: valueClamped.end }
        }
      }
    }
    const rangeFromEnd = (end: Month | { numMonths: number }): MonthRange => {
      if ('numMonths' in end) {
        switch (valueClamped.type) {
          case 'startAndNumMonths':
            return { ...valueClamped, numMonths: end.numMonths }
          case 'startAndEnd':
            return {
              type: 'startAndNumMonths',
              start: valueClamped.start,
              numMonths: monthRangeLength(valueClamped),
            }
          case 'endAndNumMonths':
            return {
              type: 'startAndNumMonths',
              start: mfnToNumericAge(asMFN(valueClamped).start),
              numMonths: monthRangeLength(valueClamped),
            }
          default:
            noCase(valueClamped)
        }
      } else {
        if (valueClamped.type === 'endAndNumMonths') {
          return { ...valueClamped, end }
        } else {
          return { type: 'startAndEnd', end, start: valueClamped.start }
        }
      }
    }
    const toString = (value: MonthRange) =>
      monthRangeWithStartClampedToNowToString(value, paramsExt)
    return (
      <div className={`${className}`}>
        <MonthInput
          className=""
          classNameForNumberInput="mb-4"
          valueClamped={
            valueClamped.type === 'endAndNumMonths'
              ? valueClamped
              : valueClamped.start
          }
          onChange={(x) => setValue(rangeFromStart(x))}
          toMFN={(x) => asMFN(rangeFromStart(x)).start}
          location="rangeStart"
          range={rangeAsMFN}
          choices={choices.start}
          modalTextInputOnMobile={modalTextInputOnMobile}
          getMonthLabel={(x) => toString(rangeFromStart(x)).from.start}
        />
        <MonthInput
          classNameForNumberInput="mb-4"
          valueClamped={
            valueClamped.type === 'startAndNumMonths'
              ? valueClamped
              : valueClamped.end
          }
          onChange={(x) => setValue(rangeFromEnd(x))}
          toMFN={(x) => asMFN(rangeFromEnd(x)).end}
          location="rangeEnd"
          range={
            valueClamped.type === 'endAndNumMonths'
              ? rangeAsMFN
              : { start: asMFN(valueClamped.start), end: rangeAsMFN.end }
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
            {`That's ${numMonthsStr(monthRangeLength(valueClamped))}`}
          </h2>
        )}
      </div>
    )
  },
)
