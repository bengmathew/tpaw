import React from 'react'
import {Year, YearRange} from '@tpaw/common'
import {pluralize} from '../../../Utils/Pluralize'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {assert, noCase} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {YearInput, YearType} from './YearInput/YearInput'

export type YearRangeInputProps = React.ComponentProps<typeof YearRangeInput>

export const YearRangeInput = React.memo(
  ({
    className = '',
    value,
    setValue,
    range,
    choices,
    modalTextInputOnMobile
  }: {
    className?: string
    value: YearRange
    setValue: (year: YearRange) => void
    range: SimpleRange
    choices: {start: YearType[]; end: YearType[]}
    modalTextInputOnMobile:boolean
  }) => {
    const {params, paramsExt} = useSimulation()
    const {people} = params
    const {yearRangeBoundsCheck, yearRangeLength, asYFN, years} = paramsExt
    const {errorMsgs} = yearRangeBoundsCheck(value, range)

    const yfnToAge = (yfn: number): Year => {
      if (yfn > people.person1.ages.max - people.person1.ages.current) {
        assert(people.withPartner)
        return years.person2.numericAge(yfn + people.person2.ages.current)
      } else {
        return years.person1.numericAge(yfn + people.person1.ages.current)
      }
    }
    const rangeFromStart = (start: Year | {numYears: number}): YearRange => {
      if ('numYears' in start) {
        switch (value.type) {
          case 'endAndNumYears':
            return {...value, numYears: start.numYears}
          case 'startAndEnd':
            return {
              type: 'endAndNumYears',
              end: value.end,
              numYears: yearRangeLength(value),
            }
          case 'startAndNumYears':
            return {
              type: 'endAndNumYears',
              end: yfnToAge(asYFN(value).end),
              numYears: yearRangeLength(value),
            }
          default:
            noCase(value)
        }
      } else {
        if (value.type === 'startAndNumYears') {
          return {...value, start}
        } else {
          return {type: 'startAndEnd', start, end: value.end}
        }
      }
    }
    const rangeFromEnd = (end: Year | {numYears: number}): YearRange => {
      if ('numYears' in end) {
        switch (value.type) {
          case 'startAndNumYears':
            return {...value, numYears: end.numYears}
          case 'startAndEnd':
            return {
              type: 'startAndNumYears',
              start: value.start,
              numYears: yearRangeLength(value),
            }
          case 'endAndNumYears':
            return {
              type: 'startAndNumYears',
              start: yfnToAge(asYFN(value).start),
              numYears: yearRangeLength(value),
            }
          default:
            noCase(value)
        }
      } else {
        if (value.type === 'endAndNumYears') {
          return {...value, end}
        } else {
          return {type: 'startAndEnd', end, start: value.start}
        }
      }
    }
    return (
      <div className={`${className}`}>
        <YearInput
          className=""
          value={value.type === 'endAndNumYears' ? value : value.start}
          onChange={x => setValue(rangeFromStart(x))}
          toYFN={x => asYFN(rangeFromStart(x)).start}
          location="rangeStart"
          range={range}
          choices={choices.start}
          modalTextInputOnMobile={modalTextInputOnMobile}
        />
        <YearInput
          className=""
          value={value.type === 'startAndNumYears' ? value : value.end}
          onChange={x => setValue(rangeFromEnd(x))}
          toYFN={x => asYFN(rangeFromEnd(x)).end}
          location="rangeEnd"
          range={
            value.type === 'endAndNumYears'
              ? range
              : {start: asYFN(value.start), end: range.end}
          }
          choices={choices.end}
          modalTextInputOnMobile={modalTextInputOnMobile}
        />
        {errorMsgs.length > 0 ? (
          <div className="bg-red-100 rounded-lg p-2 text-errorFG mt-4">
            {errorMsgs.map((x, i) => (
              <h2 key={i}>{x}</h2>
            ))}
          </div>
        ) : (
          <h2 className="mt-1 ">
            {`That's ${pluralize(yearRangeLength(value), 'year')}.`}
          </h2>
        )}
      </div>
    )
  }
)
