import { faCaretDown } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { linearFnFomPoints, Month } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { noCase } from '../../../../Utils/Utils'
import { useSimulation } from '../../../App/WithSimulation'
import { CalendarMonthInput } from '../CalendarMonthInput'
import { NumMonthsInput } from '../NumMonthsInput'
import { MonthInputProps } from './MonthInput'

export const MonthInputButton = React.memo(
  ({
    className = '',
    classNameForNumberInput = '',
    onOpenPicker,
    setRef,
    ...props
  }: {
    onOpenPicker: () => void
    setRef: (x: HTMLDivElement) => void
  } & MonthInputProps) => {
    const { valueClamped, range, toMFN, getMonthLabel, onChange } = props
    const label = getMonthLabel(valueClamped)
    const numMonthsProps = useNumMonthsProps(valueClamped)
    const inRange =
      _.inRange(toMFN(valueClamped), range.start, range.end + 1) &&
      (numMonthsProps
        ? _.inRange(
            numMonthsProps.asNumber,
            numMonthsProps.range.start,
            numMonthsProps.range.end + 1,
          )
        : true)

    return (
      <div
        className={`${className} 
        ${!inRange ? 'text-errorFG' : ''} `}
        ref={setRef}
      >
        <button
          className={`flex items-center gap-x-2 py-1.5`}
          onClick={onOpenPicker}
        >
          <h2 className="inline-block text-left">{label}</h2>
          <FontAwesomeIcon icon={faCaretDown} />
        </button>
        {numMonthsProps && (
          <_NumMonths
            className={`${classNameForNumberInput} ml-6 mt-2`}
            {...props}
            numericalComponent={numMonthsProps}
          />
        )}
        {'type' in valueClamped && valueClamped.type === 'calendarMonth' && (
          <CalendarMonthInput
            className={`${classNameForNumberInput} ml-6 mt-2`}
            value={valueClamped.calendarMonth}
            onChange={(calendarMonth) =>
              onChange({ type: 'calendarMonth', calendarMonth })
            }
            rangeAsMFN={range}
          />
        )}
      </div>
    )
  },
)

const _NumMonths = React.memo(
  ({
    className = '',
    onChange,
    toMFN,
    range,
    numericalComponent,
    modalTextInputOnMobile,
  }: MonthInputProps & {
    numericalComponent: Exclude<ReturnType<typeof useNumMonthsProps>, null>
  }) => {
    const { asNumber, fromNumber, modalLabel } = numericalComponent
    const numMonthRange = (() => {
      const mfnToNumber = (x: number) =>
        Math.round(
          linearFnFomPoints(
            0,
            toMFN(fromNumber(0)),
            1,
            toMFN(fromNumber(1)),
          ).inverse(x),
        )
      const edge1 = mfnToNumber(range.start)
      const edge2 = mfnToNumber(range.end)
      // Cannot assume edge1 < edge2 because of endWithInterval.
      const start = Math.min(edge1, edge2)
      const end = Math.max(edge1, edge2)
      return {
        start: Math.max(start, numericalComponent.range.start),
        end: Math.min(end, numericalComponent.range.end),
      }
    })()
    return (
      <NumMonthsInput
        className={className}
        value={asNumber}
        onChange={(x) => onChange(fromNumber(x))}
        rangeAsMFN={numMonthRange}
        modalLabel={modalTextInputOnMobile ? modalLabel : null}
      />
    )
  },
)

const useNumMonthsProps = (
  value: Month | { numMonths: number },
): {
  asNumber: number
  fromNumber: (x: number) => Month | { numMonths: number }
  range: SimpleRange
  modalLabel: string
} | null => {
  const { paramsExt } = useSimulation()
  const { pickPerson, getCurrentAgeOfPerson } = paramsExt

  if ('numMonths' in value) {
    return {
      asNumber: value.numMonths,
      fromNumber: (numMonths: number) => ({ numMonths }),
      range: { start: 0, end: Number.MAX_SAFE_INTEGER },
      modalLabel: 'Number of Months',
    }
  } else {
    switch (value.type) {
      case 'calendarMonthAsNow':
      case 'calendarMonth':
      case 'namedAge':
        return null
      case 'numericAge':
        const { ages } = pickPerson(value.person)
        return {
          asNumber: value.age.inMonths,
          fromNumber: (inMonths: number): Month => ({
            ...value,
            age: { inMonths },
          }),
          range: {
            start: getCurrentAgeOfPerson(value.person).inMonths,
            end: ages.maxAge.inMonths,
          },
          modalLabel: `${
            value.person === 'person1' ? 'Your' : "Your Partner's"
          } Age`,
        }
      default:
        noCase(value)
    }
  }
}
