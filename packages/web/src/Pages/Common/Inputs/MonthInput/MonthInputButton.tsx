import { faCaretDown } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { linearFnFomPoints, Month } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { noCase } from '../../../../Utils/Utils'
import { useSimulation } from '../../../App/WithSimulation'
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
    const { value, range, toMFN, getMonthLabel } = props
    const label = getMonthLabel(value)
    const numericalComponent = useNumericalComponent(value)
    const inRange =
      _.inRange(toMFN(value), range.start, range.end + 1) &&
      (numericalComponent
        ? _.inRange(
            numericalComponent.asNumber,
            numericalComponent.range.start,
            numericalComponent.range.end,
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
        {numericalComponent && (
          <_Number
            className={`${classNameForNumberInput} ml-6 mt-2`}
            {...props}
            numericalComponent={numericalComponent}
          />
        )}
      </div>
    )
  },
)

const _Number = React.memo(
  ({
    className = '',
    onChange,
    toMFN,
    range,
    numericalComponent,
    modalTextInputOnMobile,
  }: MonthInputProps & {
    numericalComponent: Exclude<ReturnType<typeof useNumericalComponent>, null>
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
        range={numMonthRange}
        modalLabel={modalTextInputOnMobile ? modalLabel : null}
      />
    )
  },
)

const useNumericalComponent = (value: Month | { numMonths: number }) => {
  const { paramsExt } = useSimulation()
  const { pickPerson } = paramsExt

  if ('numMonths' in value) {
    return {
      asNumber: value.numMonths,
      fromNumber: (numMonths: number) => ({ numMonths }),
      range: { start: 0, end: Number.MAX_SAFE_INTEGER },
      modalLabel: 'Number of Months',
    }
  } else {
    switch (value.type) {
      case 'now':
      case 'namedAge':
        return null
      case 'numericAge':
        const { ages } = pickPerson(value.person)
        return {
          asNumber: value.ageInMonths,
          fromNumber: (ageInMonths: number): Month => ({
            ...value,
            ageInMonths,
          }),
          range: { start: ages.currentMonth, end: ages.maxMonth },
          modalLabel: `${
            value.person === 'person1' ? 'Your' : "Your Partner's"
          } Age`,
        }
      default:
        noCase(value)
    }
  }
}
