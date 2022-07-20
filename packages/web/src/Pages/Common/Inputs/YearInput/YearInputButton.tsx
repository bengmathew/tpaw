import {faCaretDown} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _, {capitalize} from 'lodash'
import React from 'react'
import {Year} from '../../../../TPAWSimulator/TPAWParams'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {noCase} from '../../../../Utils/Utils'
import {useSimulation} from '../../../App/WithSimulation'
import {NumberInput} from '../NumberInput'
import {YearInputProps} from './YearInput'
import {yearLabel} from './YearLabel'

export const YearInputButton = React.memo(
  ({
    className = '',
    onOpenPicker,
    setRef,
    ...props
  }: {
    onOpenPicker: () => void
    setRef: (x: HTMLDivElement) => void
  } & YearInputProps) => {
    const {value, location, range, toYFN} = props
    const {params} = useSimulation()
    const label = capitalize(
      yearLabel(value, location, 'numberIsBlank', params)
    )
    const numericalComponent = useNumericalComponent(value)
    const inRange =
      _.inRange(toYFN(value), range.start, range.end + 1) &&
      (numericalComponent
        ? _.inRange(
            numericalComponent.asNumber,
            numericalComponent.range.start,
            numericalComponent.range.end
          )
        : true)
    return (
      <div
        className={`${className} 
        ${!inRange ? 'text-errorFG' : ''}  
        gap-x-2 flex flex-wrap gap-y-2  items-center`}
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
          <_Number {...props} numericalComponent={numericalComponent} />
        )}
      </div>
    )
  }
)

const _Number = React.memo(
  ({
    className = '',
    onChange,
    toYFN,
    range,
    numericalComponent,
    modalTextInputOnMobile
  }: YearInputProps & {
    numericalComponent: Exclude<ReturnType<typeof useNumericalComponent>, null>
  }) => {
    const {asNumber, fromNumber, modalLabel} = numericalComponent
    return (
      <NumberInput
        className={className}
        value={asNumber}
        setValue={x => onChange(fromNumber(x))}
        clamp={x => {
          const yfnToNumber = (x: number) =>
            Math.round(
              linearFnFomPoints(
                0,
                toYFN(fromNumber(0)),
                1,
                toYFN(fromNumber(1))
              ).inverse(x)
            )
          const edge1 = yfnToNumber(range.start)
          const edge2 = yfnToNumber(range.end)

          // Cannot assume edge1 < edge2 because of endWithInterval.
          const start = Math.min(edge1, edge2)
          const end = Math.max(edge1, edge2)
          return _.clamp(
            x,
            Math.max(start, numericalComponent.range.start),
            Math.min(end, numericalComponent.range.end)
          )
        }}
        showPlusMinus
        modalLabel={modalTextInputOnMobile? modalLabel: null}
      />
    )
  }
)

const useNumericalComponent = (value: Year | {numYears: number}) => {
  const {paramsExt} = useSimulation()
  const {pickPerson} = paramsExt

  if ('numYears' in value) {
    return {
      asNumber: value.numYears,
      fromNumber: (numYears: number) => ({numYears}),
      range: {start: 0, end: Number.MAX_SAFE_INTEGER},
      modalLabel:'Number of Years'
    }
  } else {
    switch (value.type) {
      case 'now':
      case 'namedAge':
        return null
      case 'numericAge':
        const {ages} = pickPerson(value.person)
        return {
          asNumber: value.age,
          fromNumber: (age: number) => ({...value, age}),
          range: {start: ages.current, end: ages.max},
          modalLabel:`${value.person === 'person1'? 'Your':'Your Partner\'s'} Age`
        }
      default:
        noCase(value)
    }
  }
}
