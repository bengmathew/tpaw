import {faCaretDown} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _, {capitalize} from 'lodash'
import React from 'react'
import {Year, YearRange} from '../../../../TPAWSimulator/TPAWParams'
import {extendTPAWParams} from '../../../../TPAWSimulator/TPAWParamsExt'
import {linearFnFomPoints} from '../../../../Utils/LinearFn'
import {assert, noCase} from '../../../../Utils/Utils'
import {useSimulation} from '../../../App/WithSimulation'
import {NumberInput} from '../NumberInput'
import {YearRangeEdgeInputProps} from './YearRangeEdgeInput'
import {yearRangeEdgeLabel} from './YearRangeEdgeLabel'

export const YearRangeEdgeInputButton = React.memo(
  ({
    className = '',
    onOpenPicker,
    setRef,
    ...props
  }: {
    onOpenPicker: () => void
    setRef: (x: HTMLDivElement) => void
  } & YearRangeEdgeInputProps) => {
    const {value, edge, range} = props
    const {params, paramsExt} = useSimulation()
    const {asYFN} = paramsExt
    const label = capitalize(
      yearRangeEdgeLabel(value, edge, 'numberIsBlank', params)
    )
    const inRange = _.inRange(
      asYFN(value)[edge],
      asYFN(range).start,
      asYFN(range).end + 1
    )
    const outOfOrder = edge === 'end' && asYFN(value).end < asYFN(value).start
    return (
      <div
        className={`${className} ${
          !inRange || outOfOrder ? 'text-errorFG' : ''
        }  gap-x-2 flex flex-wrap gap-y-2  items-center`}
        ref={setRef}
      >
        <button
          className={`flex items-center gap-x-2 py-1.5`}
          onClick={onOpenPicker}
        >
          <h2 className="inline-block text-left">{label}</h2>
          <FontAwesomeIcon icon={faCaretDown} />
        </button>
        <_Number {...props} />
      </div>
    )
  }
)

const _Number = React.memo(
  ({className = '', value, setValue, edge, range}: YearRangeEdgeInputProps) => {
    const {params} = useSimulation()
    const {pickPerson} = extendTPAWParams(params)
    const props = (() => {
      const fromYear = (getYear: (x: YearRange) => Year) => {
        const year = getYear(value)
        switch (year.type) {
          case 'now':
          case 'namedAge':
            return null
          default:
            const setNumValue = (x: number, value: YearRange) => {
              const year = getYear(value)
              assert(year.type === 'numericAge')
              year.age = x
              return value
            }
            const person = pickPerson(year.person)
            return {
              numValue: year.age,
              setNumValue,
              additionalClamp: (x: number) =>
                _.clamp(x, person.ages.current, person.ages.max),
            }
        }
      }
      const fromNumYears = () => {
        assert('numYears' in value)
        const setNumValue = (x: number, value: YearRange) => {
          assert('numYears' in value)
          value.numYears = x
          return value
        }
        return {
          numValue: value.numYears,
          setNumValue,
          additionalClamp: (x: number) => Math.max(1, x),
        }
      }
      switch (value.type) {
        case 'startAndEnd':
          return fromYear(x => {
            assert(x.type === 'startAndEnd')
            return x[edge]
          })
        case 'startAndNumYears':
          return edge === 'start'
            ? fromYear(x => {
                assert(x.type === 'startAndNumYears')
                return x[edge]
              })
            : fromNumYears()
        case 'endAndNumYears':
          return edge === 'end'
            ? fromYear(x => {
                assert(x.type === 'endAndNumYears')
                return x[edge]
              })
            : fromNumYears()
        default:
          noCase(value)
      }
    })()
    if (!props) return <></>

    const {numValue, setNumValue, additionalClamp} = props
    const {asYFN, minYear, maxYear} = extendTPAWParams(params)

    return (
      <NumberInput
        className={className}
        value={numValue}
        setValue={x => {
          const clone = _.cloneDeep(value)
          setNumValue(x, clone)
          setValue(clone)
        }}
        clamp={x => {
          const toYFN = linearFnFomPoints(
            0,
            asYFN(setNumValue(0, _.cloneDeep(value)))[edge],
            1,
            asYFN(setNumValue(1, _.cloneDeep(value)))[edge]
          )
          // Cannot assume start is < end because of endWithInterval.
          const range2 =
            edge === 'start'
              ? {
                  start: asYFN(range).start,
                  end: Math.min(asYFN(range).end, asYFN(value).end),
                }
              : {
                  start: Math.max(asYFN(range).start, asYFN(value).start),
                  end: asYFN(range).end,
                }
          const start = Math.round(toYFN.inverse(range2.start))
          const end = Math.round(toYFN.inverse(range2.end))
          return additionalClamp(
            _.clamp(x, Math.min(start, end), Math.max(start, end))
          )
        }}
        showPlusMinus
      />
    )
  }
)
