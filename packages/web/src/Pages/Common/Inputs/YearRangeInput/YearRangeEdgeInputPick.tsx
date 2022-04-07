import {faCircle} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSolid} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useState} from 'react'
import {Year} from '../../../../TPAWSimulator/TPAWParams'
import {extendTPAWParams} from '../../../../TPAWSimulator/TPAWParamsExt'
import {fGet, noCase} from '../../../../Utils/Utils'
import {useSimulation} from '../../../App/WithSimulation'
import {ContextMenu} from '../../Modal/ContextMenu'
import {YearRangeEdgeInputProps} from './YearRangeEdgeInput'

export const YearRangeEdgeInputPick = React.memo(
  ({
    edge,
    value,
    setValue: setValueIn,
    referenceElement,
    onClose,
    range,
    choices: choicesIn,
  }: {
    referenceElement: HTMLElement
    onClose: () => void
  } & YearRangeEdgeInputProps) => {
    const {params} = useSimulation()
    const {asYFN, yearRangeLength, yearRangeClamp, pickPerson} =
      extendTPAWParams(params)
    const choices = choicesIn[edge]

    const [person, setPerson] = useState(() => {
      const edgeYear =
        'start' in value && edge === 'start'
          ? value.start
          : 'end' in value && edge === 'end'
          ? value.end
          : null
      return !edgeYear || edgeYear.type === 'now' ? 'person1' : edgeYear.person
    })
    const {ages: personAges} = pickPerson(person)

    const years = {
      now: {type: 'now'} as const,
      lastWorkingYear: {
        type: 'namedAge',
        person,
        age: 'lastWorkingYear',
      } as const,
      retirement:
        personAges.type === 'retired'
          ? null
          : ({type: 'namedAge', person, age: 'retirement'} as const),
      max: {type: 'namedAge', person, age: 'max'} as const,
      age: {
        type: 'numericAge',
        person,
        age: _.clamp(
          personAges.current + asYFN(yearRangeClamp(range, value))[edge],
          personAges.current,
          personAges.max
        ),
      } as const,
    }

    const asAge = (year: Extract<Year, {type: 'namedAge'}>) =>
      asYFN(year) + pickPerson(year.person).ages.current
    const {people} = params
    const setValue = (x: Year | 'duration') => {
      const yfnToAge = (yfn: number): Year =>
        yfn > people.person1.ages.max - people.person1.ages.current &&
        people.withPartner
          ? extendTPAWParams.years.person2.numericAge(
              yfn + people.person2.ages.current
            )
          : extendTPAWParams.years.person1.numericAge(
              yfn + people.person1.ages.current
            )

      if (x === 'duration') {
        if (edge === 'start') {
          const end = (() => {
            switch (value.type) {
              case 'startAndEnd':
              case 'endAndNumYears':
                return value.end
              case 'startAndNumYears':
                return yfnToAge(asYFN(value).end)
              default:
                noCase(value)
            }
          })()
          const duration = Math.max(1, yearRangeLength(value))
          const maxDuration = Math.max(1, asYFN(end) + 1 - asYFN(range.start))
          setValueIn({
            type: 'endAndNumYears',
            end,
            numYears: Math.min(maxDuration, duration),
          })
        } else {
          const start = (() => {
            switch (value.type) {
              case 'startAndEnd':
              case 'startAndNumYears':
                return value.start
              case 'endAndNumYears':
                return yfnToAge(asYFN(value).start)
              default:
                noCase(value)
            }
          })()
          const duration = Math.max(1, yearRangeLength(value))
          const maxDuration = Math.max(1, asYFN(range.end) + 1 - asYFN(start))
          setValueIn({
            type: 'startAndNumYears',
            start,
            numYears: Math.min(maxDuration, duration),
          })
        }
      } else {
        if (edge === 'start') {
          switch (value.type) {
            case 'startAndEnd':
            case 'startAndNumYears':
              setValueIn({...value, start: x})
              break
            case 'endAndNumYears':
              setValueIn({type: 'startAndEnd', end: value.end, start: x})
              break
            default:
              noCase(value)
          }
        } else {
          switch (value.type) {
            case 'startAndEnd':
            case 'endAndNumYears':
              setValueIn({...value, end: x})
              break
            case 'startAndNumYears':
              setValueIn({type: 'startAndEnd', start: value.start, end: x})
              break
            default:
              noCase(value)
          }
        }
      }
    }

    const inRange = (year: Year) =>
      _.inRange(asYFN(year), asYFN(range).start, asYFN(range).end + 1)

    return (
      <ContextMenu
        referenceElement={referenceElement}
        onClose={onClose}
        darkBG={false}
      >
        {onHide => (
          <div className={` w-[100vw] sm:w-[300px] my-2`} onClick={() => {}}>
            <h2 className="px-3 font-bold py-2">
              {edge === 'start' ? 'From' : 'Until (and including)'}
            </h2>

            <div className="flex flex-col">
              {inRange(years.now) && choices.includes('now') && (
                <button
                  className="text-left px-3 py-2"
                  onClick={() => {
                    onHide()
                    setValue(years.now)
                  }}
                >
                  Now
                </button>
              )}
              <div className="flex flex-col">
                {people.withPartner && (
                  <div className="flex  px-3 ">
                    <div className="flex items-center  gap-x-4  ">
                      <button
                        className={`flex items-center gap-x-1   py-2
                                      ${
                                        person === 'person1'
                                          ? ' bg-gray-30 '
                                          : ''
                                      }`}
                        onClick={() => setPerson('person1')}
                      >
                        <FontAwesomeIcon
                          className="text-[10px] mr-1"
                          icon={person === 'person1' ? faCircleSolid : faCircle}
                        />
                        Your
                      </button>
                      <button
                        className={`flex items-center gap-x-1   px-2 py-2 
                                      ${
                                        person === 'person1'
                                          ? ''
                                          : 'bg-gray-30  '
                                      }`}
                        onClick={() => setPerson('person2')}
                      >
                        <FontAwesomeIcon
                          className="text-[10px] mr-1"
                          icon={person === 'person1' ? faCircle : faCircleSolid}
                        />
                        {`Your partner's`}
                      </button>
                    </div>
                  </div>
                )}
                <div
                  className={`flex flex-col
              ${people.withPartner ? 'pl-8' : ''}`}
                >
                  {choices.includes('lastWorkingYear') &&
                    inRange(years.lastWorkingYear) && (
                      <button
                        className="text-left   px-3 py-2"
                        onClick={() => {
                          onHide()
                          setValue(years.lastWorkingYear)
                        }}
                      >
                        Last working year (Age{' '}
                        {`${asAge(years.lastWorkingYear)}`})
                      </button>
                    )}
                  {choices.includes('retirement') &&
                    years.retirement &&
                    inRange(years.retirement) && (
                      <button
                        className="text-left  px-3 py-2"
                        onClick={() => {
                          onHide()
                          setValue(fGet(years.retirement))
                        }}
                      >
                        Retirement (Age {`${asAge(years.retirement)}`})
                      </button>
                    )}
                  {choices.includes('maxAge') && inRange(years.max) && (
                    <button
                      className="text-left  px-3 py-2"
                      onClick={() => {
                        onHide()
                        setValue(years.max)
                      }}
                    >
                      Max age (Age {`${asAge(years.max)}`})
                    </button>
                  )}
                  {choices.includes('numericAge') && (
                    <button
                      className="text-left  px-3 py-2"
                      onClick={() => {
                        onHide()
                        setValue(years.age)
                      }}
                    >
                      Age
                      <span className="inline-block border-b border-gray-700 w-[25px] ml-2" />
                    </button>
                  )}
                </div>
                {choices.includes('forNumOfYears') && (
                  <button
                    className="text-left  px-3 py-2"
                    onClick={() => {
                      onHide()
                      setValue('duration')
                    }}
                  >
                    For number of years
                    <span className="inline-block border-b border-gray-700 w-[25px] ml-2" />
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </ContextMenu>
    )
  }
)
