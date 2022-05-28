import {faCircle} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSolid} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useState} from 'react'
import {Year} from '../../../../TPAWSimulator/TPAWParams'
import {noCase} from '../../../../Utils/Utils'
import {useSimulation} from '../../../App/WithSimulation'
import {ContextMenu} from '../../Modal/ContextMenu'
import {YearInputProps} from './YearInput'

export const YearInputPick = React.memo(
  ({
    location,
    value,
    onChange: setValueIn,
    referenceElement,
    onClose,
    range,
    choices,
    toYFN,
  }: {
    referenceElement: HTMLElement
    onClose: () => void
  } & YearInputProps) => {
    const {params, paramsExt} = useSimulation()
    const {asYFN, pickPerson, years} = paramsExt

    const [person, setPerson] = useState(() =>
      'numYears' in value || value.type === 'now' ? 'person1' : value.person
    )
    const {ages: personAges} = pickPerson(person)

    const nullIfNotInRange = (year: Year) =>
      _.inRange(asYFN(year), range.start, range.end + 1) ? year : null

    const choiceNow = choices.includes('now')
      ? nullIfNotInRange(years.now)
      : null
    const choiceLastWorkingYear = choices.includes('lastWorkingYear')
      ? nullIfNotInRange(years.person(person).lastWorkingYear)
      : null
    const choiceRetirement =
      personAges.type !== 'retired' && choices.includes('retirement')
        ? nullIfNotInRange(years.person(person).retirement)
        : null

    const choiceMax = choices.includes('maxAge')
      ? nullIfNotInRange(years.person(person).max)
      : null

    const choiceAge = choices.includes('numericAge')
      ? years
          .person(person)
          .numericAge(
            _.clamp(
              personAges.current +
                _.clamp(toYFN(value), range.start, range.end),
              personAges.current,
              personAges.max
            )
          )
      : null

    // Expect YearRangeInput to override the actual value here.
    const choiceNumYears =
      choices.includes('forNumOfYears') && location !== 'standalone'
        ? {numYears: 1}
        : null

    const asAge = (year: Year) =>
      year.type === 'namedAge'
        ? `(Age ${asYFN(year) + pickPerson(year.person).ages.current})`
        : ''
    const {people} = params

    return (
      <ContextMenu
        referenceElement={referenceElement}
        onClose={onClose}
        darkBG={false}
      >
        {onHide => {
          const setValue = (x: Year | {numYears: number}) => {
            onHide()
            setValueIn(x)
          }
          return (
            <div className={` w-[100vw] sm:w-[300px] my-2`} onClick={() => {}}>
              {location !== 'standalone' && (
                <h2 className="px-3 font-bold py-2">
                  {location === 'rangeStart'
                    ? 'From'
                    : location === 'rangeEnd'
                    ? 'Until (and including)'
                    : noCase(location)}
                </h2>
              )}

              <div className="flex flex-col">
                {choiceNow && (
                  <button
                    className="text-left px-3 py-2"
                    onClick={() => setValue(choiceNow)}
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
                            icon={
                              person === 'person1' ? faCircleSolid : faCircle
                            }
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
                            icon={
                              person === 'person1' ? faCircle : faCircleSolid
                            }
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
                    {choiceLastWorkingYear && (
                      <button
                        className="text-left   px-3 py-2"
                        onClick={() => setValue(choiceLastWorkingYear)}
                      >
                        Last working year {`${asAge(choiceLastWorkingYear)}`}
                      </button>
                    )}
                    {choiceRetirement && (
                      <button
                        className="text-left  px-3 py-2"
                        onClick={() => setValue(choiceRetirement)}
                      >
                        Retirement {`${asAge(choiceRetirement)}`}
                      </button>
                    )}
                    {choiceMax && (
                      <button
                        className="text-left  px-3 py-2"
                        onClick={() => setValue(choiceMax)}
                      >
                        Max age {`${asAge(choiceMax)}`}
                      </button>
                    )}
                    {choiceAge && (
                      <button
                        className="text-left  px-3 py-2"
                        onClick={() => setValue(choiceAge)}
                      >
                        Age
                        <span className="inline-block border-b border-gray-700 w-[25px] ml-2" />
                      </button>
                    )}
                  </div>
                  {choiceNumYears && (
                    <button
                      className="text-left  px-3 py-2"
                      onClick={() => setValue(choiceNumYears)}
                    >
                      For number of years
                      <span className="inline-block border-b border-gray-700 w-[25px] ml-2" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          )
        }}
      </ContextMenu>
    )
  }
)
