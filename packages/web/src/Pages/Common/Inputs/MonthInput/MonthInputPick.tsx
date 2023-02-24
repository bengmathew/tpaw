import { faCheck } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Month } from '@tpaw/common'
import _ from 'lodash'
import React, { useState } from 'react'
import { useSimulation } from '../../../App/WithSimulation'
import { ContextMenu } from '../../Modal/ContextMenu'
import { MonthInputProps } from './MonthInput'

export const MonthInputPick = React.memo(
  ({
    location,
    value,
    onChange: setValueIn,
    referenceElement,
    onClose,
    range,
    choices,
    toMFN,
    getMonthLabel,
  }: {
    referenceElement: HTMLElement
    onClose: () => void
  } & MonthInputProps) => {
    const { params, paramsExt } = useSimulation()
    const { asMFN, pickPerson, months } = paramsExt

    const [startingValue] = useState(value)
    const [person, setPerson] = useState(() =>
      'numMonths' in value || value.type === 'now' ? 'person1' : value.person,
    )
    const { ages: personAges } = pickPerson(person)

    const nullIfNotInRange = (month: Month) =>
      _.inRange(asMFN(month), range.start, range.end + 1) ? month : null

    const choicesForPerson = (person: 'person1' | 'person2') => ({
      lastWorkingMonth: choices.includes('lastWorkingMonth')
        ? nullIfNotInRange(months[person].lastWorkingMonth)
        : null,
      retirement:
        personAges.type !== 'retired' && choices.includes('retirement')
          ? nullIfNotInRange(months[person].retirement)
          : null,

      max: choices.includes('maxAge')
        ? nullIfNotInRange(months[person].max)
        : null,

      age: choices.includes('numericAge')
        ? months[person].numericAge(
            _.clamp(
              personAges.currentMonth +
                _.clamp(toMFN(value), range.start, range.end),
              personAges.currentMonth,
              personAges.maxMonth,
            ),
          )
        : null,
    })
    const choice = {
      now: choices.includes('now') ? nullIfNotInRange(months.now) : null,
      person1: choicesForPerson('person1'),
      person2: params.people.withPartner ? choicesForPerson('person2') : null,
      // Expect MonthRangeInput to override the actual value here.
      numMonths:
        choices.includes('forNumOfMonths') && location !== 'standalone'
          ? { numMonths: 1 }
          : null,
    } as const

    const title = getMonthLabel(months.now).split(' ')[0]
    const removeTitle = (x: string) => _.capitalize(x.replace(title, '').trim())

    return (
      <ContextMenu
        referenceElement={referenceElement}
        onClose={onClose}
        darkBG={false}
        align="left"
      >
        {(onHide) => {
          const setValue = (x: Month | { numMonths: number }) => {
            onHide()
            setValueIn(x)
          }
          const button = (
            value: Month | { numMonths: number } | null,
            numeric: 'numeric' | 'notNumeric',
            current: boolean,
          ) =>
            value && (
              <button
                className={`text-left py-2 hover:bg-gray-200 px-6 ${
                  current ? 'font-' : ''
                }`}
                onClick={() => setValue(value)}
              >
                {current && (
                  <span className="inline-block -ml-6 w-6">
                    <FontAwesomeIcon className="ml-1 text-sm lighten" icon={faCheck} />
                  </span>
                )}
                {removeTitle(getMonthLabel(value))}
                {numeric === 'numeric' && (
                  <span className="inline-block border-b border-gray-700 w-[25px] ml-2" />
                )}
              </button>
            )
          return (
            <div className={` w-[100vw] sm:w-[350px] my-2`} onClick={() => {}}>
              {location !== 'standalone' && (
                <h2 className="px-3 font-bold py-2">{title}</h2>
              )}

              <div className="flex flex-col">
                {button(
                  choice.now,
                  'notNumeric',
                  'type' in startingValue && startingValue.type === 'now',
                )}
                {/* {params.people.withPartner && (
                  <h2 className="font-bold mx-3 uppercase lighten-2 text-xs  py-2">
                    You
                  </h2>
                )} */}
                {button(
                  choice.person1.lastWorkingMonth,
                  'notNumeric',
                  'type' in startingValue &&
                    startingValue.type === 'namedAge' &&
                    startingValue.person === 'person1' &&
                    startingValue.age === 'lastWorkingMonth',
                )}
                {button(
                  choice.person1.retirement,
                  'notNumeric',
                  'type' in startingValue &&
                    startingValue.type === 'namedAge' &&
                    startingValue.person === 'person1' &&
                    startingValue.age === 'retirement',
                )}
                {button(
                  choice.person1.max,
                  'notNumeric',
                  'type' in startingValue &&
                    startingValue.type === 'namedAge' &&
                    startingValue.person === 'person1' &&
                    startingValue.age === 'max',
                )}
                {button(
                  choice.person1.age,
                  'numeric',
                  'type' in startingValue &&
                    startingValue.type === 'numericAge' &&
                    startingValue.person === 'person1',
                )}
                {/* {params.people.withPartner && (
                  <h2 className="font-bold mx-3 uppercase lighten-2 text-xs py-2">
                    Your Partner
                  </h2>
                )} */}
                {choice.person2 && (
                  <>
                    {button(
                      choice.person2.lastWorkingMonth,
                      'notNumeric',
                      'type' in startingValue &&
                        startingValue.type === 'namedAge' &&
                        startingValue.person === 'person2' &&
                        startingValue.age === 'lastWorkingMonth',
                    )}
                    {button(
                      choice.person2.retirement,
                      'notNumeric',
                      'type' in startingValue &&
                        startingValue.type === 'namedAge' &&
                        startingValue.person === 'person2' &&
                        startingValue.age === 'retirement',
                    )}
                    {button(
                      choice.person2.max,
                      'notNumeric',
                      'type' in startingValue &&
                        startingValue.type === 'namedAge' &&
                        startingValue.person === 'person2' &&
                        startingValue.age === 'max',
                    )}
                    {button(
                      choice.person2.age,
                      'numeric',
                      'type' in startingValue &&
                        startingValue.type === 'numericAge' &&
                        startingValue.person === 'person2',
                    )}
                  </>
                )}
                {choice.numMonths && (
                  <h2 className="px-3 font-bold py-2">Or</h2>
                )}

                {button(choice.numMonths, 'numeric', 'numMonths' in value)}

                {/* <div className="flex flex-col">
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
                    {choiceLastWorkingMonth && (
                      <button
                        className="text-left   px-3 py-2"
                        onClick={() => setValue(choiceLastWorkingMonth)}
                      >
                        Last working month
                      </button>
                    )}
                    {choiceRetirement && (
                      <button
                        className="text-left  px-3 py-2"
                        onClick={() => setValue(choiceRetirement)}
                      >
                        Retirement
                      </button>
                    )}
                    {choiceMax && (
                      <button
                        className="text-left  px-3 py-2"
                        onClick={() => setValue(choiceMax)}
                      >
                        Max age
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
                  {choiceNumMonths && (
                    <>
                      <h2 className="font-bold px-3 py-2">Or</h2>
                      <button
                        className="text-left  px-3 py-2"
                        onClick={() => setValue(choiceNumMonths)}
                      >
                        For a period of
                        <span className="inline-block border-b border-gray-700 w-[25px] ml-2" />
                      </button>
                    </>
                  )}
                </div> */}
              </div>
            </div>
          )
        }}
      </ContextMenu>
    )
  },
)
