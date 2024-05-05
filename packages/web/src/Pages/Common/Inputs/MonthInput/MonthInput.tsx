import {
  faCaretDown,
  faCheck,
  faExclamationCircle,
  faTurnDownLeft,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Listbox } from '@headlessui/react'
import {
  CalendarMonth,
  CalendarMonthFns,
  Month,
  block,
  fGet,
  linearFnFromPointAndSlope,
  noCase,
} from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import {
  NormalizedAges,
  getFromMFNToNumericAge,
  getMonthToMFN,
} from '../../../../UseSimulator/NormalizePlanParams/NormalizeAges'
import {
  NormalizedMonthInThePast,
  NormalizedMonthNotInThePast,
} from '../../../../UseSimulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizedMonth'
import { InMonthsFns } from '../../../../Utils/InMonthsFns'
import { Record } from '../../../../Utils/Record'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { useSimulation } from '../../../PlanRoot/PlanRootHelpers/WithSimulation'
import { ContextModal } from '../../Modal/ContextModal'
import { getNormalizedMonthStr } from '../../MonthOrDurationDisplay'
import { CalendarMonthInput } from './CalendarMonthInput'
import { InMonthsInput } from './InMonthsInput'

export type MonthInputProps = React.ComponentProps<typeof MonthInput>

export type MonthType =
  | 'now'
  | 'lastWorkingMonth'
  | 'retirement'
  | 'maxAge'
  | 'numericAge'
  | 'calendarMonth'

// No assumptions are made about the month. In particular this supports months
// that have no validRangeAsMFN, are in the past, and out of range.
export const MonthInput = React.memo(
  ({
    className,
    normValue,
    onChange,
    choicesPreFilter,
    modalTextInputOnMobile,
  }: {
    className?: string
    normValue: NormalizedMonthNotInThePast | NormalizedMonthInThePast
    onChange: (month: Month) => void
    choicesPreFilter: MonthType[]
    modalTextInputOnMobile: boolean
  }) => {
    const { planParamsNorm } = useSimulation()

    const choices = useMemo(
      () => _processChoices(normValue, planParamsNorm, choicesPreFilter),
      [choicesPreFilter, planParamsNorm, normValue],
    )
    const currChoiceIndex = useMemo(() => {
      const _getMonthStructure = (month: Month) => {
        switch (month.type) {
          case 'now':
            return { type: 'now' }
          case 'namedAge':
            return month
          case 'calendarMonth':
            return { type: 'calendarMonth' }
          case 'numericAge':
            return { type: 'numericAge', person: month.person }
          default:
            noCase(month)
        }
      }
      return normValue.isInThePast
        ? -1
        : choices.findIndex((x) =>
            _.isEqual(
              _getMonthStructure(x),
              _getMonthStructure(normValue.baseValue),
            ),
          )
    }, [choices, normValue])

    return (
      <div className={clsx(className, normValue.errorMsg && ' text-errorFG')}>
        <Listbox
          value={currChoiceIndex}
          onChange={(choiceIndex) => {
            onChange(fGet(choices[choiceIndex]))
          }}
        >
          {({ open }) => (
            <ContextModal
              align="left"
              open={open}
              onOutsideClickOrEscape={null}
            >
              {({ ref }) => (
                <Listbox.Button className="py-1.5" ref={ref}>
                  {getNormalizedMonthStr(normValue).forMonthTypeMenu.label}
                  <FontAwesomeIcon className="ml-2" icon={faCaretDown} />
                </Listbox.Button>
              )}
              <Listbox.Options className="context-menu-outer-div">
                {choices.map((choice, i) => {
                  const { label, dash } = getNormalizedMonthStr({
                    isInThePast: false,
                    baseValue: choice,
                  }).forMonthTypeMenu
                  return (
                    <Listbox.Option
                      key={i}
                      className="context-menu-item"
                      value={i}
                    >
                      {({ selected }) => (
                        <div className="flex">
                          <div className="w-[25px]">
                            {selected && <FontAwesomeIcon icon={faCheck} />}
                          </div>
                          {label}{' '}
                          {dash && (
                            <span className="inline-block border-b border-gray-700 w-[25px] ml-2 mb-1" />
                          )}
                        </div>
                      )}
                    </Listbox.Option>
                  )
                })}
              </Listbox.Options>
            </ContextModal>
          )}
        </Listbox>
        {block(() => {
          if (normValue.isInThePast) {
            return <></>
          }
          const { baseValue, asMFN, validRangeAsMFN } = normValue
          switch (baseValue.type) {
            case 'now':
            case 'namedAge':
              return <></>
            case 'calendarMonth':
              return (
                <CalendarMonthInput
                  className="mt-1 mb-2"
                  normValue={{
                    baseValue: baseValue.calendarMonth,
                    validRangeAsMFN,
                  }}
                  onChange={(calendarMonth) => {
                    onChange({ type: 'calendarMonth', calendarMonth })
                  }}
                />
              )
            case 'numericAge':
              return (
                <InMonthsInput
                  className="mt-1 mb-2"
                  modalLabel={
                    modalTextInputOnMobile ? 'Number of Months' : null
                  }
                  normValue={{
                    baseValue: baseValue.age,
                    validRangeInMonths: {
                      includingLocalConstraints:
                        validRangeAsMFN.includingLocalConstraints
                          ? Record.mapValues(
                              validRangeAsMFN.includingLocalConstraints,
                              InMonthsFns.getFromMFN(baseValue.age, asMFN, 1),
                            )
                          : null,
                    },
                  }}
                  onChange={(inMonths) => {
                    onChange({
                      type: 'numericAge',
                      person: baseValue.person,
                      age: inMonths,
                    })
                  }}
                />
              )
            default:
              noCase(baseValue)
          }
        })}
        {normValue.errorMsg && (
          <div>
            <FontAwesomeIcon className="" icon={faExclamationCircle} />{' '}
            {normValue.errorMsg}
          </div>
        )}
      </div>
    )
  },
)

// These choices are not guaranteed to lead to error free months (but they will
// pass planParamsGuard()). When there is a numerical component (eg. numericAge)
// the numeric value is set to prioritize stability with respect to the current
// month. This can be outside the validRangeAsMFN and also past max age.
const _processChoices = (
  value:
    | {
        isInThePast: false
        asMFN: number
        validRangeAsMFN: NormalizedMonthInThePast['validRangeAsMFN']
      }
    | {
        isInThePast: true
        validRangeAsMFN: NormalizedMonthInThePast['validRangeAsMFN']
      },
  {
    ages,
    datingInfo,
  }: {
    datingInfo: PlanParamsNormalized['datingInfo']
    ages: NormalizedAges
  },
  choicesPreFilter: MonthType[],
) => {
  const monthToMFN = getMonthToMFN(datingInfo.nowAsCalendarDay, ages)
  const preferredMonthAsMFN = value.isInThePast
    ? value.validRangeAsMFN.includingLocalConstraints
      ? value.validRangeAsMFN.includingLocalConstraints.start
      : value.validRangeAsMFN.excludingLocalConstraints
        ? value.validRangeAsMFN.excludingLocalConstraints.start
        : 0
    : value.asMFN
  return _.sortBy(
    _.compact(
      _.flatten(
        choicesPreFilter.map((choice): (Month | null)[] => {
          switch (choice) {
            case 'now':
              return [
                {
                  type: 'now',
                  monthOfEntry: datingInfo.isDated
                    ? {
                        isDatedPlan: true,
                        calendarMonth: datingInfo.nowAsCalendarDay,
                      }
                    : { isDatedPlan: false },
                },
              ]
            case 'lastWorkingMonth':
              return [
                ages.person1.retirement.isRetired
                  ? null
                  : {
                      type: 'namedAge',
                      person: 'person1',
                      age: 'lastWorkingMonth',
                    },
                !ages.person2 || ages.person2.retirement.isRetired
                  ? null
                  : {
                      type: 'namedAge',
                      person: 'person2',
                      age: 'lastWorkingMonth',
                    },
              ]
            case 'retirement':
              return [
                ages.person1.retirement.isRetired
                  ? null
                  : {
                      type: 'namedAge',
                      person: 'person1',
                      age: 'retirement',
                    },
                !ages.person2 || ages.person2.retirement.isRetired
                  ? null
                  : {
                      type: 'namedAge',
                      person: 'person2',
                      age: 'retirement',
                    },
              ]
            case 'maxAge':
              return [
                { type: 'namedAge', person: 'person1', age: 'max' },
                ages.person2
                  ? { type: 'namedAge', person: 'person2', age: 'max' }
                  : null,
              ]
            case 'calendarMonth': {
              return [
                datingInfo.isDated
                  ? {
                      type: 'calendarMonth',
                      calendarMonth: CalendarMonthFns.getFromMFN(
                        datingInfo.nowAsCalendarDay,
                      )(preferredMonthAsMFN),
                    }
                  : null,
              ]
            }
            case 'numericAge': {
              // This can be out of range (with respect to the max age, and
              // validRangeAsMFN), but that's ok. Choices here do not have to
              // lead to error free months.
              const mfnToNumericAge = getFromMFNToNumericAge({ ages })
              return [
                mfnToNumericAge.person1(preferredMonthAsMFN),
                mfnToNumericAge.person2?.(preferredMonthAsMFN) ?? null,
              ]
            }
            default:
              noCase(choice)
          }
        }),
      ),
    ).filter((x) => {
      switch (x.type) {
        case 'now':
        case 'namedAge':
          return SimpleRange.Closed.isIn(
            monthToMFN.fNotInPast(x),
            value.validRangeAsMFN.excludingLocalConstraints,
          )
        case 'calendarMonth':
          return true
        case 'numericAge':
          // Don't allow if there is no age for this person that in in range.
          return (
            SimpleRange.Closed.intersection(
              value.validRangeAsMFN.excludingLocalConstraints,
              {
                start: 0,
                end: monthToMFN.fNotInPast({
                  type: 'namedAge',
                  person: x.person,
                  age: 'max',
                }),
              },
            ) !== null
          )
        default:
          noCase(x)
      }
    }),
    (x) => {
      switch (x.type) {
        case 'now':
          return 0
        case 'namedAge':
          const personOffset = x.person === 'person1' ? 1 : 2
          switch (x.age) {
            case 'lastWorkingMonth':
              return personOffset + 0.1
            case 'retirement':
              return personOffset + 0.2
            // 0.3 for numeric age
            case 'max':
              return personOffset + 0.4
          }
        case 'numericAge':
          return x.person === 'person1' ? 1.3 : 2.3
        case 'calendarMonth':
          return 3
        default:
          noCase(x)
      }
    },
  )
}

export const _forTesting = {
  _processChoices,
}
