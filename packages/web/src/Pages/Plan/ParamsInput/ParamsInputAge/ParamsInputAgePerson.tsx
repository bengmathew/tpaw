import { faGameConsoleHandheld } from '@fortawesome/pro-light-svg-icons'
import {faTrash} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {Switch} from '@headlessui/react'
import _ from 'lodash'
import React, {ReactNode, useState} from 'react'
import {getDefaultParams} from '../../../../TPAWSimulator/DefaultParams'
import {
  MAX_AGE,
  Person,
  TPAWParams,
  Year,
} from '../../../../TPAWSimulator/TPAWParams'
import {
  extendTPAWParams,
  TPAWParamsExt,
} from '../../../../TPAWSimulator/TPAWParamsExt'
import {assert} from '../../../../Utils/Utils'
import {useSimulation} from '../../../App/WithSimulation'
import {CheckBox} from '../../../Common/Inputs/CheckBox'
import {NumberInput} from '../../../Common/Inputs/NumberInput'
import {ConfirmAlert} from '../../../Common/Modal/ConfirmAlert'
import {ValueForYearRangeDisplay} from '../../../Common/ValueForYearRangeDisplay'
import {mapAllValueForYearsRangesInParams} from '../Helpers/MapAllValueForYearRangesInParams'
import {paramsInputValidateYearRange} from '../Helpers/ParamInputValidate'

export const ParamsInputAgePerson = React.memo(
  ({
    type,
    className = '',
  }: {
    type: 'person1' | 'person2'
    className?: string
  }) => {
    const {params, paramsExt, setParams} = useSimulation()
    const setPersonInParams = (params: TPAWParams, person: Person) => {
      if (type === 'person1') {
        params.people.person1 = person
      } else {
        assert(params.people.withPartner)
        params.people.person2 = person
      }
    }
    const getPersonInParams = (params: TPAWParams) => {
      if (type === 'person1') return params.people.person1
      assert(params.people.withPartner)
      return params.people.person2
    }

    const person = getPersonInParams(params)
    const handleDeletePerson =
      type === 'person1'
        ? null
        : () => {
            const errMessage = _person2RequiredMessage(paramsExt)
            if (errMessage.length > 0) {
              setPerson2DeleteError(errMessage)
              return
            }
            setParams(params => {
              const clone = _.cloneDeep(params)
              const {person1} = clone.people
              clone.people = {withPartner: false, person1}
              return _setXAxisDisplay(clone)
            })
          }

    const handleRetired = () => {
      setParams(params => {
        const paramsExt = extendTPAWParams(params)
        const clone = _.cloneDeep(params)
        const p = getPersonInParams(clone)
        const futureSavingsWarning = _futureSavingsWarning(paramsExt, type)
        if (futureSavingsWarning.length > 0) {
          setFutureSavingsWarning(futureSavingsWarning)
          return params
        }
        const retirementRemovalWarning = _retirementReferenceWarning(
          paramsExt,
          type
        )
        if (retirementRemovalWarning.length > 0) {
          setRetirementRemovalWarning(retirementRemovalWarning)
          return params
        } else {
          p.ages = {
            type: 'retired',
            current: p.ages.current,
            max: p.ages.max,
          }
        }
        return _setXAxisDisplay(clone)
      })
    }
    const [person2DeleteError, setPerson2DeleteError] = useState<ReactNode[]>(
      []
    )
    const [futureSavingsWarning, setFutureSavingsWarning] = useState<
      ReactNode[]
    >([])
    const [retirementReferenceWarning, setRetirementRemovalWarning] = useState<
      ReactNode[]
    >([])
    return (
      <div className={`${className} `}>
        <div className="flex justify-between">
          <h2 className="font-bold text-lg">
            {type === 'person1' ? 'You' : 'Your Partner'}
          </h2>
          {handleDeletePerson && (
            <button className="" onClick={handleDeletePerson}>
              <FontAwesomeIcon className="mr-2" icon={faTrash} />
            </button>
          )}
        </div>
        <div className="flex items-center col-span-2 gap-x-4 mt-4 mb-3">
          <Switch.Group>
            <Switch.Label className="">
              {type === 'person1'
                ? 'Are you retired?'
                : 'Is your partner retired?'}
            </Switch.Label>
            <CheckBox
              className=""
              enabled={person.ages.type === 'retired'}
              setEnabled={retired => {
                if (retired) {
                  handleRetired()
                  return
                }
                setParams(params => {
                  const clone = _.cloneDeep(params)
                  const p = getPersonInParams(clone)
                  const defaultPerson1 = getDefaultParams().people.person1
                  assert(defaultPerson1.ages.type === 'notRetired')
                  const defaultRetirementAge = defaultPerson1.ages.retirement
                  const retirement =
                    defaultRetirementAge <= p.ages.current
                      ? Math.floor((p.ages.max + p.ages.current) / 2)
                      : defaultRetirementAge
                  p.ages = {
                    type: 'notRetired',
                    current: p.ages.current,
                    max: p.ages.max,
                    retirement,
                  }
                  return _setXAxisDisplay(clone)
                })
              }}
            />
          </Switch.Group>
        </div>

        <div
          className="grid gap-y-2 items-center gap-x-4"
          style={{grid: 'auto / 145px 1fr'}}
        >
          <h2 className="">Current Age</h2>
          <NumberInput
            value={person.ages.current}
            setValue={value =>
              setParams(params => {
                const clone = _.cloneDeep(params)
                setPersonInParams(clone, _.cloneDeep(person))
                getPersonInParams(clone).ages.current = value
                return _setXAxisDisplay(clone)
              })
            }
            clamp={value =>
              _.clamp(
                value,
                0,
                person.ages.type === 'retired'
                  ? person.ages.max - 1
                  : person.ages.retirement - 1
              )
            }
            showPlusMinus
          />

          {person.ages.type === 'notRetired' && (
            <>
              <h2 className="">Retirement Age</h2>
              <NumberInput
                value={person.ages.retirement}
                setValue={value =>
                  setParams(params => {
                    const clone = _.cloneDeep(params)
                    setPersonInParams(clone, _.cloneDeep(person))
                    const p = getPersonInParams(clone)
                    assert(p.ages.type === 'notRetired')
                    p.ages.retirement = value
                    return _setXAxisDisplay(clone)
                  })
                }
                clamp={value =>
                  _.clamp(value, person.ages.current + 1, person.ages.max - 1)
                }
                showPlusMinus
              />
            </>
          )}

          <h2 className="">Max Age</h2>
          <NumberInput
            value={person.ages.max}
            setValue={value =>
              setParams(params => {
                const clone = _.cloneDeep(params)
                setPersonInParams(clone, _.cloneDeep(person))
                getPersonInParams(clone).ages.max = value
                return _setXAxisDisplay(clone)
              })
            }
            clamp={value =>
              _.clamp(
                value,
                person.ages.type === 'retired'
                  ? person.ages.current + 1
                  : person.ages.retirement + 1,
                MAX_AGE
              )
            }
            showPlusMinus
          />
        </div>
        {person2DeleteError.length > 0 && (
          <ConfirmAlert
            title={'Error Removing Partner'}
            isWarningTitle
            isWarningButton
            confirmText={'Close'}
            onCancel={null}
            onConfirm={() => setPerson2DeleteError([])}
          >
            {person2DeleteError}
          </ConfirmAlert>
        )}
        {futureSavingsWarning.length > 0 && (
          <ConfirmAlert
            title={'Warning'}
            isWarningTitle
            isWarningButton
            confirmText={'Delete Entries'}
            onCancel={() => setFutureSavingsWarning([])}
            onConfirm={() => {
              setFutureSavingsWarning([])
              setParams(params => {
                const clone = _.cloneDeep(params)
                clone.savings = []
                return clone
              })
              handleRetired()
            }}
          >
            {futureSavingsWarning}
          </ConfirmAlert>
        )}
        {retirementReferenceWarning.length > 0 && (
          <ConfirmAlert
            title={'Warning'}
            isWarningTitle
            isWarningButton
            confirmText={'Convert References'}
            onCancel={() => setRetirementRemovalWarning([])}
            onConfirm={() => {
              setRetirementRemovalWarning([])
              setParams(params => {
                const clone = _.cloneDeep(params)
                mapAllValueForYearsRangesInParams(
                  clone,
                  paramsInputValidateYearRange,
                  (year: Year): Year => {
                    if (
                      year.type === 'namedAge' &&
                      year.person === type &&
                      (year.age === 'retirement' ||
                        year.age === 'lastWorkingYear')
                    ) {
                      return {type: 'now'}
                    } else {
                      return year
                    }
                  }
                )

                const p = getPersonInParams(clone)
                p.ages = {
                  type: 'retired',
                  current: p.ages.current,
                  max: p.ages.max,
                }
                return clone
              })
            }}
          >
            {retirementReferenceWarning}
          </ConfirmAlert>
        )}
      </div>
    )
  }
)

const _person2RequiredMessage = (params: TPAWParamsExt) => {
  const fromRanges = mapAllValueForYearsRangesInParams(
    params.params,
    paramsInputValidateYearRange
  ).filter(x => x.usesPerson2)

  if (fromRanges.length === 0) return []
  const result = [
    `The following years are specified in terms of your partner's age:`,
  ] as ReactNode[]
  result.push(
    <ol className="list-decimal list-outside ml-10">
      {fromRanges.map((x, i) => (
        <li key={i} className="list-item">
          <ValueForYearRangeDisplay entry={x.entry} range={null} />
        </li>
      ))}
    </ol>,
    `Remove these references to your partner's age before removing your partner.`
  )
  return result
}

const _futureSavingsWarning = (
  paramsExt: TPAWParamsExt,
  person: 'person1' | 'person2'
) => {
  const result = [] as ReactNode[]
  const {people, savings} = paramsExt.params
  const otherPerson = people.withPartner
    ? person === 'person1'
      ? people.person2
      : people.person1
    : null
  const bothWillBeRetired = !otherPerson || otherPerson.ages.type === 'retired'
  if (!bothWillBeRetired || savings.length === 0) return result

  result.push(
    `The Future Savings section is not applicable once ${
      people.withPartner ? 'both you and your partner' : 'you'
    } are retired but has the following ${
      savings.length === 1 ? 'entry' : `${savings.length} entries`
    }:`,
    <ol className=" list-decimal list-outside ml-10 ">
      {savings.map((x, i) => (
        <li key={i} className="list-item">
          <ValueForYearRangeDisplay entry={x} range={null} />
        </li>
      ))}
    </ol>,
    'Would you like to delete these entries?'
  )
  return result
}

const _retirementReferenceWarning = (
  paramsExt: TPAWParamsExt,
  person: 'person1' | 'person2'
) => {
  const fromRanges = mapAllValueForYearsRangesInParams(
    paramsExt.params,
    paramsInputValidateYearRange
  ).filter(x => x.useRetirement(person))

  if (fromRanges.length === 0) return []

  const result = [
    `The following years are specified in terms of ${
      person === 'person1' ? 'your' : "your partner's"
    } retirement age:`,
  ] as ReactNode[]
  result.push(
    <ol className="list-decimal list-outside ml-10">
      {fromRanges.map((x, i) => (
        <li key={i} className="list-item">
          <ValueForYearRangeDisplay entry={x.entry} range={null} />
        </li>
      ))}
    </ol>,
    `Would your like to convert these references to ${
      person === 'person1' ? 'your' : "your partner's"
    } retirement to "now"?`
  )
  return result
}

const _setXAxisDisplay = (params: TPAWParams) => {
  if (!params.people.withPartner) return params
  const {asYFN, years} = extendTPAWParams(params)
  params.people.xAxis =
    asYFN(years.person1.max) >= asYFN(years.person2.max) ? 'person1' : 'person2'
  return params
}
