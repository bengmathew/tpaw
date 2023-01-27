import { faTrash } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import {
  getDefaultPlanParams,
  MAX_AGE,
  Person,
  PlanParams,
  Year,
} from '@tpaw/common'
import _ from 'lodash'
import React, { ReactNode, useState } from 'react'
import {
  extendPlanParams,
  PlanParamsExt,
} from '../../../../TPAWSimulator/PlanParamsExt'
import { assert, noCase } from '../../../../Utils/Utils'
import { useSimulation } from '../../../App/WithSimulation'
import { CheckBox } from '../../../Common/Inputs/CheckBox'
import { NumberInput } from '../../../Common/Inputs/NumberInput'
import { ConfirmAlert } from '../../../Common/Modal/ConfirmAlert'
import { ValueForYearRangeDisplay } from '../../../Common/ValueForYearRangeDisplay'
import { analyzeYearsInParams } from '../Helpers/AnalyzeYearsInParams'
import { nextPlanSectionDialogPosition } from '../Helpers/PlanSectionDialogPosition'

export const PlanInputAgePerson = React.memo(
  ({
    type,
    className = '',
    style,
  }: {
    type: 'person1' | 'person2'
    className?: string
    style?: React.CSSProperties
  }) => {
    const { params, paramsExt, setParams } = useSimulation()
    const setPersonInParams = (params: PlanParams, person: Person) => {
      if (type === 'person1') {
        params.people.person1 = person
      } else {
        assert(params.people.withPartner)
        params.people.person2 = person
      }
    }
    const getPersonInParams = (params: PlanParams) => {
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
            setParams((params) => {
              const clone = _.cloneDeep(params)
              const { person1 } = clone.people
              // Validation won't trigger if strategy is not SPAW or SWR and the
              // glide path actually has a ref to person2. So remove those
              // entires here.
              clone.risk.spawAndSWR.allocation.intermediate =
                clone.risk.spawAndSWR.allocation.intermediate.filter(
                  ({ year }) =>
                    year.type === 'now' || year.person === 'person1',
                )
              clone.people = { withPartner: false, person1 }

              return _setXAxisDisplay(clone)
            })
          }

    const handleRetired = () => {
      setParams((params) => {
        const paramsExt = extendPlanParams(params)
        const clone = _.cloneDeep(params)
        const p = getPersonInParams(clone)
        const futureSavingsWarning = _futureSavingsWarning(paramsExt, type)
        if (futureSavingsWarning.length > 0) {
          setFutureSavingsWarning(futureSavingsWarning)
          return params
        }
        const retirementRemovalWarning = _retirementReferenceWarning(
          paramsExt,
          type,
        )
        if (retirementRemovalWarning.length > 0) {
          setRetirementRemovalWarning(retirementRemovalWarning)
          return params
        }
        if (clone.dialogPosition === 'future-savings') {
          const withdrawalStartPerson = !clone.people.withPartner
            ? 'person1'
            : clone.people.withdrawalStart
          if (type === withdrawalStartPerson) {
            clone.dialogPosition = nextPlanSectionDialogPosition(
              'future-savings',
              1, // 0 throws the calculations off.
            )
          }
        }
        p.ages = {
          type: 'retired',
          current: p.ages.current,
          max: p.ages.max,
        }

        return _setXAxisDisplay(clone)
      })
    }
    const [person2DeleteError, setPerson2DeleteError] = useState<ReactNode[]>(
      [],
    )
    const [futureSavingsWarning, setFutureSavingsWarning] = useState<
      ReactNode[]
    >([])
    const [retirementReferenceWarning, setRetirementRemovalWarning] = useState<
      ReactNode[]
    >([])
    return (
      <div className={`${className} `} style={style}>
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
              setEnabled={(retired) => {
                if (retired) {
                  handleRetired()
                  return
                }
                setParams((params) => {
                  const clone = _.cloneDeep(params)
                  const p = getPersonInParams(clone)
                  const defaultPerson1 = getDefaultPlanParams().people.person1
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
          style={{ grid: 'auto / 145px 1fr' }}
        >
          <h2 className="">Current Age</h2>
          <NumberInput
            value={person.ages.current}
            setValue={(value) =>
              setParams((params) => {
                const clone = _.cloneDeep(params)
                setPersonInParams(clone, _.cloneDeep(person))
                getPersonInParams(clone).ages.current = value
                return _setXAxisDisplay(clone)
              })
            }
            clamp={(value) =>
              _.clamp(
                value,
                0,
                person.ages.type === 'retired'
                  ? person.ages.max - 1
                  : person.ages.retirement - 1,
              )
            }
            showPlusMinus
            modalLabel="Current Age"
          />

          {person.ages.type === 'notRetired' && (
            <>
              <h2 className="">Retirement Age</h2>
              <NumberInput
                value={person.ages.retirement}
                setValue={(value) =>
                  setParams((params) => {
                    const clone = _.cloneDeep(params)
                    setPersonInParams(clone, _.cloneDeep(person))
                    const p = getPersonInParams(clone)
                    assert(p.ages.type === 'notRetired')
                    p.ages.retirement = value
                    return _setXAxisDisplay(clone)
                  })
                }
                clamp={(value) =>
                  _.clamp(value, person.ages.current + 1, person.ages.max - 1)
                }
                showPlusMinus
                modalLabel="Retirement Age"
              />
            </>
          )}

          <h2 className="">Max Age</h2>
          <NumberInput
            value={person.ages.max}
            setValue={(value) =>
              setParams((params) => {
                const clone = _.cloneDeep(params)
                setPersonInParams(clone, _.cloneDeep(person))
                getPersonInParams(clone).ages.max = value
                return _setXAxisDisplay(clone)
              })
            }
            clamp={(value) =>
              _.clamp(
                value,
                person.ages.type === 'retired'
                  ? person.ages.current + 1
                  : person.ages.retirement + 1,
                MAX_AGE,
              )
            }
            showPlusMinus
            modalLabel="Max Age"
          />
        </div>
        {person2DeleteError.length > 0 && (
          <ConfirmAlert
            title={'Error Removing Partner'}
            isWarningTitle
            option1={{
              onClose: () => setPerson2DeleteError([]),
              label: 'Close',
              isWarning: true,
            }}
            onCancel={null}
          >
            {person2DeleteError}
          </ConfirmAlert>
        )}
        {futureSavingsWarning.length > 0 && (
          <ConfirmAlert
            title={'Warning'}
            isWarningTitle
            option1={{
              onClose: () => {
                setFutureSavingsWarning([])
                setParams((params) => {
                  const clone = _.cloneDeep(params)
                  clone.wealth.futureSavings = []
                  return clone
                })
                handleRetired()
              },
              label: 'Delete Entries',
              isWarning: true,
            }}
            onCancel={() => setFutureSavingsWarning([])}
          >
            {futureSavingsWarning}
          </ConfirmAlert>
        )}
        {retirementReferenceWarning.length > 0 && (
          <ConfirmAlert
            title={'Warning'}
            isWarningTitle
            option1={{
              label: 'Convert References',
              onClose: () => {
                setRetirementRemovalWarning([])
                setParams((params) => {
                  const clone = _.cloneDeep(params)
                  analyzeYearsInParams(
                    extendPlanParams(clone),
                    (year: Year): Year => {
                      if (
                        year.type === 'namedAge' &&
                        year.person === type &&
                        (year.age === 'retirement' ||
                          year.age === 'lastWorkingYear')
                      ) {
                        return { type: 'now' }
                      } else {
                        return year
                      }
                    },
                  )

                  const p = getPersonInParams(clone)
                  p.ages = {
                    type: 'retired',
                    current: p.ages.current,
                    max: p.ages.max,
                  }
                  return clone
                })
              },
              isWarning: true,
            }}
            onCancel={() => setRetirementRemovalWarning([])}
          >
            {retirementReferenceWarning}
          </ConfirmAlert>
        )}
      </div>
    )
  },
)

const _person2RequiredMessage = (params: PlanParamsExt) => {
  const analysis = analyzeYearsInParams(params)
  const fromRanges = analysis.valueForYearRange.filter((x) => x.usesPerson2)

  const fromGlidePath = analysis.glidePath.filter((x) => x.usesPerson2)

  if (fromRanges.length === 0 || fromGlidePath.length === 0) return []
  const result = [
    `The following are specified in terms of your partner's age:`,
  ] as ReactNode[]
  result.push(
    <ol className=" list-disc list-outside ml-10">
      {fromRanges.map((x, i) => (
        <li key={`range-${i}`} className="list-item">
          <ValueForYearRangeDisplay entry={x.entry} range={null} />
        </li>
      ))}
      {fromGlidePath.map((x, i) => (
        <li key={`glidePath-${i}`} className="list-item">
          Asset allocation for the{' '}
          {x.location === 'assetAllocationForSPAW'
            ? 'savings portfolio approach'
            : noCase(x.location)}{' '}
          in the {`"${x.sectionLabel}"`} section.
        </li>
      ))}
    </ol>,
    `Remove these references to your partner's age before removing your partner.`,
  )
  return result
}

const _futureSavingsWarning = (
  paramsExt: PlanParamsExt,
  person: 'person1' | 'person2',
) => {
  const result = [] as ReactNode[]
  const { people, wealth } = paramsExt.params
  const { futureSavings } = wealth
  const otherPerson = people.withPartner
    ? person === 'person1'
      ? people.person2
      : people.person1
    : null
  const bothWillBeRetired = !otherPerson || otherPerson.ages.type === 'retired'
  if (!bothWillBeRetired || futureSavings.length === 0) return result

  result.push(
    `The Future Savings section is not applicable once ${
      people.withPartner ? 'both you and your partner' : 'you'
    } are retired but has the following ${
      futureSavings.length === 1 ? 'entry' : `${futureSavings.length} entries`
    }:`,
    <ol className=" list-decimal list-outside ml-10 ">
      {futureSavings.map((x, i) => (
        <li key={i} className="list-item">
          <ValueForYearRangeDisplay entry={x} range={null} />
        </li>
      ))}
    </ol>,
    'Would you like to delete these entries?',
  )
  return result
}

const _retirementReferenceWarning = (
  paramsExt: PlanParamsExt,
  person: 'person1' | 'person2',
) => {
  const analysis = analyzeYearsInParams(paramsExt)
  const fromRanges = analysis.valueForYearRange.filter((x) =>
    x.useRetirement(person),
  )
  const fromGlidePath = analysis.glidePath.filter((x) =>
    x.usesRetirement(person),
  )

  if (fromRanges.length === 0 && fromGlidePath.length === 0) return []

  const result = [
    `The following are specified in terms of ${
      person === 'person1' ? 'your' : "your partner's"
    } retirement age:`,
  ] as ReactNode[]
  result.push(
    <ul className="list-disc list-outside ml-10">
      {fromRanges.map((x, i) => (
        <li key={`ranges-${i}`} className="list-item">
          <ValueForYearRangeDisplay entry={x.entry} range={null} />
        </li>
      ))}
      {fromGlidePath.map((x, i) => (
        <li key={`glidePath-${i}`} className="list-item">
          Asset allocation for the{' '}
          {x.location === 'assetAllocationForSPAW'
            ? 'savings portfolio approach'
            : noCase(x.location)}{' '}
          in the {`"${x.sectionLabel}"`} section.
        </li>
      ))}
    </ul>,
    `Would your like to convert these references to ${
      person === 'person1' ? 'your' : "your partner's"
    } retirement to "now"?`,
  )
  return result
}

const _setXAxisDisplay = (params: PlanParams) => {
  if (!params.people.withPartner) return params
  const { asYFN, years } = extendPlanParams(params)
  params.people.xAxis =
    asYFN(years.person1.max) >= asYFN(years.person2.max) ? 'person1' : 'person2'
  return params
}
