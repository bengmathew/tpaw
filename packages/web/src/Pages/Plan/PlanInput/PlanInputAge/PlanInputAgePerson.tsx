import {
  faCaretDown,
  faCaretRight,
  faTrash,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import {
  getDefaultPlanParams,
  MAX_AGE_IN_MONTHS,
  Month,
  Person,
  PlanParams,
} from '@tpaw/common'
import _ from 'lodash'
import React, { ReactNode, useRef, useState } from 'react'
import {
  extendPlanParams,
  PlanParamsExt,
} from '../../../../TPAWSimulator/PlanParamsExt'
import { numMonthsStr } from '../../../../Utils/NumMonthsStr'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { assert, noCase } from '../../../../Utils/Utils'
import { yourOrYourPartners } from '../../../../Utils/YourOrYourPartners'
import { useSimulation } from '../../../App/WithSimulation'
import { CheckBox } from '../../../Common/Inputs/CheckBox'
import { NumMonthsInput } from '../../../Common/Inputs/NumMonthsInput'
import { ConfirmAlert } from '../../../Common/Modal/ConfirmAlert'
import { ValueForMonthRangeDisplay } from '../../../Common/ValueForMonthRangeDisplay'
import { analyzeMonthsInParams } from '../Helpers/AnalyzeMonthsInParams'
import { nextPlanSectionDialogPosition } from '../Helpers/PlanSectionDialogPosition'
import { PlanInputAgeOpenableSection } from './PlanInputAge'

export const PlanInputAgePerson = React.memo(
  ({
    personType,
    className = '',
    style,
    openSection,
    setOpenSection,
  }: {
    personType: 'person1' | 'person2'
    className?: string
    style?: React.CSSProperties
    openSection: PlanInputAgeOpenableSection
    setOpenSection: (x: PlanInputAgeOpenableSection) => void
  }) => {
    const { params, paramsExt, setParams } = useSimulation()
    const divRef = useRef<HTMLDivElement>(null)

    const getPersonInParams = (params: PlanParams) => {
      if (personType === 'person1') return params.people.person1
      assert(params.people.withPartner)
      return params.people.person2
    }
    const ages = getPersonInParams(params).ages
    const handleDeletePerson =
      personType === 'person1'
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
                  ({ month }) =>
                    month.type === 'now' || month.person === 'person1',
                )
              clone.people = { withPartner: false, person1 }

              return clone
            })
          }

    const handleRetired = () => {
      setParams((params) => {
        const paramsExt = extendPlanParams(params)
        const clone = _.cloneDeep(params)
        const p = getPersonInParams(clone)
        const futureSavingsWarning = _futureSavingsWarning(
          paramsExt,
          personType,
        )
        if (futureSavingsWarning.length > 0) {
          setFutureSavingsWarning(futureSavingsWarning)
          return params
        }
        const retirementRemovalWarning = _retirementReferenceWarning(
          paramsExt,
          personType,
        )
        if (retirementRemovalWarning.length > 0) {
          setRetirementRemovalWarning(retirementRemovalWarning)
          return params
        }
        if (clone.dialogPosition === 'future-savings') {
          const withdrawalStartPerson = !clone.people.withPartner
            ? 'person1'
            : clone.people.withdrawalStart
          if (personType === withdrawalStartPerson) {
            clone.dialogPosition = nextPlanSectionDialogPosition(
              'future-savings',
              1, // 0 throws the calculations off.
            )
          }
        }
        p.ages = {
          type: 'retired',
          currentMonth: p.ages.currentMonth,
          maxMonth: p.ages.maxMonth,
        }

        return clone
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
      <div
        className={`${className} `}
        style={style}
        ref={divRef}
        onClick={(e) => {
          if (e.target === divRef.current) setOpenSection(`none`)
        }}
      >
        <div
          className="flex justify-between"
          onClick={() => setOpenSection('none')}
        >
          <h2 className="font-bold text-lg">
            {personType === 'person1' ? 'You' : 'Your Partner'}
          </h2>
          {handleDeletePerson && (
            <button className="" onClick={handleDeletePerson}>
              <FontAwesomeIcon className="mr-2" icon={faTrash} />
            </button>
          )}
        </div>
        <div
          className="flex items-center col-span-2 gap-x-4 mt-4 mb-2"
          onClick={() => setOpenSection('none')}
        >
          <Switch.Group>
            <Switch.Label className="">
              {personType === 'person1'
                ? 'Are you retired?'
                : 'Is your partner retired?'}
            </Switch.Label>
            <CheckBox
              className=""
              enabled={ages.type === 'retired'}
              setEnabled={(retired) => {
                if (retired) {
                  handleRetired()
                  return
                }
                setParams((params) => {
                  const clone = _.cloneDeep(params)
                  const currPerson = getPersonInParams(clone)
                  const defaultPerson = getDefaultPlanParams().people.person1
                  assert(defaultPerson.ages.type === 'notRetired')
                  const retirementMonth =
                    defaultPerson.ages.retirementMonth <=
                    currPerson.ages.currentMonth
                      ? Math.floor(
                          (currPerson.ages.maxMonth +
                            currPerson.ages.currentMonth) /
                            2,
                        )
                      : defaultPerson.ages.retirementMonth
                  currPerson.ages = {
                    type: 'notRetired',
                    currentMonth: currPerson.ages.currentMonth,
                    maxMonth: currPerson.ages.maxMonth,
                    retirementMonth,
                  }
                  return clone
                })
              }}
            />
          </Switch.Group>
        </div>

        <div
          className="grid items-center "
          style={{ grid: 'auto / 145px 1fr' }}
        >
          <_AgeInput
            label="Current Age"
            getPersonInParams={getPersonInParams}
            numMonths={ages.currentMonth}
            setOnAges={(n, ages) => (ages.currentMonth = n)}
            range={{
              start: 0,
              end:
                ages.type === 'retired'
                  ? ages.maxMonth - 2
                  : ages.retirementMonth - 1,
            }}
            type={`${personType}-current`}
            openSection={openSection}
            setOpenSection={setOpenSection}
          />
          {ages.type === 'notRetired' && (
            <_AgeInput
              label="Retirement Age"
              getPersonInParams={getPersonInParams}
              numMonths={ages.retirementMonth}
              setOnAges={(n, ages) => {
                assert(ages.type === 'notRetired')
                ages.retirementMonth = n
              }}
              range={{
                start: ages.currentMonth + 1,
                end: ages.maxMonth - 1,
              }}
              type={`${personType}-retirement`}
              openSection={openSection}
              setOpenSection={setOpenSection}
            />
          )}
          <_AgeInput
            label="Max Age"
            getPersonInParams={getPersonInParams}
            numMonths={ages.maxMonth}
            setOnAges={(n, ages) => (ages.maxMonth = n)}
            range={{
              start:
                ages.type === 'retired'
                  ? ages.currentMonth + 2
                  : ages.retirementMonth + 1,
              end: MAX_AGE_IN_MONTHS,
            }}
            type={`${personType}-max`}
            openSection={openSection}
            setOpenSection={setOpenSection}
          />
        </div>
        <div
          className="grid gap-y-2 items-center gap-x-4"
          style={{ grid: 'auto / 145px 1fr' }}
        ></div>
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
                  analyzeMonthsInParams(
                    extendPlanParams(clone),
                    (month: Month): Month => {
                      if (
                        month.type === 'namedAge' &&
                        month.person === personType &&
                        (month.age === 'retirement' ||
                          month.age === 'lastWorkingMonth')
                      ) {
                        return { type: 'now' }
                      } else {
                        return month
                      }
                    },
                  )

                  const p = getPersonInParams(clone)
                  p.ages = {
                    type: 'retired',
                    currentMonth: p.ages.currentMonth,
                    maxMonth: p.ages.maxMonth,
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

export const _AgeInput = React.memo(
  ({
    label,
    getPersonInParams,
    numMonths,
    setOnAges,
    range,
    type,
    openSection,
    setOpenSection,
  }: {
    label: string
    getPersonInParams: (params: PlanParams) => Person
    numMonths: number
    setOnAges: (numMonths: number, ages: Person['ages']) => void
    range: SimpleRange
    type: Exclude<PlanInputAgeOpenableSection, 'none'>
    openSection: PlanInputAgeOpenableSection
    setOpenSection: (open: PlanInputAgeOpenableSection) => void
  }) => {
    const { setParams } = useSimulation()
    const buttonDivRef = useRef<HTMLDivElement>(null)

    return openSection !== type ? (
      <>
        <button
          className="text-start py-1.5 self-start whitespace-nowrap"
          onClick={() => setOpenSection(type)}
        >
          <FontAwesomeIcon className="text-xs mr-2" icon={faCaretRight} />
          {label}
        </button>
        <div
          className=""
          ref={buttonDivRef}
          onClick={(e) => {
            if (e.target === buttonDivRef.current) setOpenSection('none')
          }}
        >
          <button
            className="text-start pl-2 py-1.5"
            onClick={() => setOpenSection(type)}
          >
            {numMonthsStr(numMonths)}
          </button>
        </div>
      </>
    ) : (
      <div className={`col-span-2 bg-gray-100 rounded-xl p-2 my-2`}>
        <button
          className="py-1.5 text-start "
          onClick={() => setOpenSection('none')}
        >
          <FontAwesomeIcon className="text-xs mr-2" icon={faCaretDown} />
          {label}
        </button>
        <NumMonthsInput
          className="mt-2 mb-4 ml-4"
          modalLabel={label}
          value={numMonths}
          onChange={(numMonths) =>
            setParams((params) => {
              const clone = _.cloneDeep(params)
              const ages = getPersonInParams(clone).ages
              setOnAges(numMonths, ages)
              return clone
            })
          }
          range={range}
        />
      </div>
    )
  },
)

const _person2RequiredMessage = (params: PlanParamsExt) => {
  const analysis = analyzeMonthsInParams(params)
  const fromRanges = analysis.valueForMonthRange.filter((x) => x.usesPerson2)
  const fromGlidePath = analysis.glidePath.filter((x) => x.usesPerson2)

  if (fromRanges.length === 0 && fromGlidePath.length === 0) return []
  const result = [
    `The following are specified in terms of your partner's age:`,
  ] as ReactNode[]
  result.push(
    <ol className=" list-disc list-outside ml-10" key={_.uniqueId()}>
      {fromRanges.map((x, i) => (
        <li key={`range-${i}`} className="list-item">
          <ValueForMonthRangeDisplay entry={x.entry} range={null} skipLength/>
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
    <ol className=" list-decimal list-outside ml-10 " key={_.uniqueId()}>
      {futureSavings.map((x, i) => (
        <li key={i} className="list-item">
          <ValueForMonthRangeDisplay entry={x} range={null}  skipLength/>
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
  const analysis = analyzeMonthsInParams(paramsExt)
  const fromRanges = analysis.valueForMonthRange.filter((x) =>
    x.useRetirement(person),
  )
  const fromGlidePath = analysis.glidePath.filter((x) =>
    x.usesRetirement(person),
  )

  if (fromRanges.length === 0 && fromGlidePath.length === 0) return []

  const result = [
    `The following are specified in terms of ${yourOrYourPartners(
      person,
    )} retirement age:`,
  ] as ReactNode[]
  result.push(
    <ul className="list-disc list-outside ml-10" key={_.uniqueId()}>
      {fromRanges.map((x, i) => (
        <li key={`ranges-${i}`} className="list-item">
          <ValueForMonthRangeDisplay entry={x.entry} range={null}  skipLength/>
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
    `Would your like to convert these references to ${yourOrYourPartners(
      person,
    )} retirement to "now"?`,
  )
  return result
}
