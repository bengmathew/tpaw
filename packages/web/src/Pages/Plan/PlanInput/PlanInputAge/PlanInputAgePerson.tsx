import {
  faCaretDown,
  faCaretRight,
  faTrash,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import {
  InMonths,
  MAX_AGE_IN_MONTHS,
  Month,
  PlanParams,
  Params,
  Person,
} from '@tpaw/common'
import _ from 'lodash'
import React, { ReactNode, useMemo, useRef, useState } from 'react'
import {
  ParamsExtended,
  extendParams,
} from '../../../../TPAWSimulator/ExtentParams'
import { normalizeGlidePath } from '../../../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessRisk'
import { calendarMonthStr } from '../../../../Utils/CalendarMonthStr'
import { numMonthsStr } from '../../../../Utils/NumMonthsStr'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { assert } from '../../../../Utils/Utils'
import { yourOrYourPartners } from '../../../../Utils/YourOrYourPartners'
import { useSimulation } from '../../../App/WithSimulation'
import { CalendarMonthInput } from '../../../Common/Inputs/CalendarMonthInput'
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
    const { params, paramsExt, setPlanParams, currentTime, defaultParams } =
      useSimulation()
    const divRef = useRef<HTMLDivElement>(null)
    const { isPersonRetired, getCurrentAgeOfPerson, isAgesNotRetired } =
      paramsExt

    const getPersonInPlan = (plan: PlanParams) => {
      if (personType === 'person1') return plan.people.person1
      assert(plan.people.withPartner)
      return plan.people.person2
    }
    const { ages } = getPersonInPlan(params.plan)
    const handleDeletePerson =
      personType === 'person1'
        ? null
        : () => {
            const errMessage = _person2RequiredMessage(params.plan, paramsExt)
            if (errMessage.length > 0) {
              setPerson2DeleteError(errMessage)
              return
            }
            setPlanParams((plan, params) => {
              const clone = _.cloneDeep(plan)
              // There might be references in the past, that can be resolved to
              // "now", and references in glide paths that are not currently
              // active for the strategy that won't show up in the validation
              // errors.
              _handlePerson2ReferencesThatDontTriggerError(
                clone,
                extendParams(params, currentTime),
              )
              const { person1 } = clone.people
              clone.people = { withPartner: false, person1 }

              return clone
            })
          }

    const setRetiredOnPlan = (plan: PlanParams, params: Params) => {
      // Even if there are no warnings, there might still be hidden
      // retirement references. Move those to "now".
      _convertRetirementReferencesToNow(
        plan,
        extendParams(params, currentTime),
        personType,
      )
      const p = getPersonInPlan(plan)
      if (plan.dialogPosition === 'future-savings') {
        const withdrawalStartPerson = !plan.people.withPartner
          ? 'person1'
          : plan.people.withdrawalStart
        if (personType === withdrawalStartPerson) {
          plan.dialogPosition = nextPlanSectionDialogPosition(
            'future-savings',
            1, // 0 throws the calculations off.
          )
        }
      }
      p.ages = {
        type: 'retiredWithNoRetirementDateSpecified',
        monthOfBirth: p.ages.monthOfBirth,
        maxAge: p.ages.maxAge,
      }
      return plan
    }

    const handleRetired = () => {
      const futureSavingsWarning = _futureSavingsWarning(paramsExt)
      const retirementRemovalWarning = _retirementReferenceWarning(
        params.plan,
        paramsExt,
        personType,
      )
      if (futureSavingsWarning.length > 0) {
        setFutureSavingsWarning(futureSavingsWarning)
      } else if (retirementRemovalWarning.length > 0) {
        setRetirementReferenceWarning(retirementRemovalWarning)
      } else {
        setPlanParams((plan, params) => {
          const clone = _.cloneDeep(plan)
          setRetiredOnPlan(clone, params)
          return clone
        })
      }
    }

    const handleNotRetired = () => {
      setPlanParams((plan) => {
        const clone = _.cloneDeep(plan)
        const currPerson = getPersonInPlan(clone)
        const defaultPerson = defaultParams.plan.people.person1
        assert(defaultPerson.ages.type === 'retirementDateSpecified')
        const retirementAge =
          defaultPerson.ages.retirementAge.inMonths <=
          getCurrentAgeOfPerson(personType).inMonths
            ? {
                inMonths: Math.floor(
                  (currPerson.ages.maxAge.inMonths +
                    getCurrentAgeOfPerson(personType).inMonths) /
                    2,
                ),
              }
            : defaultPerson.ages.retirementAge
        currPerson.ages = {
          type: 'retirementDateSpecified',
          monthOfBirth: currPerson.ages.monthOfBirth,
          maxAge: currPerson.ages.maxAge,
          retirementAge,
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
    const [retirementReferenceWarning, setRetirementReferenceWarning] =
      useState<ReactNode[]>([])
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
              enabled={isPersonRetired({ ages })}
              setEnabled={(retired) =>
                retired ? handleRetired() : handleNotRetired()
              }
            />
          </Switch.Group>
        </div>

        <div
          className="grid items-center "
          style={{ grid: 'auto / 145px 1fr' }}
        >
          <_MonthOfBirthInput
            sectionName="Month of Birth"
            getPersonInParams={getPersonInPlan}
            sectionType={`${personType}-monthOfBirth`}
            currSection={openSection}
            setCurrSection={setOpenSection}
          />
          {isAgesNotRetired(ages) && (
            <_AgeInput
              sectionName="Retirement Age"
              getPersonInParams={getPersonInPlan}
              age={ages.retirementAge}
              setOnAges={(age, ages) => {
                assert(ages.type === 'retirementDateSpecified')
                ages.retirementAge = age
              }}
              range={{
                start: getCurrentAgeOfPerson(personType).inMonths + 1,
                end: ages.maxAge.inMonths - 1,
              }}
              sectionType={`${personType}-retirementAge`}
              currSection={openSection}
              setCurrSection={setOpenSection}
            />
          )}
          <_AgeInput
            sectionName="Max Age"
            getPersonInParams={getPersonInPlan}
            age={ages.maxAge}
            setOnAges={(n, ages) => (ages.maxAge = n)}
            range={{
              start: isAgesNotRetired(ages)
                ? ages.retirementAge.inMonths + 1
                : getCurrentAgeOfPerson(personType).inMonths + 2,
              end: MAX_AGE_IN_MONTHS,
            }}
            sectionType={`${personType}-maxAge`}
            currSection={openSection}
            setCurrSection={setOpenSection}
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
                setPlanParams((plan, params) => {
                  const clone = _.cloneDeep(plan)
                  clone.wealth.futureSavings = []
                  setRetiredOnPlan(clone, params)
                  return clone
                })
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
                setRetirementReferenceWarning([])
                setPlanParams((plan, params) => {
                  const clone = _.cloneDeep(plan)
                  _convertRetirementReferencesToNow(
                    clone,
                    extendParams(params, currentTime),
                    personType,
                  )
                  setRetiredOnPlan(clone, params)
                  return clone
                })
              },
              isWarning: true,
            }}
            onCancel={() => setRetirementReferenceWarning([])}
          >
            {retirementReferenceWarning}
          </ConfirmAlert>
        )}
      </div>
    )
  },
)

const _convertRetirementReferencesToNow = (
  plan: PlanParams,
  paramsExt: ParamsExtended,
  personType: 'person1' | 'person2',
) => {
  const { months } = paramsExt
  const referencesRetirement = (month: Month) =>
    month.type === 'namedAge' &&
    month.person === personType &&
    (month.age === 'retirement' || month.age === 'lastWorkingMonth')
  analyzeMonthsInParams(plan, paramsExt, {
    type: 'raw',
    monthInRangeUpdater: (month: Month): Month => {
      return referencesRetirement(month) ? months.now : month
    },
    glidePathUpdater: (x) => {
      const clone = _.cloneDeep(x)
      clone.intermediate = clone.intermediate.map((x) => ({
        month: referencesRetirement(x.month) ? months.now : x.month,
        stocks: x.stocks,
      }))
      return clone
    },
  })
}
const _handlePerson2ReferencesThatDontTriggerError = (
  plan: PlanParams,
  paramsExt: ParamsExtended,
) => {
  const { months, asMFN, currentMonth } = paramsExt
  const usesPerson2 = (x: Month) =>
    x.type !== 'calendarMonth' &&
    x.type !== 'calendarMonthAsNow' &&
    x.person === 'person2'
  analyzeMonthsInParams(plan, paramsExt, {
    type: 'raw',
    // Error won't trigger if month is in the past and references person2.
    monthInRangeUpdater: (month: Month): Month => {
      if (usesPerson2(month)) {
        assert(asMFN(month) < 0)
        return months.now
      } else {
        return month
      }
    },
    // Error won't trigger if strategy is not SPAW or SWR and the
    // glide path actually has a ref to person2 or if ref to person2 is in the past.
    glidePathUpdater: (x) => {
      const clone = _.cloneDeep(x)
      if (clone.intermediate.some((x) => usesPerson2(x.month))) {
        const startAllocation = normalizeGlidePath(clone, paramsExt)[0]
        clone.intermediate = clone.intermediate.filter(
          (x) => !usesPerson2(x.month),
        )
        // Preserve the current stock allocation if it changed due to filtering.
        if (normalizeGlidePath(clone, paramsExt)[0] !== startAllocation) {
          clone.start.month = currentMonth
          clone.start.stocks = startAllocation
        }
      }
      return clone
    },
    // glidePathFilter: (month) => {
    //   if (usesPerson2(month)) {
    //     assert(asMFN(month) < 0)
    //     return false
    //   } else {
    //     return true
    //   }
    // },
  })
}

export const _Section = React.memo(
  ({
    sectionName,
    sectionType,
    currSection,
    setCurrSection,
    children: [summaryChild, editChild],
  }: {
    sectionName: string
    sectionType: Exclude<PlanInputAgeOpenableSection, 'none'>
    currSection: PlanInputAgeOpenableSection
    setCurrSection: (open: PlanInputAgeOpenableSection) => void
    children: [React.ReactNode, React.ReactNode]
  }) => {
    const buttonDivRef = useRef<HTMLDivElement>(null)

    return currSection !== sectionType ? (
      <>
        <button
          className="text-start py-1.5 self-start whitespace-nowrap"
          onClick={() => setCurrSection(sectionType)}
        >
          <FontAwesomeIcon className="text-xs mr-2" icon={faCaretRight} />
          {sectionName}
        </button>
        <div
          className=""
          ref={buttonDivRef}
          onClick={(e) => {
            if (e.target === buttonDivRef.current) setCurrSection('none')
          }}
        >
          <button
            className="text-start pl-2 py-1.5"
            onClick={() => setCurrSection(sectionType)}
          >
            {summaryChild}
          </button>
        </div>
      </>
    ) : (
      <div className={`col-span-2 bg-gray-100 rounded-xl p-2 my-2`}>
        <button
          className="py-1.5 text-start "
          onClick={() => setCurrSection('none')}
        >
          <FontAwesomeIcon className="text-xs mr-2" icon={faCaretDown} />
          {sectionName}
        </button>
        {editChild}
      </div>
    )
  },
)

export const _MonthOfBirthInput = React.memo(
  ({
    sectionName,
    getPersonInParams,
    sectionType,
    currSection,
    setCurrSection,
  }: {
    sectionName: string
    getPersonInParams: (params: PlanParams) => Person
    sectionType: Exclude<PlanInputAgeOpenableSection, 'none'>
    currSection: PlanInputAgeOpenableSection
    setCurrSection: (open: PlanInputAgeOpenableSection) => void
  }) => {
    const { params, setPlanParams, paramsExt } = useSimulation()
    const value = getPersonInParams(params.plan).ages.monthOfBirth
    const { asMFN } = paramsExt
    const valueAsMFN = useMemo(() => asMFN(value), [asMFN, value])

    return (
      <_Section
        sectionName={sectionName}
        sectionType={sectionType}
        currSection={currSection}
        setCurrSection={setCurrSection}
      >
        {calendarMonthStr(value)}
        <div className="">
          <CalendarMonthInput
            className="mt-2 ml-4"
            value={value}
            onChange={(value) =>
              setPlanParams((plan) => {
                const clone = _.cloneDeep(plan)
                const ages = getPersonInParams(clone).ages
                ages.monthOfBirth = value
                return clone
              })
            }
            rangeAsMFN={{ start: -MAX_AGE_IN_MONTHS + 3, end: 0 }}
          />
          <h2 className="mb-2 mt-3 ml-4">
            Age: {numMonthsStr(-valueAsMFN)}
          </h2>
        </div>
      </_Section>
    )
  },
)
export const _AgeInput = React.memo(
  ({
    sectionName,
    getPersonInParams,
    age,
    setOnAges,
    range,
    sectionType,
    currSection,
    setCurrSection,
  }: {
    sectionName: string
    getPersonInParams: (params: PlanParams) => Person
    age: InMonths
    setOnAges: (age: InMonths, ages: Person['ages']) => void
    range: SimpleRange
    sectionType: Exclude<PlanInputAgeOpenableSection, 'none'>
    currSection: PlanInputAgeOpenableSection
    setCurrSection: (open: PlanInputAgeOpenableSection) => void
  }) => {
    const { setPlanParams } = useSimulation()

    return (
      <_Section
        sectionName={sectionName}
        sectionType={sectionType}
        currSection={currSection}
        setCurrSection={setCurrSection}
      >
        {numMonthsStr(age.inMonths)}
        <NumMonthsInput
          className="mt-2 mb-4 ml-4"
          modalLabel={sectionName}
          value={age.inMonths}
          onChange={(inMonths) =>
            setPlanParams((plan) => {
              const clone = _.cloneDeep(plan)
              const ages = getPersonInParams(clone).ages
              setOnAges({ inMonths }, ages)
              return clone
            })
          }
          rangeAsMFN={range}
        />
      </_Section>
    )
  },
)

const _person2RequiredMessage = (
  plan: PlanParams,
  paramsExtended: ParamsExtended,
) => {
  const analysis = analyzeMonthsInParams(plan, paramsExtended, {
    type: 'asVisible',
  })
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
          <ValueForMonthRangeDisplay
            entry={x.entry}
            rangeAsMFN={null}
            skipLength
          />
        </li>
      ))}
      {fromGlidePath.map((x, i) => (
        <li key={`glidePath-${i}`} className="list-item">
          Stock allocation in the {`"${x.sectionLabel}"`} section.
        </li>
      ))}
    </ol>,
    `Remove these references to your partner's age before removing your partner.`,
  )
  return result
}

const _futureSavingsWarning = (paramsExt: ParamsExtended) => {
  const { allowFutureSavingsEntries } = paramsExt
  const { people, wealth } = paramsExt.params.plan
  const { futureSavings } = wealth
  if (allowFutureSavingsEntries || futureSavings.length === 0) return []

  const result = [] as ReactNode[]
  result.push(
    `The Future Savings section is not applicable once ${
      people.withPartner ? 'both you and your partner' : 'you'
    } are retired but has the following ${
      futureSavings.length === 1 ? 'entry' : `${futureSavings.length} entries`
    }:`,
    <ol className=" list-decimal list-outside ml-10 " key={_.uniqueId()}>
      {futureSavings.map((x, i) => (
        <li key={i} className="list-item">
          <ValueForMonthRangeDisplay entry={x} rangeAsMFN={null} skipLength />
        </li>
      ))}
    </ol>,
    'Would you like to delete these entries?',
  )
  return result
}

const _retirementReferenceWarning = (
  plan: PlanParams,
  paramsExt: ParamsExtended,
  person: 'person1' | 'person2',
) => {
  const analysis = analyzeMonthsInParams(plan, paramsExt, { type: 'asVisible' })
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
          <ValueForMonthRangeDisplay
            entry={x.entry}
            rangeAsMFN={null}
            skipLength
          />
        </li>
      ))}
      {fromGlidePath.map((x, i) => (
        <li key={`glidePath-${i}`} className="list-item">
          Stock allocation in the {`"${x.sectionLabel}"`} section.
        </li>
      ))}
    </ul>,
    `Would your like to convert these references to ${yourOrYourPartners(
      person,
    )} retirement to "now"?`,
  )
  return result
}
