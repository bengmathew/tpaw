import {
  faCaretDown,
  faCaretRight,
  faTrash,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import {
  MAX_AGE_IN_MONTHS,
  PlanParams,
  block,
  noCase,
  planParamsFns,
} from '@tpaw/common'
import _ from 'lodash'
import React, { ReactNode, useMemo, useRef, useState } from 'react'
import { calendarMonthStr } from '../../../../../Utils/CalendarMonthStr'
import { numMonthsStr } from '../../../../../Utils/NumMonthsStr'
import { pluralize } from '../../../../../Utils/Pluralize'
import { SimpleRange } from '../../../../../Utils/SimpleRange'
import { assert } from '../../../../../Utils/Utils'
import { yourOrYourPartners } from '../../../../../Utils/YourOrYourPartners'
import { CalendarMonthInput } from '../../../../Common/Inputs/CalendarMonthInput'
import { CheckBox } from '../../../../Common/Inputs/CheckBox'
import { NumMonthsInput } from '../../../../Common/Inputs/NumMonthsInput'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import { ValueForMonthRangeDisplay } from '../../../../Common/ValueForMonthRangeDisplay'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { analyzeMonthsInParams } from '../Helpers/AnalyzeMonthsInParams'
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
    const { planParams, planParamsExt, updatePlanParams } = useSimulation()
    const divRef = useRef<HTMLDivElement>(null)
    const { isPersonRetired, getCurrentAgeOfPerson, isAgesNotRetired } =
      planParamsExt

    const { ages } = _getPersonInPlan(planParams, personType)

    const retireWarnings = useRetireWarnings(personType)
    const [showRetireWarnings, setShowRetireWarnings] = useState(false)
    const person2DeleteWarnings = usePerson2DeleteWarnings()
    const [showPerson2DeleteWarnings, setShowPerson2DeleteWarnings] =
      useState(false)

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
          {personType === 'person2' && (
            <button
              className=""
              onClick={() => {
                if (person2DeleteWarnings) {
                  setShowPerson2DeleteWarnings(true)
                } else {
                  updatePlanParams('deletePartner', null)
                }
              }}
            >
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
              setEnabled={(retired) => {
                if (retired) {
                  if (retireWarnings) {
                    setShowRetireWarnings(true)
                  } else {
                    updatePlanParams('setPersonRetired', personType)
                  }
                } else {
                  updatePlanParams('setPersonNotRetired', personType)
                }
              }}
            />
          </Switch.Group>
        </div>

        <div
          className="grid items-center "
          style={{ grid: 'auto / 145px 1fr' }}
        >
          <_MonthOfBirthInput
            sectionName="Month of Birth"
            personType={personType}
            sectionType={`${personType}-monthOfBirth`}
            currSection={openSection}
            setCurrSection={setOpenSection}
          />
          {isAgesNotRetired(ages) && (
            <_AgeInput
              sectionName="Retirement Age"
              personType={personType}
              type="retirementAge"
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
            personType={personType}
            type="maxAge"
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
        <CenteredModal
          className=" dialog-outer-div"
          show={showPerson2DeleteWarnings}
          onOutsideClickOrEscape={null}
        >
          <h2 className=" dialog-heading">Error Removing Partner</h2>
          <div className=" dialog-content-div">{person2DeleteWarnings}</div>
          <div className=" dialog-button-div">
            <button
              className=" dialog-button-dark"
              onClick={() => {
                setShowPerson2DeleteWarnings(false)
              }}
            >
              Close
            </button>
          </div>
        </CenteredModal>

        <CenteredModal
          className=" dialog-outer-div"
          show={showRetireWarnings}
          onOutsideClickOrEscape={null}
        >
          <h2 className=" dialog-heading text-errorFG">Warning</h2>
          <div className=" dialog-content-div">{retireWarnings}</div>
          <div className=" dialog-button-div">
            <button
              className=" dialog-button-cancel"
              onClick={() => {
                setShowRetireWarnings(false)
              }}
            >
              Cancel
            </button>
            <button
              className=" dialog-button-warning"
              onClick={() => {
                setShowRetireWarnings(false)
                updatePlanParams('setPersonRetired', personType)
              }}
            >
              Continue
            </button>
          </div>
        </CenteredModal>
      </div>
    )
  },
)

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
    personType,
    sectionType,
    currSection,
    setCurrSection,
  }: {
    sectionName: string
    personType: 'person1' | 'person2'
    sectionType: Exclude<PlanInputAgeOpenableSection, 'none'>
    currSection: PlanInputAgeOpenableSection
    setCurrSection: (open: PlanInputAgeOpenableSection) => void
  }) => {
    const { planParams, updatePlanParams, planParamsExt } = useSimulation()
    const value = _getPersonInPlan(planParams, personType).ages.monthOfBirth
    const { asMFN } = planParamsExt
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
            onChange={(monthOfBirth) =>
              updatePlanParams('setPersonMonthOfBirth', {
                person: personType,
                monthOfBirth,
              })
            }
            rangeAsMFN={{ start: -MAX_AGE_IN_MONTHS + 3, end: 0 }}
          />
          <h2 className="mb-2 mt-3 ml-4">Age: {numMonthsStr(-valueAsMFN)}</h2>
        </div>
      </_Section>
    )
  },
)
export const _AgeInput = React.memo(
  ({
    sectionName,
    personType,
    type,
    range,
    sectionType,
    currSection,
    setCurrSection,
  }: {
    sectionName: string
    personType: 'person1' | 'person2'
    type: 'retirementAge' | 'maxAge'
    range: SimpleRange
    sectionType: Exclude<PlanInputAgeOpenableSection, 'none'>
    currSection: PlanInputAgeOpenableSection
    setCurrSection: (open: PlanInputAgeOpenableSection) => void
  }) => {
    const { planParams, updatePlanParams } = useSimulation()
    const age = (() => {
      const person = _getPersonInPlan(planParams, personType)
      if (type === 'retirementAge') {
        assert(person.ages.type === 'retirementDateSpecified')
        return person.ages.retirementAge
      }
      if (type === 'maxAge') return person.ages.maxAge
      noCase(type)
    })()

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
            type === 'retirementAge'
              ? updatePlanParams('setPersonRetirementAge', {
                  person: personType,
                  retirementAge: { inMonths },
                })
              : type === 'maxAge'
              ? updatePlanParams('setPersonMaxAge', {
                  person: personType,
                  maxAge: { inMonths },
                })
              : noCase(type)
          }
          rangeAsMFN={range}
        />
      </_Section>
    )
  },
)

const usePerson2DeleteWarnings = () => {
  const { planParamsExt } = useSimulation()
  const { fromRanges, fromGlidePath } = useMemo(() => {
    const analysis = analyzeMonthsInParams(planParamsExt, {
      type: 'asVisible',
    })
    const fromRanges = analysis.valueForMonthRange.filter((x) => x.usesPerson2)
    const fromGlidePath = analysis.glidePath.filter((x) => x.usesPerson2)
    return { fromRanges, fromGlidePath }
  }, [planParamsExt])

  if (fromRanges.length === 0 && fromGlidePath.length === 0) return null
  return (
    <div className="">
      <h2 className="">{`The following are specified in terms of your partner's age:`}</h2>
      <ol className=" list-disc list-outside ml-10 mt-4" key={_.uniqueId()}>
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
      </ol>
      <h2 className="mt-4">{`Remove these references to your partner's age before removing your partner.`}</h2>
    </div>
  )
}

const useRetireWarnings = (personType: 'person1' | 'person2') => {
  const { getIsFutureSavingsAllowed } = planParamsFns
  const { planParamsExt } = useSimulation()
  const { isPersonRetired, planParams } = planParamsExt
  const futureSavingsWarnings = block(() => {
    const { futureSavings } = planParams.wealth
    const count = _.values(futureSavings).length
    if (count === 0) return null
    const isFutureSavingsGoingToBeAllowed = getIsFutureSavingsAllowed(
      personType === 'person1' ? true : isPersonRetired('person1'),
      personType === 'person2'
        ? true
        : planParams.people.withPartner
        ? isPersonRetired('person2')
        : undefined,
    )
    if (isFutureSavingsGoingToBeAllowed) return null
    return (
      <div className="">
        <h2 key={1} className="">
          {`The Future Savings section is not applicable once ${
            planParams.people.withPartner ? 'both you and your partner' : 'you'
          } are retired but has the following ${pluralize(
            count,
            'entry',
            'entries',
          )}:`}
        </h2>
        <ul className=" list-disc list-outside ml-10 mt-4">
          {_.values(futureSavings)
            .sort((a, b) => a.sortIndex - b.sortIndex)
            .map((x, i) => (
              <li key={i} className="list-item">
                <ValueForMonthRangeDisplay
                  entry={x}
                  rangeAsMFN={null}
                  skipLength
                />
              </li>
            ))}
        </ul>
        <h2 className="mt-4  font-bold">
          {'These entries will be deleted if you continue!'}
        </h2>
      </div>
    )
  })
  const hasFutureSavingsWarnings = !!futureSavingsWarnings
  const retirementRefAnalysis = useMemo(() => {
    const analysis = analyzeMonthsInParams(planParamsExt, {
      type: 'asVisible',
    })
    const fromRanges = analysis.valueForMonthRange
      .filter((x) => x.useRetirement(personType))
      .filter((x) =>
        hasFutureSavingsWarnings ? x.section !== 'future-savings' : true,
      )
    const fromGlidePath = analysis.glidePath.filter((x) =>
      x.usesRetirement(personType),
    )
    return { fromRanges, fromGlidePath }
  }, [hasFutureSavingsWarnings, personType, planParamsExt])

  const retirementReferencesWarnings = block(() => {
    const { fromRanges, fromGlidePath } = retirementRefAnalysis
    if (fromRanges.length === 0 && fromGlidePath.length === 0) return null
    return (
      <div className="">
        <h2 className="">
          {`The following are specified in terms of ${yourOrYourPartners(
            personType,
          )} retirement age:`}
        </h2>
        <ul className="list-disc list-outside ml-10 mt-4" key={_.uniqueId()}>
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
        </ul>
        <h2 className="mt-4 font-bold">
          {`These references to ${yourOrYourPartners(
            personType,
          )} retirement will be converted to "now" if you continue`}
        </h2>
      </div>
    )
  })
  if (!futureSavingsWarnings && !retirementReferencesWarnings) return null
  return (
    <div className="flex flex-col gap-y-6">
      {futureSavingsWarnings}
      {retirementReferencesWarnings}
    </div>
  )
}

const _getPersonInPlan = (
  plan: PlanParams,
  personType: 'person1' | 'person2',
) => {
  if (personType === 'person1') return plan.people.person1
  assert(plan.people.withPartner)
  return plan.people.person2
}
