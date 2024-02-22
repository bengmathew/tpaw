import { faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faExclamationCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clix from 'clsx'
import _ from 'lodash'
import React, { useRef, useState } from 'react'
import { PlanParamsExtended } from '../../../../../UseSimulator/ExtentPlanParams'
import { calendarMonthStr } from '../../../../../Utils/CalendarMonthStr'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { joinWithCommaAnd } from '../../../../../Utils/JoinWithAnd'
import { numMonthsStr } from '../../../../../Utils/NumMonthsStr'
import { yourOrYourPartners } from '../../../../../Utils/YourOrYourPartners'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { analyzeMonthsInParams } from '../Helpers/AnalyzeMonthsInParams'
import { planSectionLabel } from '../Helpers/PlanSectionLabel'
import {
    PlanInputBody,
    PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import { PlanInputAgePerson } from './PlanInputAgePerson'
import { PlanInputAgeWithdrawalStart } from './PlanInputAgeWithdrawalStart'

export type PlanInputAgeOpenableSection =
  | `${'person1' | 'person2'}-${'monthOfBirth' | 'retirementAge' | 'maxAge'}`
  | 'none'
export const PlanInputAge = React.memo((props: PlanInputBodyPassThruProps) => {
  const { planParams, updatePlanParams, planParamsExt } = useSimulation()
  const contentDivRef = useRef<HTMLDivElement>(null)
  const [openSection, setOpenSection] =
    useState<PlanInputAgeOpenableSection>('none')

  const monthAnalysis = analyzeMonthsInParams(planParamsExt, {
    type: 'asVisible',
  })
  const warnings = _.uniq(
    [
      ...monthAnalysis.valueForMonthRange.filter(
        (x) =>
          x.boundsCheck &&
          (x.boundsCheck.start !== 'ok' || x.boundsCheck.end !== 'ok'),
      ),
      ...monthAnalysis.glidePath.filter((x) =>
        x.analyzed.some((x) => x.issue !== 'none'),
      ),
    ]
      .map((x) => x.section)
      .map(planSectionLabel)
      .map((x) => `"${x}"`),
  )

  return (
    <PlanInputBody {...props} onBackgroundClick={() => setOpenSection('none')}>
      <div
        ref={contentDivRef}
        className=""
        onClick={(e) => {
          if (e.target === contentDivRef.current) setOpenSection('none')
        }}
      >
        <PlanInputAgePerson
          className="params-card"
          personType="person1"
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
          openSection={openSection}
          setOpenSection={setOpenSection}
        />
        {planParams.people.withPartner ? (
          <>
            <PlanInputAgePerson
              className="mt-10 params-card"
              personType="person2"
              style={{ padding: paddingCSS(props.sizing.cardPadding) }}
              openSection={openSection}
              setOpenSection={setOpenSection}
            />
            <PlanInputAgeWithdrawalStart
              className="mt-8 params-card"
              style={{ padding: paddingCSS(props.sizing.cardPadding) }}
            />
          </>
        ) : (
          <button
            className="params-card font-bold text-lg mt-10 flex items-center gap-x-2 w-full"
            style={{ padding: paddingCSS(props.sizing.cardPadding) }}
            onClick={() => updatePlanParams('addPartner', null)}
          >
            <h2 className="">Add a Partner</h2>
            <FontAwesomeIcon className=" " icon={faPlus} />
          </button>
        )}
      </div>
      {{
        error:
          warnings.length === 0 ? undefined : (
            <div className="p-base">
              <h2 className="text-errorFG ">
                <FontAwesomeIcon
                  className="mr-2 text-errorFG"
                  icon={faExclamationCircle}
                />{' '}
                {`Based on the ages set here, one or more months specified in the ${joinWithCommaAnd(
                  warnings,
                )} ${
                  warnings.length > 1 ? 'sections' : 'section'
                } needs to be updated.`}
              </h2>
            </div>
          ),
      }}
    </PlanInputBody>
  )
})

export const PlanInputAgeSummary = React.memo(
  ({ planParamsExt }: { planParamsExt: PlanParamsExtended }) => {
    const { planParams } = planParamsExt
    const {
      isAgesNotRetired,
      isPersonRetired,
      pickPerson,
      getCurrentAgeOfPerson,
      dialogPositionEffective,
    } = planParamsExt
    if (dialogPositionEffective === 'age') {
      return (
        <>
          <h2>Month of Birth: </h2>
          <h2>Retirement: </h2>
          <h2>Max: </h2>
        </>
      )
    }
    const forPerson = (person: 'person1' | 'person2', className = '') => {
      const { ages } = pickPerson(person)
      return isAgesNotRetired(ages) ? (
        <>
          <h2 className={`${className}`}>Month of Birth</h2>
          <h2>
            {calendarMonthStr(ages.monthOfBirth)} (Age:{' '}
            {numMonthsStr(getCurrentAgeOfPerson(person).inMonths)})
          </h2>
          <h2 className={`${className}`}>Retirement</h2>
          <h2> {numMonthsStr(ages.retirementAge.inMonths)}</h2>
          <h2 className={`${className}`}>Max</h2>
          <h2> {numMonthsStr(ages.maxAge.inMonths)}</h2>
        </>
      ) : (
        <>
          <h2 className={`${className} col-span-2`}>Retired</h2>
          <h2 className={`${className}`}>Month of Birth</h2>
          <h2>
            {' '}
            {calendarMonthStr(ages.monthOfBirth)} (Age:{' '}
            {numMonthsStr(getCurrentAgeOfPerson(person).inMonths)})
          </h2>
          <h2 className={`${className}`}>Max</h2>
          <h2> {numMonthsStr(ages.maxAge.inMonths)}</h2>
        </>
      )
    }
    if (planParams.people.withPartner) {
      const withdrawalPerson = pickPerson(planParams.people.withdrawalStart)
      return (
        <div
          className={clix('grid gap-x-3 gap-y-1')}
          style={{ grid: 'auto/auto 1fr' }}
        >
          <h2 className="font-medium col-span-2">You</h2>
          {forPerson('person1', 'ml-4')}
          <h2 className="mt-2 font-medium  col-span-2">Your Partner</h2>
          {forPerson('person2', 'ml-4')}
          {!(isPersonRetired('person1') && isPersonRetired('person2')) && (
            <h2 className="mt-2  col-span-2">
              Withdrawals start{' '}
              {isPersonRetired(withdrawalPerson)
                ? 'now.'
                : `at ${yourOrYourPartners(
                    planParams.people.withdrawalStart,
                  )} retirement.`}
            </h2>
          )}
        </div>
      )
    } else {
      return (
        <div
          className={clix('grid gap-x-3 gap-y-1')}
          style={{ grid: 'auto/auto 1fr' }}
        >
          {forPerson('person1')}
        </div>
      )
    }
  },
)
