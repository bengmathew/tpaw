import { faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faExclamationCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { CalendarMonthFns, fGet, noCase, yourOrYourPartners } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import React, { useRef, useState } from 'react'
import { PlanParamsNormalized } from '@tpaw/common'
import { PlanParamsHelperFns } from '@tpaw/common'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { InMonthsFns } from '@tpaw/common'
import { joinWithCommaAnd } from '../../../../../Utils/JoinWithAnd'
import { useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'
import { planSectionLabel } from '../Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import { PlanInputAgePerson } from './PlanInputAgePerson'
import { PlanInputAgeWithdrawalStart } from './PlanInputAgeWithdrawalStart'

export type PlanInputAgeOpenableSection =
  | `${'person1' | 'person2'}-${'currentAgeInfo' | 'retirementAge' | 'maxAge'}`
  | 'none'
export const PlanInputAge = React.memo((props: PlanInputBodyPassThruProps) => {
  const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
  const contentDivRef = useRef<HTMLDivElement>(null)
  const [openSection, setOpenSection] =
    useState<PlanInputAgeOpenableSection>('none')

  const planSectionsWithIssues = _.uniq(
    PlanParamsHelperFns.mapAllLabeledAmountTimedList(
      planParamsNormInstant,
      ({ amountAndTiming }, location) => {
        switch (amountAndTiming.type) {
          case 'inThePast':
            return { location, months: [] }
          case 'oneTime':
            return { location, months: [amountAndTiming.month] }
          case 'recurring':
            const { monthRange: range } = amountAndTiming
            switch (range.type) {
              case 'startAndEnd':
                return { location, months: [range.start, range.end] }
              case 'startAndDuration':
                return { location, months: [range.start, range.duration] }
              case 'endAndDuration':
                return { location, months: [range.end, range.duration] }
              default:
                noCase(range)
            }
          default:
            noCase(amountAndTiming)
        }
      },
    )
      .filter((x) => x.months.some((x) => x.errorMsg !== null))
      .map((x) => x.location)
      .map((x) => {
        switch (x) {
          case 'futureSavings':
            return 'future-savings'
          case 'incomeDuringRetirement':
            return 'income-during-retirement'
          case 'extraSpendingEssential':
            return 'extra-spending'
          case 'extraSpendingDiscretionary':
            return 'extra-spending'
          default:
            return noCase(x)
        }
      }),
  ).map(planSectionLabel)

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
          personId="person1"
          style={{ padding: paddingCSS(props.sizing.cardPadding) }}
          openSection={openSection}
          setOpenSection={setOpenSection}
        />
        {planParamsNormInstant.ages.person2 ? (
          <>
            <PlanInputAgePerson
              className="mt-10 params-card"
              personId="person2"
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
          planSectionsWithIssues.length === 0 ? undefined : (
            <div className="p-base">
              <h2 className="text-errorFG ">
                <FontAwesomeIcon
                  className="mr-2 text-errorFG"
                  icon={faExclamationCircle}
                />{' '}
                {`Based on the ages set here, one or more months specified in the ${joinWithCommaAnd(
                  planSectionsWithIssues,
                )} ${
                  planSectionsWithIssues.length > 1 ? 'sections' : 'section'
                } needs to be updated.`}
              </h2>
            </div>
          ),
      }}
    </PlanInputBody>
  )
})

export const PlanInputAgeSummary = React.memo(
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    if (planParamsNorm.dialogPosition.effective === 'age') {
      return (
        <>
          <h2>{planParamsNorm.datingInfo.isDated? 'Month of Birth': 'Current'}: </h2>
          <h2>Retirement: </h2>
          <h2>Max: </h2>
        </>
      )
    }
    const { ages } = planParamsNorm
    const forPerson = (personType: 'person1' | 'person2', className = '') => {
      const person = fGet(ages[personType])
      return (
        <>
          {person.retirement.isRetired && (
            <h2 className={`${className} col-span-2`}>Retired</h2>
          )}
          {person.currentAgeInfo.isDatedPlan ? (
            <>
              <h2 className={`${className}`}>Month of Birth</h2>
              <h2>
                {CalendarMonthFns.toStr(person.currentAgeInfo.baseValue)} (Age:{' '}
                {InMonthsFns.toStr(person.currentAgeInfo)})
              </h2>
            </>
          ) : (
            <>
              <h2 className={`${className}`}>Current</h2>
              <h2>{InMonthsFns.toStr(person.currentAgeInfo)}</h2>
            </>
          )}

          {person.retirement.ageIfInFuture && (
            <>
              <h2 className={`${className}`}>Retirement</h2>
              <h2>
                {InMonthsFns.toStr(person.retirement.ageIfInFuture.baseValue)}
              </h2>
            </>
          )}
          <h2 className={`${className}`}>Max</h2>
          <h2> {InMonthsFns.toStr(person.maxAge.baseValue)}</h2>
        </>
      )
    }
    if (ages.person2) {
      return (
        <div
          className={clix('grid gap-x-3 gap-y-1')}
          style={{ grid: 'auto/auto 1fr' }}
        >
          <h2 className="font-medium col-span-2">You</h2>
          {forPerson('person1', 'ml-4')}
          <h2 className="mt-2 font-medium  col-span-2">Your Partner</h2>
          {forPerson('person2', 'ml-4')}
          {!(
            ages.person1.retirement.isRetired &&
            ages.person2.retirement.isRetired
          ) && (
            <h2 className="mt-2  col-span-2">
              Withdrawals starts{' '}
              {fGet(
                ages[ages.simulationMonths.withdrawalStartMonth.atRetirementOf],
              ).retirement.isRetired
                ? 'now'
                : `at ${yourOrYourPartners(
                    ages.simulationMonths.withdrawalStartMonth.atRetirementOf,
                  )} retirement`}
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
