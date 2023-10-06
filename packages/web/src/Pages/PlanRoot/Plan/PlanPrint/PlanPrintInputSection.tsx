import { assert } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { ReactNode } from 'react'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { useGetPlanInputVisibility } from '../PlanInput/Helpers/UseGetPlanInputVisibility'
import { PlanInputAgeSummary } from '../PlanInput/PlanInputAge/PlanInputAge'
import { PlanInputCurrentPortfolioBalanceSummary } from '../PlanInput/PlanInputCurrentPortfolioBalance'
import { PlanInputExpectedReturnsAndVolatilitySummary } from '../PlanInput/PlanInputExpectedReturnsAndVolatility'
import { PlanInputExtraSpendingSummary } from '../PlanInput/PlanInputExtraSpending'
import { PlanInputFutureSavingsSummary } from '../PlanInput/PlanInputFutureSavings'
import { PlanInputIncomeDuringRetirementSummary } from '../PlanInput/PlanInputIncomeDuringRetirement'
import { PlanInputInflationSummary } from '../PlanInput/PlanInputInflation'
import { PlanInputLegacySummary } from '../PlanInput/PlanInputLegacy'
import { PlanInputRiskSummary } from '../PlanInput/PlanInputRisk/PlanInputRisk'
import { PlanInputSimulationSummary } from '../PlanInput/PlanInputSimulation'
import { PlanInputSpendingCeilingAndFloorSummary } from '../PlanInput/PlanInputSpendingCeilingAndFloor'
import { PlanInputStrategySummary } from '../PlanInput/PlanInputStrategy'
import { PlanPrintSection } from './PlanPrintSection'

// Note: The show/hide should mirror PlanSummary. "None" displays should mirror
// PlanSummaryButtons's empty attribute from PlanSummary.
export const PlanPrintInputSection = React.memo(() => {
  const { planParams, defaultPlanParams } = useSimulation()

  const getVisibility = useGetPlanInputVisibility()

  // TODO: show advanced input modified info.

  const _assert = (condition: boolean) => {
    assert(condition)
    return true
  }
  return (
    <>
      <PlanPrintSection className="flex items-center justify-center">
        <h1 className="font-bold text-4xl text-center ">Your Inputs</h1>
      </PlanPrintSection>
      <PlanPrintSection
        className="grid gap-y-12 gap-x-5"
        style={{ grid: 'auto / auto 1fr' }}
      >
        <_SectionHeading className="">Age</_SectionHeading>
        {_assert(getVisibility('age').visible) && (
          <div className="mt-1.5">
            <PlanInputAgeSummary />
          </div>
        )}
        <_SectionHeading className="">Wealth</_SectionHeading>
        <div className="mt-1.5">
          {_assert(getVisibility('current-portfolio-balance').visible) && (
            <div className=" break-inside-avoid-page">
              <_SubSectionHeading className="" isFirst>
                Current Portfolio Balance
              </_SubSectionHeading>
              <div className="mt-2">
                <PlanInputCurrentPortfolioBalanceSummary />
              </div>
            </div>
          )}
          {getVisibility('future-savings') && (
            <div className=" ">
              <_SubSectionHeading className="">
                Future Savings
              </_SubSectionHeading>
              <div className="mt-2">
                {_.values(planParams.wealth.futureSavings).length > 0 ? (
                  <PlanInputFutureSavingsSummary />
                ) : (
                  <_None />
                )}
              </div>
            </div>
          )}
          {_assert(getVisibility('income-during-retirement').visible) && (
            <div className=" ">
              <_SubSectionHeading className="">
                Income During Retirement
              </_SubSectionHeading>
              <div className="mt-2">
                {_.values(planParams.wealth.incomeDuringRetirement).length >
                0 ? (
                  <PlanInputIncomeDuringRetirementSummary />
                ) : (
                  <_None />
                )}
              </div>
            </div>
          )}
        </div>

        <_SectionHeading className="">
          Adjustments <br /> to <br /> Spending
        </_SectionHeading>

        <div className="mt-1.5">
          <_SubSectionHeading className="" isFirst>
            Extra Spending
          </_SubSectionHeading>
          <PlanInputExtraSpendingSummary forPrint />

          {getVisibility('legacy').visible && (
            <>
              <_SubSectionHeading className="">Legacy</_SubSectionHeading>
              <div className="mt-2">
                {_.values(
                  planParams.adjustmentsToSpending.tpawAndSPAW.legacy.external,
                ).length === 0 &&
                planParams.adjustmentsToSpending.tpawAndSPAW.legacy.total ===
                  0 ? (
                  <_None />
                ) : (
                  <PlanInputLegacySummary />
                )}
              </div>
            </>
          )}

          {getVisibility('spending-ceiling-and-floor').visible && (
            <>
              <_SubSectionHeading className="">
                Spending Ceiling And Floor
              </_SubSectionHeading>
              <div className="mt-2">
                {planParams.adjustmentsToSpending.tpawAndSPAW
                  .monthlySpendingCeiling === null &&
                planParams.adjustmentsToSpending.tpawAndSPAW
                  .monthlySpendingFloor === null ? (
                  <_None />
                ) : (
                  <PlanInputSpendingCeilingAndFloorSummary />
                )}
              </div>
            </>
          )}
        </div>
        {_assert(getVisibility('risk').visible) && (
          <>
            <_SectionHeading className="">Risk</_SectionHeading>
            <div className=" break-inside-avoid-page mt-2">
              <div className="">
                <PlanInputRiskSummary />
              </div>
            </div>
          </>
        )}

        <_SectionHeading className="">Advanced</_SectionHeading>
        <div className="mt-1.5">
          {_assert(getVisibility('expected-returns-and-volatility').visible) && (
            <div className="">
              <_SubSectionHeading className="" isFirst>
                Expected Returns
              </_SubSectionHeading>
              <div className="mt-2">
                <PlanInputExpectedReturnsAndVolatilitySummary />
              </div>
            </div>
          )}
          {_assert(getVisibility('inflation').visible) && (
            <div className="">
              <_SubSectionHeading className="">Inflation</_SubSectionHeading>
              <div className="mt-2">
                <PlanInputInflationSummary />
              </div>
            </div>
          )}

          {_assert(getVisibility('simulation').visible) && (
            <div className="">
              <_SubSectionHeading className="">Simulation</_SubSectionHeading>
              <div className="mt-2">
                <PlanInputSimulationSummary />
              </div>
            </div>
          )}

          {_assert(getVisibility('strategy').visible) && (
            <div className="">
              <_SubSectionHeading className="">Strategy</_SubSectionHeading>
              <div className="mt-2">
                <PlanInputStrategySummary />
              </div>
            </div>
          )}
        </div>
      </PlanPrintSection>
    </>
  )
})

const _SectionHeading = React.memo(
  ({ className, children }: { className?: string; children: ReactNode }) => {
    return (
      <h2
        className={clsx(
          className,
          'font-bold text-3xl text-right border-r-2 border-black px-5',
        )}
      >
        {children}
      </h2>
    )
  },
)
const _SubSectionHeading = React.memo(
  ({
    className,
    children,
    isFirst = false,
  }: {
    className?: string
    children: ReactNode
    isFirst?: boolean
  }) => {
    return (
      <h2 className={clsx(className, 'font-bold text-xl', !isFirst && 'mt-6')}>
        {children}
      </h2>
    )
  },
)

const _None = React.memo(() => (
  <div className="">
    <h2 className="">None</h2>
  </div>
))
