import { assert } from '@tpaw/common'
import clsx from 'clsx'
import React, { ReactNode } from 'react'
import { useGetPlanInputVisibility } from '../../Plan/PlanInput/Helpers/UseGetPlanInputVisibility'
import { PlanInputAgeSummary } from '../../Plan/PlanInput/PlanInputAge/PlanInputAge'
import { PlanInputCurrentPortfolioBalanceSummary } from '../../Plan/PlanInput/PlanInputCurrentPortfolioBalance'
import { PlanInputExpectedReturnsAndVolatilitySummary } from '../../Plan/PlanInput/PlanInputExpectedReturnsAndVolatility'
import { PlanInputExtraSpendingSummary } from '../../Plan/PlanInput/PlanInputExtraSpending'
import { PlanInputFutureSavingsSummary } from '../../Plan/PlanInput/PlanInputFutureSavings'
import { PlanInputIncomeDuringRetirementSummary } from '../../Plan/PlanInput/PlanInputIncomeDuringRetirement'
import { PlanInputInflationSummary } from '../../Plan/PlanInput/PlanInputInflation'
import { PlanInputLegacySummary } from '../../Plan/PlanInput/PlanInputLegacy'
import { PlanInputRiskSummary } from '../../Plan/PlanInput/PlanInputRisk/PlanInputRisk'
import { PlanInputSimulationSummary } from '../../Plan/PlanInput/PlanInputSimulation'
import { PlanInputSpendingCeilingAndFloorSummary } from '../../Plan/PlanInput/PlanInputSpendingCeilingAndFloor'
import { PlanInputStrategySummary } from '../../Plan/PlanInput/PlanInputStrategy'
import { useSimulationResultInfo } from '../WithSimulation'
import { PlanPrintViewPageGroup } from './Helpers/PlanPrintViewPageGroup'
import { PlanPrintViewSectionTitlePageGroup } from './Helpers/PlanPrintViewSectionTitlePageGroup'
import { PlanPrintViewArgs } from './PlanPrintViewArgs'

// Note: The show/hide should mirror PlanSummary. "None" displays should mirror
// PlanSummaryButtons's empty attribute from PlanSummary.
export const PlanPrintViewInputSection = React.memo(
  ({
    settings,
    currentPortfolioBalanceAmount,
  }: {
    currentPortfolioBalanceAmount: number
    settings: PlanPrintViewArgs['settings']
  }) => {
    const { planParamsNormOfResult } = useSimulationResultInfo().simulationResult

    const getVisibility = useGetPlanInputVisibility(planParamsNormOfResult)

    const _assert = (condition: boolean) => {
      assert(condition)
      return true
    }
    return (
      <>
        <PlanPrintViewSectionTitlePageGroup
          title="Your Inputs"
          settings={settings}
        />
        <PlanPrintViewPageGroup
          className="grid gap-y-12 gap-x-5"
          style={{ grid: 'auto / auto 1fr' }}
          settings={settings}
        >
          <_SectionHeading className="">Age</_SectionHeading>
          {_assert(getVisibility('age').visible) && (
            <div className="mt-1.5">
              <PlanInputAgeSummary planParamsNorm={planParamsNormOfResult} />
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
                  <PlanInputCurrentPortfolioBalanceSummary forPrint />
                </div>
              </div>
            )}
            {getVisibility('future-savings') && (
              <div className=" ">
                <_SubSectionHeading className="">
                  Future Savings
                </_SubSectionHeading>
                <div className="mt-2">
                  {planParamsNormOfResult.wealth.futureSavings.length > 0 ? (
                    <PlanInputFutureSavingsSummary
                      planParamsNorm={planParamsNormOfResult}
                    />
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
                  {planParamsNormOfResult.wealth.incomeDuringRetirement.length >
                  0 ? (
                    <PlanInputIncomeDuringRetirementSummary
                      planParamsNorm={planParamsNormOfResult}
                    />
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
            <PlanInputExtraSpendingSummary
              planParamsNorm={planParamsNormOfResult}
              forPrint
            />

            {getVisibility('legacy').visible && (
              <>
                <_SubSectionHeading className="">Legacy</_SubSectionHeading>
                <div className="mt-2">
                  {planParamsNormOfResult.adjustmentsToSpending.tpawAndSPAW
                    .legacy.external.length === 0 &&
                  planParamsNormOfResult.adjustmentsToSpending.tpawAndSPAW
                    .legacy.total === 0 ? (
                    <_None />
                  ) : (
                    <PlanInputLegacySummary
                      planParamsNorm={planParamsNormOfResult}
                    />
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
                  {planParamsNormOfResult.adjustmentsToSpending.tpawAndSPAW
                    .monthlySpendingCeiling === null &&
                  planParamsNormOfResult.adjustmentsToSpending.tpawAndSPAW
                    .monthlySpendingFloor === null ? (
                    <_None />
                  ) : (
                    <PlanInputSpendingCeilingAndFloorSummary
                      planParamsNorm={planParamsNormOfResult}
                    />
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
                  <PlanInputRiskSummary
                    planParamsNorm={planParamsNormOfResult}
                  />
                </div>
              </div>
            </>
          )}

          <_SectionHeading className="">Advanced</_SectionHeading>
          <div className="mt-1.5">
            {_assert(
              getVisibility('expected-returns-and-volatility').visible,
            ) && (
              <div className="">
                <_SubSectionHeading className="" isFirst>
                  Expected Returns
                </_SubSectionHeading>
                <div className="mt-2">
                  <PlanInputExpectedReturnsAndVolatilitySummary
                    planParamsNorm={planParamsNormOfResult}
                  />
                </div>
              </div>
            )}
            {_assert(getVisibility('inflation').visible) && (
              <div className="">
                <_SubSectionHeading className="">Inflation</_SubSectionHeading>
                <div className="mt-2">
                  <PlanInputInflationSummary
                    planParamsNorm={planParamsNormOfResult}
                  />
                </div>
              </div>
            )}

            {_assert(getVisibility('simulation').visible) && (
              <div className="">
                <_SubSectionHeading className="">Simulation</_SubSectionHeading>
                <div className="mt-2">
                  <PlanInputSimulationSummary
                    planParamsNorm={planParamsNormOfResult}
                  />
                </div>
              </div>
            )}

            {_assert(getVisibility('strategy').visible) && (
              <div className="">
                <_SubSectionHeading className="">Strategy</_SubSectionHeading>
                <div className="mt-2">
                  <PlanInputStrategySummary
                    planParamsNorm={planParamsNormOfResult}
                  />
                </div>
              </div>
            )}
          </div>
        </PlanPrintViewPageGroup>
      </>
    )
  },
)

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
