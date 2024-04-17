import { assert } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { ReactNode } from 'react'
import { useGetPlanInputVisibility } from '../../Plan/PlanInput/Helpers/UseGetPlanInputVisibility'
import { PlanInputAgeSummary } from '../../Plan/PlanInput/PlanInputAge/PlanInputAge'
import { PlanInputCurrentPortfolioBalanceSummary } from '../../Plan/PlanInput/PlanInputCurrentPortfolioBalance'
import { PlanInputExtraSpendingSummary } from '../../Plan/PlanInput/PlanInputExtraSpending'
import { PlanInputFutureSavingsSummary } from '../../Plan/PlanInput/PlanInputFutureSavings'
import { PlanInputIncomeDuringRetirementSummary } from '../../Plan/PlanInput/PlanInputIncomeDuringRetirement'
import { PlanInputInflationSummary } from '../../Plan/PlanInput/PlanInputInflation'
import { PlanInputLegacySummary } from '../../Plan/PlanInput/PlanInputLegacy'
import { PlanInputRiskSummary } from '../../Plan/PlanInput/PlanInputRisk/PlanInputRisk'
import { PlanInputSimulationSummary } from '../../Plan/PlanInput/PlanInputSimulation'
import { PlanInputSpendingCeilingAndFloorSummary } from '../../Plan/PlanInput/PlanInputSpendingCeilingAndFloor'
import { PlanInputStrategySummary } from '../../Plan/PlanInput/PlanInputStrategy'
import { useSimulationResult } from '../WithSimulation'
import { PlanPrintViewPageGroup } from './Helpers/PlanPrintViewPageGroup'
import { PlanPrintViewSectionTitlePageGroup } from './Helpers/PlanPrintViewSectionTitlePageGroup'
import { PlanPrintViewArgs } from './PlanPrintViewArgs'
import { CurrentPortfolioBalance } from '../CurrentPortfolioBalance'
import { PlanInputExpectedReturnsAndVolatilitySummary } from '../../Plan/PlanInput/PlanInputExpectedReturnsAndVolatility/PlanInputExpectedReturnsAndVolatility'

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
    const { args } = useSimulationResult()

    const getVisibility = useGetPlanInputVisibility(args.planParamsNorm)

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
              <PlanInputAgeSummary planParamsNorm={args.planParamsNorm} />
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
                  <PlanInputCurrentPortfolioBalanceSummary
                    amountInfo={
                      args.planParamsNorm.datingInfo.isDated
                        ? {
                            isDatedPlan: true,
                            info: {
                              isEstimate: false,
                              amount: currentPortfolioBalanceAmount,
                            },
                          }
                        : {
                            isDatedPlan: false,
                            amount: currentPortfolioBalanceAmount,
                          }
                    }
                    forPrint
                  />
                </div>
              </div>
            )}
            {getVisibility('future-savings') && (
              <div className=" ">
                <_SubSectionHeading className="">
                  Future Savings
                </_SubSectionHeading>
                <div className="mt-2">
                  {args.planParamsNorm.wealth.futureSavings.length > 0 ? (
                    <PlanInputFutureSavingsSummary
                      planParamsNorm={args.planParamsNorm}
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
                  {args.planParamsNorm.wealth.incomeDuringRetirement.length >
                  0 ? (
                    <PlanInputIncomeDuringRetirementSummary
                      planParamsNorm={args.planParamsNorm}
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
              planParamsNorm={args.planParamsNorm}
              forPrint
            />

            {getVisibility('legacy').visible && (
              <>
                <_SubSectionHeading className="">Legacy</_SubSectionHeading>
                <div className="mt-2">
                  {args.planParamsNorm.adjustmentsToSpending.tpawAndSPAW.legacy
                    .external.length === 0 &&
                  args.planParamsNorm.adjustmentsToSpending.tpawAndSPAW.legacy
                    .total === 0 ? (
                    <_None />
                  ) : (
                    <PlanInputLegacySummary
                      planParamsNorm={args.planParamsNorm}
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
                  {args.planParamsNorm.adjustmentsToSpending.tpawAndSPAW
                    .monthlySpendingCeiling === null &&
                  args.planParamsNorm.adjustmentsToSpending.tpawAndSPAW
                    .monthlySpendingFloor === null ? (
                    <_None />
                  ) : (
                    <PlanInputSpendingCeilingAndFloorSummary
                      planParamsNorm={args.planParamsNorm}
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
                  <PlanInputRiskSummary planParamsNorm={args.planParamsNorm} />
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
                    planParamsNorm={args.planParamsNorm}
                  />
                </div>
              </div>
            )}
            {_assert(getVisibility('inflation').visible) && (
              <div className="">
                <_SubSectionHeading className="">Inflation</_SubSectionHeading>
                <div className="mt-2">
                  <PlanInputInflationSummary
                    planParamsNorm={args.planParamsNorm}
                  />
                </div>
              </div>
            )}

            {_assert(getVisibility('simulation').visible) && (
              <div className="">
                <_SubSectionHeading className="">Simulation</_SubSectionHeading>
                <div className="mt-2">
                  <PlanInputSimulationSummary
                    planParamsNorm={args.planParamsNorm}
                  />
                </div>
              </div>
            )}

            {_assert(getVisibility('strategy').visible) && (
              <div className="">
                <_SubSectionHeading className="">Strategy</_SubSectionHeading>
                <div className="mt-2">
                  <PlanInputStrategySummary
                    planParamsNorm={args.planParamsNorm}
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
