import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { MonthLocation, assert, noCase } from '@tpaw/common'
import _ from 'lodash'
import React, { ReactNode, useMemo, useState } from 'react'
import { NormalizedGlidePath } from '../../../../UseSimulator/NormalizePlanParams/NormalizeGlidePath'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { NormalizedLabeledAmountTimed } from '../../../../UseSimulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import {
  Padding,
  Size,
  XY,
  newPaddingHorz,
  originCSSStyle,
  paddingCSSStyleHorz,
  sizeCSSStyle,
} from '../../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../../Utils/NoDisplayOnOpacity0Transition'
import { Config } from '../../../Config'
import { useNonPlanParams } from '../../PlanRootHelpers/WithNonPlanParams'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { PlanInputType } from '../PlanInput/Helpers/PlanInputType'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import { useGetPlanInputVisibility } from '../PlanInput/Helpers/UseGetPlanInputVisibility'
import { useIsPlanInputDevMiscModified } from '../PlanInput/PlanInputDev/PlanInputDevMisc'
import { useIsPlanInputDevTimeModified } from '../PlanInput/PlanInputDev/PlanInputDevTime'
import { useIsPlanInputDevSimulationsModified } from '../PlanInput/PlanInputDev/UseIsPlanInputDevSimulationsModified'
import { useIsPlanInputExpectedReturnsAndVolatilityModified } from '../PlanInput/PlanInputExpectedReturnsAndVolatility'
import { useIsPlanInputInflationModified } from '../PlanInput/PlanInputInflation'
import { useIsPlanInputSimulationModifed } from '../PlanInput/PlanInputSimulation'
import { useIsPlanInputStrategyModified } from '../PlanInput/PlanInputStrategy'
import {
  PlanTransitionState,
  simplifyPlanTransitionState4,
} from '../PlanTransition'
import { PlanSummaryButton } from './PlanSummaryButton'
import { PlanSummaryDialog } from './PlanSummaryDialog'

type _FixedSizingByMode = {
  size: Size
  padding:
    | { left: number; right: number; top: number }
    | { horz: number; top: number }
}
export type PlanSummaySizing = {
  dynamic: Record<PlanSummaryTransitionState, { origin: XY; opacity: number }>
  fixed: {
    dialogMode: _FixedSizingByMode
    notDialogMode: _FixedSizingByMode
    cardPadding: Padding
  }
}

const _toPlanSummaryTransitionState = simplifyPlanTransitionState4(
  {
    label: 'dialogIn',
    sections: [{ section: 'summary', dialogMode: true }],
  },
  {
    label: 'dialogOut',
    sections: [{ section: 'rest', dialogMode: true }],
  },
  {
    label: 'notDialogIn',
    sections: [{ section: 'summary', dialogMode: false }],
  },
  {
    label: 'notDialogOut',
    sections: [{ section: 'rest', dialogMode: false }],
  },
)
export type PlanSummaryTransitionState = ReturnType<
  typeof _toPlanSummaryTransitionState
>

export const PlanSummary = React.memo(
  ({
    section,
    sizing,
    planTransition,
  }: {
    section: PlanSectionName
    sizing: PlanSummaySizing
    planTransition: { target: PlanTransitionState; duration: number }
  }) => {
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const [outerElement, setOuterElement] = useState<HTMLElement | null>(null)
    const [bodyElement, setBodyElement] = useState<HTMLElement | null>(null)
    const [ageElement, setAgeElement] = useState<HTMLElement | null>(null)
    const [currentPortfolioBalanceElement, setCurrentPortfolioBalanceElement] =
      useState<HTMLElement | null>(null)
    const [futureSavingsElement, setFutureSavingsElement] =
      useState<HTMLElement | null>(null)
    const [incomeDuringRetirementElement, setIncomeDuringRetirementElement] =
      useState<HTMLElement | null>(null)
    const [adjustmentsToSpendingElement, setAdjustmentsToSpendingElement] =
      useState<HTMLElement | null>(null)

    const { planParamsNorm } = useSimulation()
    const { dialogPosition } = planParamsNorm

    const [expandAdvanced, setExpandAdvanced] = useState(false)
    const [showDevClickCount, setShowDevClickCount] = useState(0)
    const [expandDev, setExpandDev] = useState(false)

    const advancedModifiedByType = {
      'expected-returns-and-volatility':
        useIsPlanInputExpectedReturnsAndVolatilityModified(),
      inflation: useIsPlanInputInflationModified(),
      strategy: useIsPlanInputStrategyModified(),
      simulation: useIsPlanInputSimulationModifed(),
    }
    const advancedModifiedCount = _.values(advancedModifiedByType).filter(
      (x) => x,
    ).length

    const devModifiedByType = {
      'dev-misc': useIsPlanInputDevMiscModified(),
      'dev-simulations': useIsPlanInputDevSimulationsModified(),
      'dev-time': useIsPlanInputDevTimeModified(),
    }
    const devModifiedCount = _.values(devModifiedByType).filter((x) => x).length

    const targetSizing = useMemo(
      () =>
        sizing.dynamic[_toPlanSummaryTransitionState(planTransition.target)],
      [planTransition.target, sizing],
    )

    const fixedSizing = planTransition.target.dialogMode
      ? sizing.fixed.dialogMode
      : sizing.fixed.notDialogMode

    const cardPadding = sizing.fixed.cardPadding

    return (
      <NoDisplayOnOpacity0Transition
        ref={setOuterElement}
        // Don't destroy this because otherwise scroll position will be lost.
        noDisplayMeans="visibility:hidden"
        className={`absolute overflow-y-auto `}
        style={{
          transitionProperty: 'transform, opacity',
          transitionDuration: `${planTransition.duration}ms`,
          transform: `translate(${targetSizing.origin.x}px,${targetSizing.origin.y}px)`,
          opacity: `${targetSizing.opacity}`,
          ...sizeCSSStyle(fixedSizing.size),
          ...originCSSStyle({ x: 0, y: 0 }),
          ...paddingCSSStyleHorz(newPaddingHorz(fixedSizing.padding)),
          paddingTop: `${fixedSizing.padding.top}px`,
        }}
        onClick={() => {
          setShowDevClickCount((prev) => {
            if (prev === 0)
              window.setTimeout(() => setShowDevClickCount(0), 3000)
            if (prev === 9) {
              const clone = _.cloneDeep(nonPlanParams)
              clone.dev.showDevFeatures = !clone.dev.showDevFeatures
              setNonPlanParams(clone)
            }
            return prev + 1
          })
        }}
      >
        <div className="mt-0" ref={setBodyElement}>
          {dialogPosition.effective !== 'done' && (
            <PlanSummaryDialog
              elements={{
                outer: outerElement,
                body: bodyElement,
                age: ageElement,
                currentPortfolioBalance: currentPortfolioBalanceElement,
                futureSavings: futureSavingsElement,
                incomeDuringRetirement: incomeDuringRetirementElement,
                adjustmentsToSpending: adjustmentsToSpendingElement,
              }}
              fixedSizing={fixedSizing}
              dialogPosition={dialogPosition.effective}
            />
          )}

          <div
            className={`flex flex-col gap-y-12 sm:gap-y-16 relative z-0 w-full  mb-20`}
          >
            <div className="">
              <_Heading cardPadding={cardPadding} firstItem="age">
                Age
              </_Heading>
              <div className="flex flex-col gap-y-6 ">
                <PlanSummaryButton
                  ref={setAgeElement}
                  type="age"
                  section={section}
                  padding={cardPadding}
                  hideTitle
                />
              </div>
            </div>

            <div className="">
              <_Heading
                cardPadding={cardPadding}
                firstItem="current-portfolio-balance"
              >
                Wealth
              </_Heading>
              <div className="flex flex-col gap-y-6 ">
                <PlanSummaryButton
                  ref={setCurrentPortfolioBalanceElement}
                  type="current-portfolio-balance"
                  section={section}
                  padding={cardPadding}
                />
                {planParamsNorm.ages.validMonthRangesAsMFN.futureSavings ? (
                  <PlanSummaryButton
                    ref={setFutureSavingsElement}
                    type="future-savings"
                    section={section}
                    warn={!_paramsOk(planParamsNorm, 'future-savings')}
                    padding={cardPadding}
                    empty={planParamsNorm.wealth.futureSavings.length === 0}
                  />
                ) : (
                  // This avoids managing null/non-null future savings element
                  // in the PlanSummaryDialog, which was very error prone.
                  <div
                    ref={setFutureSavingsElement}
                    className="h-0 w-0 hidden"
                  />
                )}
                <PlanSummaryButton
                  ref={setIncomeDuringRetirementElement}
                  type="income-during-retirement"
                  section={section}
                  warn={!_paramsOk(planParamsNorm, 'income-during-retirement')}
                  padding={cardPadding}
                  empty={
                    planParamsNorm.wealth.incomeDuringRetirement.length === 0
                  }
                />
              </div>
            </div>

            <div
              id="planSummaryInputsAfterDialog"
              ref={setAdjustmentsToSpendingElement}
              className="flex flex-col gap-y-12 sm:gap-y-16 "
            >
              <div className="">
                <_Heading cardPadding={cardPadding} firstItem="extra-spending">
                  Adjustments To Spending
                </_Heading>
                <div className="flex flex-col gap-y-6 ">
                  <PlanSummaryButton
                    type="extra-spending"
                    section={section}
                    warn={!_paramsOk(planParamsNorm, 'extra-spending')}
                    padding={cardPadding}
                    empty={
                      planParamsNorm.adjustmentsToSpending.extraSpending
                        .discretionary.length === 0 &&
                      planParamsNorm.adjustmentsToSpending.extraSpending
                        .essential.length === 0
                    }
                  />

                  {planParamsNorm.advanced.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="legacy"
                      section={section}
                      padding={cardPadding}
                      empty={
                        planParamsNorm.adjustmentsToSpending.tpawAndSPAW.legacy
                          .external.length === 0 &&
                        planParamsNorm.adjustmentsToSpending.tpawAndSPAW.legacy
                          .total === 0
                      }
                    />
                  )}
                  {planParamsNorm.advanced.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="spending-ceiling-and-floor"
                      section={section}
                      padding={cardPadding}
                      empty={
                        planParamsNorm.adjustmentsToSpending.tpawAndSPAW
                          .monthlySpendingCeiling === null &&
                        planParamsNorm.adjustmentsToSpending.tpawAndSPAW
                          .monthlySpendingFloor === null
                      }
                    />
                  )}
                </div>
              </div>

              <div className="">
                <_Heading cardPadding={cardPadding} firstItem="risk">
                  Risk
                </_Heading>
                <div className="flex flex-col gap-y-6 ">
                  <PlanSummaryButton
                    type="risk"
                    section={section}
                    padding={cardPadding}
                    warn={!_paramsOk(planParamsNorm, 'risk')}
                    hideTitle
                  />
                </div>
              </div>

              <div className="">
                <button
                  className="disabled:opacity-20"
                  style={{ ...paddingCSSStyleHorz(cardPadding) }}
                  onClick={() => setExpandAdvanced(!expandAdvanced)}
                  disabled={dialogPosition.effective !== 'done'}
                >
                  <div className="text-[20px] sm:text-[26px] font-bold text-left">
                    Advanced
                    <FontAwesomeIcon
                      className="ml-2"
                      icon={expandAdvanced ? faCaretDown : faCaretRight}
                    />
                  </div>
                  {!expandAdvanced && (
                    <h2 className="text-left">
                      {advancedModifiedCount === 0
                        ? 'None'
                        : `${advancedModifiedCount} modified`}
                    </h2>
                  )}
                </button>
                {expandAdvanced && (
                  <div className="flex flex-col gap-y-6 mt-6">
                    <PlanSummaryButton
                      type="expected-returns-and-volatility"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={
                        advancedModifiedByType[
                          'expected-returns-and-volatility'
                        ]
                      }
                    />
                    <PlanSummaryButton
                      type="inflation"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={advancedModifiedByType['inflation']}
                    />
                    <PlanSummaryButton
                      type="simulation"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={advancedModifiedByType['simulation']}
                    />
                    <PlanSummaryButton
                      type="strategy"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={advancedModifiedByType['strategy']}
                    />
                  </div>
                )}
              </div>
              {(!Config.client.isProduction ||
                nonPlanParams.dev.showDevFeatures) && (
                <div className="">
                  <button
                    className="disabled:opacity-20"
                    style={{ ...paddingCSSStyleHorz(cardPadding) }}
                    onClick={() => setExpandDev(!expandDev)}
                    disabled={dialogPosition.effective !== 'done'}
                  >
                    <div className="text-[20px] sm:text-[26px] font-bold text-left">
                      Dev
                      <FontAwesomeIcon
                        className="ml-2"
                        icon={expandDev ? faCaretDown : faCaretRight}
                      />
                    </div>
                    {!expandDev && (
                      <h2 className="text-left">
                        {devModifiedCount === 0
                          ? 'None'
                          : `${devModifiedCount} modified`}
                      </h2>
                    )}
                  </button>
                  {expandDev && (
                    <div className="flex flex-col gap-y-6 mt-6">
                      <PlanSummaryButton
                        type="dev-misc"
                        section={section}
                        padding={cardPadding}
                        flagAsModified={devModifiedByType['dev-misc']}
                      />
                      <PlanSummaryButton
                        type="dev-simulations"
                        section={section}
                        padding={cardPadding}
                        flagAsModified={devModifiedByType['dev-simulations']}
                      />
                      <PlanSummaryButton
                        type="dev-time"
                        section={section}
                        padding={cardPadding}
                        flagAsModified={devModifiedByType['dev-time']}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </NoDisplayOnOpacity0Transition>
    )
  },
)

const _Heading = React.memo(
  ({
    className = '',
    cardPadding,
    firstItem,
    children,
  }: {
    className?: string
    cardPadding: Padding
    firstItem: Exclude<PlanInputType, 'history'>
    children: ReactNode
  }) => {
    const { planParamsNorm } = useSimulation()
    const visibility = useGetPlanInputVisibility(planParamsNorm)(firstItem)
    // This just happens to be true for now. Will have to pass more than just
    // firstItem it if is not true. Deal with that when if we end up needing it
    // in the future.
    assert(visibility.visible)
    return (
      <h2
        className={`${className} text-[20px] sm:text-[26px] font-bold mb-6  transition-opacity
        ${visibility.grayOutButton ? 'opacity-20' : ''}`}
        style={{ ...paddingCSSStyleHorz(cardPadding) }}
      >
        {children}
      </h2>
    )
  },
)

type _Type =
  | 'future-savings'
  | 'income-during-retirement'
  | 'extra-spending'
  | 'risk'
export const _paramsOk = (
  planParamsNorm: PlanParamsNormalized,
  type: _Type,
) => {
  // To make ensure when new month locations are added we don't forget to
  // update this function.
  const __ = (x: MonthLocation) => {
    switch (x) {
      case 'futureSavings':
      case 'incomeDuringRetirement':
      case 'extraSpendingEssential':
      case 'extraSpendingDiscretionary':
      case 'spawAndSWRStockAllocation':
        return
      default:
        noCase(x)
    }
  }

  const _forLabeledAmountTimed = (x: NormalizedLabeledAmountTimed[]) =>
    x.every((x) => {
      switch (x.amountAndTiming.type) {
        case 'inThePast':
          return true
        case 'oneTime':
          return !!x.amountAndTiming.month.errorMsg
        case 'recurring':
          switch (x.amountAndTiming.monthRange.type) {
            case 'startAndEnd':
              return (
                !x.amountAndTiming.monthRange.start.errorMsg &&
                !x.amountAndTiming.monthRange.end.errorMsg
              )
            case 'startAndDuration':
              return (
                !x.amountAndTiming.monthRange.start.errorMsg &&
                !x.amountAndTiming.monthRange.duration.errorMsg
              )
            case 'endAndDuration':
              return (
                !x.amountAndTiming.monthRange.end.errorMsg &&
                !x.amountAndTiming.monthRange.duration.errorMsg
              )
            default:
              noCase(x.amountAndTiming.monthRange)
          }
        default:
          noCase(x.amountAndTiming)
      }
    })
  const _forGlidePath = (glidePath: NormalizedGlidePath) =>
    glidePath.atOrPastEnd.length === 0 &&
    glidePath.intermediate.every((x) => !x.month.errorMsg)

  switch (type) {
    case 'future-savings':
      return _forLabeledAmountTimed(planParamsNorm.wealth.futureSavings)
    case 'income-during-retirement':
      return _forLabeledAmountTimed(
        planParamsNorm.wealth.incomeDuringRetirement,
      )
    case 'extra-spending':
      return _forLabeledAmountTimed([
        ...planParamsNorm.adjustmentsToSpending.extraSpending.essential,
        ...planParamsNorm.adjustmentsToSpending.extraSpending.discretionary,
      ])
    case 'risk':
      return _forGlidePath(planParamsNorm.risk.spawAndSWR.allocation)
    default:
      noCase(type)
  }
}
