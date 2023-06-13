import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Params, noCase } from '@tpaw/common'
import React, { ReactNode, useMemo, useState } from 'react'
import { ParamsExtended } from '../../../TPAWSimulator/ExtentParams'
import {
  Padding,
  Size,
  XY,
  newPaddingHorz,
  originCSSStyle,
  paddingCSSStyleHorz,
  sizeCSSStyle,
} from '../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../Utils/NoDisplayOnOpacity0Transition'
import { useSimulation } from '../../App/WithSimulation'
import { Config } from '../../Config'
import { analyzeMonthsInParams } from '../PlanInput/Helpers/AnalyzeMonthsInParams'
import { PlanInputType } from '../PlanInput/Helpers/PlanInputType'
import { isPlanSectionDialogInOverlayMode } from '../PlanInput/Helpers/PlanSectionDialogPosition'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import {
  PlanTransitionState,
  simplifyPlanTransitionState4,
} from '../PlanTransition'
import {
  PlanSummaryButton,
  useShouldDisablePlanSummaryButton,
} from './PlanSummaryButton'
import { PlanSummaryDialog } from './PlanSummaryDialog'
import { PlanSummarySave } from './PlanSummarySave/PlanSummarySave'

type _FixedSizingByMode = {
  size: Size
  padding: ({ left: number; right: number } | { horz: number }) & {
    top: number
  }
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
  { label: 'dialogIn', sections: [{ name: 'summary', dialogMode: true }] },
  { label: 'dialogOut', sections: [{ name: 'rest', dialogMode: true }] },
  { label: 'notDialogIn', sections: [{ name: 'summary', dialogMode: false }] },
  { label: 'notDialogOut', sections: [{ name: 'rest', dialogMode: false }] },
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

    const { params, paramsExt, defaultParams } = useSimulation()
    const { allowFutureSavingsEntries } = paramsExt

    const [showAdvanced, setShowAdvanced] = useState(false)
    const [showDev, setShowDev] = useState(!Config.client.production)
    const [showDevClickCount, setShowDevClickCount] = useState(0)
    const advancedModifiedCount = _advancedInputs.filter((x) =>
      isAdvancedInputModified(x, params, defaultParams),
    ).length

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
        className={`absolute overflow-y-scroll`}
        style={{
          transitionProperty: 'transform, opacity',
          transitionDuration: `${planTransition.duration}ms`,
          transform: `translate(${targetSizing.origin.x}px,${targetSizing.origin.y}px)`,
          opacity: `${targetSizing.opacity}`,
          ...sizeCSSStyle(fixedSizing.size),
          ...originCSSStyle({ x: 0, y: 0 }),
          ...paddingCSSStyleHorz(newPaddingHorz(fixedSizing.padding)),
        }}
      >
        <div className="mt-0" ref={setBodyElement}>
          {params.plan.dialogPosition !== 'done' && (
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
              dialogPosition={params.plan.dialogPosition}
            />
          )}
          <div
            className="mb-4 w-full flex gap-x-4 justify-end items-center sticky top-0 z-20"
            style={{ marginTop: `${fixedSizing.padding.top}px` }}
          >
            <PlanSummarySave className="" />
          </div>
          <div
            className={`flex flex-col gap-y-12 sm:gap-y-16 relative z-0 w-full mb-20`}
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
                {(allowFutureSavingsEntries ||
                  params.plan.wealth.futureSavings.length > 0) && (
                  <PlanSummaryButton
                    ref={setFutureSavingsElement}
                    type="future-savings"
                    section={section}
                    warn={!_paramsOk(paramsExt, 'future-savings')}
                    padding={cardPadding}
                    empty={params.plan.wealth.futureSavings.length === 0}
                  />
                )}
                <PlanSummaryButton
                  ref={setIncomeDuringRetirementElement}
                  type="income-during-retirement"
                  section={section}
                  warn={!_paramsOk(paramsExt, 'income-during-retirement')}
                  padding={cardPadding}
                  empty={params.plan.wealth.retirementIncome.length === 0}
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
                  Adjustments to Spending
                </_Heading>
                <div className="flex flex-col gap-y-6 ">
                  <PlanSummaryButton
                    type="extra-spending"
                    section={section}
                    warn={!_paramsOk(paramsExt, 'extra-spending')}
                    padding={cardPadding}
                    empty={
                      params.plan.adjustmentsToSpending.extraSpending
                        .discretionary.length === 0 &&
                      params.plan.adjustmentsToSpending.extraSpending.essential
                        .length === 0
                    }
                  />

                  {params.plan.advanced.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="legacy"
                      section={section}
                      padding={cardPadding}
                      empty={
                        params.plan.adjustmentsToSpending.tpawAndSPAW.legacy
                          .external.length === 0 &&
                        params.plan.adjustmentsToSpending.tpawAndSPAW.legacy
                          .total === 0
                      }
                    />
                  )}
                  {params.plan.advanced.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="spending-ceiling-and-floor"
                      section={section}
                      padding={cardPadding}
                      empty={
                        params.plan.adjustmentsToSpending.tpawAndSPAW
                          .monthlySpendingCeiling === null &&
                        params.plan.adjustmentsToSpending.tpawAndSPAW
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
                    warn={!_paramsOk(paramsExt, 'risk')}
                    hideTitle
                  />
                </div>
              </div>

              <div className="">
                <button
                  className="disabled:opacity-20"
                  style={{ ...paddingCSSStyleHorz(cardPadding) }}
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  disabled={params.plan.dialogPosition !== 'done'}
                >
                  <div className="text-[20px] sm:text-[26px] font-bold text-left">
                    Advanced
                    <FontAwesomeIcon
                      className="ml-2"
                      icon={showAdvanced ? faCaretDown : faCaretRight}
                    />
                  </div>
                  {!showAdvanced && (
                    <h2 className="text-left">
                      {advancedModifiedCount === 0
                        ? 'None'
                        : `${advancedModifiedCount} modified`}
                    </h2>
                  )}
                </button>
                {showAdvanced && (
                  <div
                    className="flex flex-col gap-y-6 mt-6"
                    onClick={() => {
                      setShowDevClickCount((prev) => {
                        if (prev === 0)
                          window.setTimeout(() => setShowDevClickCount(0), 3000)
                        if (prev === 9) setShowDev((x) => !x)
                        return prev + 1
                      })
                    }}
                  >
                    <PlanSummaryButton
                      type="expected-returns"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={isAdvancedInputModified(
                        'expected-returns',
                        params,
                        defaultParams,
                      )}
                    />
                    <PlanSummaryButton
                      type="inflation"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={isAdvancedInputModified(
                        'inflation',
                        params,
                        defaultParams,
                      )}
                    />
                    <PlanSummaryButton
                      type="simulation"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={isAdvancedInputModified(
                        'simulation',
                        params,
                        defaultParams,
                      )}
                    />
                    <PlanSummaryButton
                      type="strategy"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={isAdvancedInputModified(
                        'strategy',
                        params,
                        defaultParams,
                      )}
                    />
                    {showDev && (
                      <PlanSummaryButton
                        type="dev"
                        section={section}
                        padding={cardPadding}
                      />
                    )}
                  </div>
                )}
              </div>
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
    firstItem: PlanInputType
    children: ReactNode
  }) => {
    const { params } = useSimulation()
    const isDisabled =
      useShouldDisablePlanSummaryButton(firstItem) &&
      !isPlanSectionDialogInOverlayMode(params.plan.dialogPosition)
    return (
      <h2
        className={`${className} text-[20px] sm:text-[26px] font-bold mb-6  transition-opacity
        ${isDisabled ? 'opacity-20' : ''}`}
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
export const _paramsOk = (paramsExt: ParamsExtended, type: _Type) => {
  const { valueForMonthRange, glidePath } = analyzeMonthsInParams(
    paramsExt.params.plan,
    paramsExt,
    {
      type: 'asVisible',
    },
  )
  return (
    valueForMonthRange
      .filter((x) => x.section === type)
      .every((x) =>
        x.boundsCheck
          ? x.boundsCheck.start === 'ok' && x.boundsCheck.end === 'ok'
          : true,
      ) &&
    glidePath
      .filter((x) => x.section === type)
      .every((x) => x.analyzed.every((x) => x.issue === 'none'))
  )
}

const _advancedInputs = [
  'expected-returns',
  'inflation',
  'strategy',
  'simulation',
] as const
type _AdvancedParamInputType = (typeof _advancedInputs)[number]
export const isAdvancedInputModified = (
  type: _AdvancedParamInputType,
  params: Params,
  def: Params,
) => {
  switch (type) {
    case 'expected-returns':
      return params.plan.advanced.annualReturns.expected.type !== 'suggested'
    case 'inflation':
      return params.plan.advanced.annualInflation.type !== 'suggested'
    case 'strategy':
      return params.plan.advanced.strategy !== def.plan.advanced.strategy
    case 'simulation':
      return (
        params.plan.advanced.sampling !== def.plan.advanced.sampling ||
        (params.plan.advanced.sampling === 'monteCarlo' &&
          params.plan.advanced.monteCarloSampling.blockSize !==
            def.plan.advanced.monteCarloSampling.blockSize)
      )
    default:
      noCase(type)
  }
}
