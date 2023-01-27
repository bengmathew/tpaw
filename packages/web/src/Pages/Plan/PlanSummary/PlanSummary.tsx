import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { getDefaultPlanParams, noCase, PlanParams } from '@tpaw/common'
import React, { ReactNode, useMemo, useState } from 'react'
import { PlanParamsExt } from '../../../TPAWSimulator/PlanParamsExt'
import {
  newPaddingHorz,
  originCSSStyle,
  Padding,
  paddingCSSStyleHorz,
  Size,
  sizeCSSStyle,
  XY,
} from '../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../Utils/NoDisplayOnOpacity0Transition'
import { useSimulation } from '../../App/WithSimulation'
import { Config } from '../../Config'
import { analyzeYearsInParams } from '../PlanInput/Helpers/AnalyzeYearsInParams'
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

    const { params, paramsExt } = useSimulation()
    const { asYFN, withdrawalStartYear } = paramsExt
    const isRetired = asYFN(withdrawalStartYear) <= 0

    const [showAdvanced, setShowAdvanced] = useState(false)
    const [showDev, setShowDev] = useState(!Config.client.production)
    const [showDevClickCount, setShowDevClickCount] = useState(0)
    const advancedModifiedCount = _advancedInputs.filter((x) =>
      _isModified(x, params),
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
          {params.dialogPosition !== 'done' && (
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
              dialogPosition={params.dialogPosition}
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
                {!isRetired && (
                  <PlanSummaryButton
                    ref={setFutureSavingsElement}
                    type="future-savings"
                    section={section}
                    warn={!_paramsOk(paramsExt, 'future-savings')}
                    padding={cardPadding}
                    empty={params.wealth.futureSavings.length === 0}
                  />
                )}
                <PlanSummaryButton
                  ref={setIncomeDuringRetirementElement}
                  type="income-during-retirement"
                  section={section}
                  warn={!_paramsOk(paramsExt, 'income-during-retirement')}
                  padding={cardPadding}
                  empty={params.wealth.retirementIncome.length === 0}
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
                    warn={!_paramsOk(paramsExt, 'extra-spending')}
                    padding={cardPadding}
                    empty={
                      params.adjustmentsToSpending.extraSpending.discretionary
                        .length === 0 &&
                      params.adjustmentsToSpending.extraSpending.essential
                        .length === 0
                    }
                  />

                  {params.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="legacy"
                      section={section}
                      padding={cardPadding}
                      empty={
                        params.adjustmentsToSpending.tpawAndSPAW.legacy.external
                          .length === 0 &&
                        params.adjustmentsToSpending.tpawAndSPAW.legacy
                          .total === 0
                      }
                    />
                  )}
                  {params.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="spending-ceiling-and-floor"
                      section={section}
                      padding={cardPadding}
                      empty={
                        params.adjustmentsToSpending.tpawAndSPAW
                          .spendingCeiling === null
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
                    hideTitle
                  />
                </div>
              </div>

              <div className="">
                <button
                  className="disabled:opacity-20"
                  style={{ ...paddingCSSStyleHorz(cardPadding) }}
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  disabled={params.dialogPosition !== 'done'}
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
                      if (showDev) return
                      setShowDevClickCount((prev) => {
                        if (prev === 0) {
                          window.setTimeout(() => setShowDevClickCount(0), 3000)
                        }
                        if (prev === 9) {
                          setShowDev(true)
                        }
                        return prev + 1
                      })
                    }}
                  >
                    <PlanSummaryButton
                      type="expected-returns"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={_isModified('expected-returns', params)}
                    />
                    <PlanSummaryButton
                      type="inflation"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={_isModified('inflation', params)}
                    />
                    <PlanSummaryButton
                      type="simulation"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={_isModified('simulation', params)}
                    />
                    <PlanSummaryButton
                      type="strategy"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={_isModified('strategy', params)}
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
      !isPlanSectionDialogInOverlayMode(params.dialogPosition)
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
  | 'strategy'
export const _paramsOk = (paramsExt: PlanParamsExt, type: _Type) => {
  const { valueForYearRange, glidePath } = analyzeYearsInParams(paramsExt)
  return (
    valueForYearRange
      .filter((x) => x.section === type)
      .every(
        (x) => x.boundsCheck.start === 'ok' && x.boundsCheck.end === 'ok',
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
type _AdvancedParamInputType = typeof _advancedInputs[number]
const _isModified = (type: _AdvancedParamInputType, params: PlanParams) => {
  const def = getDefaultPlanParams()
  switch (type) {
    case 'expected-returns':
      return params.returns.expected.type !== 'suggested'
    case 'inflation':
      return params.inflation.type !== 'suggested'
    case 'strategy':
      return params.strategy !== def.strategy
    case 'simulation':
      return params.sampling !== def.sampling
    default:
      noCase(type)
  }
}
