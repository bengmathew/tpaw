import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PlanParams, assert, noCase } from '@tpaw/common'
import _ from 'lodash'
import React, { ReactNode, useMemo, useState } from 'react'
import { PlanParamsExtended } from '../../../../TPAWSimulator/ExtentPlanParams'
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
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { analyzeMonthsInParams } from '../PlanInput/Helpers/AnalyzeMonthsInParams'
import { PlanInputType } from '../PlanInput/Helpers/PlanInputType'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import { useGetPlanInputVisibility } from '../PlanInput/Helpers/UseGetPlanInputVisibility'
import { useIsPlanInputDevAdditionalSpendingTiltModified } from '../PlanInput/PlanInputDev/PlanInputDevAdditionalSpendingTilt'
import { useIsPlanInputDevHistoricalReturnsModified } from '../PlanInput/PlanInputDev/PlanInputDevHistoricalReturns'
import { useIsPlanInputDevMiscModified } from '../PlanInput/PlanInputDev/PlanInputDevMisc'
import { useIsPlanInputDevSimulationsModified } from '../PlanInput/PlanInputDev/PlanInputDevSimulations'
import { useIsPlanInputDevTimeModified } from '../PlanInput/PlanInputDev/PlanInputDevTime'
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

    const {
      planParams,
      planParamsExt,
      defaultPlanParams,
      simulationInfoByMode,
    } = useSimulation()
    const { isFutureSavingsAllowed } = planParamsExt

    const [expandAdvanced, setExpandAdvanced] = useState(false)
    const [showDev, setShowDev] = useState(() => DevMode.readDevMode())
    const [showDevClickCount, setShowDevClickCount] = useState(0)
    const [expandDev, setExpandDev] = useState(false)
    const advancedModifiedCount = _advancedInputs.filter((x) =>
      isAdvancedInputModified(x, planParams, defaultPlanParams),
    ).length

    const devModifiedByType = {
      'dev-misc': useIsPlanInputDevMiscModified(),
      'dev-simulations': useIsPlanInputDevSimulationsModified(),
      'dev-time': useIsPlanInputDevTimeModified(),
      'dev-historical-returns': useIsPlanInputDevHistoricalReturnsModified(),
      'dev-additional-spending-tilt':
        useIsPlanInputDevAdditionalSpendingTiltModified(),
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
              DevMode.setDevMode(!showDev)
              setShowDev(!showDev)
            }
            return prev + 1
          })
        }}
      >
        <div className="mt-0" ref={setBodyElement}>
          {planParams.dialogPosition !== 'done' && (
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
              dialogPosition={planParams.dialogPosition}
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
                {(isFutureSavingsAllowed ||
                  _.values(planParams.wealth.futureSavings).length > 0) && (
                  <PlanSummaryButton
                    ref={setFutureSavingsElement}
                    type="future-savings"
                    section={section}
                    warn={!_paramsOk(planParamsExt, 'future-savings')}
                    padding={cardPadding}
                    empty={
                      _.values(planParams.wealth.futureSavings).length === 0
                    }
                  />
                )}
                <PlanSummaryButton
                  ref={setIncomeDuringRetirementElement}
                  type="income-during-retirement"
                  section={section}
                  warn={!_paramsOk(planParamsExt, 'income-during-retirement')}
                  padding={cardPadding}
                  empty={
                    _.values(planParams.wealth.incomeDuringRetirement)
                      .length === 0
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
                    warn={!_paramsOk(planParamsExt, 'extra-spending')}
                    padding={cardPadding}
                    empty={
                      _.values(
                        planParams.adjustmentsToSpending.extraSpending
                          .discretionary,
                      ).length === 0 &&
                      _.values(
                        planParams.adjustmentsToSpending.extraSpending
                          .essential,
                      ).length === 0
                    }
                  />

                  {planParams.advanced.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="legacy"
                      section={section}
                      padding={cardPadding}
                      empty={
                        _.values(
                          planParams.adjustmentsToSpending.tpawAndSPAW.legacy
                            .external,
                        ).length === 0 &&
                        planParams.adjustmentsToSpending.tpawAndSPAW.legacy
                          .total === 0
                      }
                    />
                  )}
                  {planParams.advanced.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="spending-ceiling-and-floor"
                      section={section}
                      padding={cardPadding}
                      empty={
                        planParams.adjustmentsToSpending.tpawAndSPAW
                          .monthlySpendingCeiling === null &&
                        planParams.adjustmentsToSpending.tpawAndSPAW
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
                    warn={!_paramsOk(planParamsExt, 'risk')}
                    hideTitle
                  />
                </div>
              </div>

              <div className="">
                <button
                  className="disabled:opacity-20"
                  style={{ ...paddingCSSStyleHorz(cardPadding) }}
                  onClick={() => setExpandAdvanced(!expandAdvanced)}
                  disabled={planParams.dialogPosition !== 'done'}
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
                      type="expected-returns"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={isAdvancedInputModified(
                        'expected-returns',
                        planParams,
                        defaultPlanParams,
                      )}
                    />
                    <PlanSummaryButton
                      type="inflation"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={isAdvancedInputModified(
                        'inflation',
                        planParams,
                        defaultPlanParams,
                      )}
                    />
                    <PlanSummaryButton
                      type="simulation"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={isAdvancedInputModified(
                        'simulation',
                        planParams,
                        defaultPlanParams,
                      )}
                    />
                    <PlanSummaryButton
                      type="strategy"
                      section={section}
                      padding={cardPadding}
                      flagAsModified={isAdvancedInputModified(
                        'strategy',
                        planParams,
                        defaultPlanParams,
                      )}
                    />
                  </div>
                )}
              </div>
              {showDev && (
                <div className="">
                  <button
                    className="disabled:opacity-20"
                    style={{ ...paddingCSSStyleHorz(cardPadding) }}
                    onClick={() => setExpandDev(!expandDev)}
                    disabled={planParams.dialogPosition !== 'done'}
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
                    <div
                      className="flex flex-col gap-y-6 mt-6"
                      onClick={() => {
                        setShowDevClickCount((prev) => {
                          if (prev === 0)
                            window.setTimeout(
                              () => setShowDevClickCount(0),
                              3000,
                            )
                          if (prev === 9) setShowDev((x) => !x)
                          return prev + 1
                        })
                      }}
                    >
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
                        type="dev-historical-returns"
                        section={section}
                        padding={cardPadding}
                        flagAsModified={
                          devModifiedByType['dev-historical-returns']
                        }
                      />
                      <PlanSummaryButton
                        type="dev-time"
                        section={section}
                        padding={cardPadding}
                        flagAsModified={devModifiedByType['dev-time']}
                      />
                      <PlanSummaryButton
                        type="dev-additional-spending-tilt"
                        section={section}
                        padding={cardPadding}
                        flagAsModified={
                          devModifiedByType['dev-additional-spending-tilt']
                        }
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
    const visibility = useGetPlanInputVisibility()(firstItem)
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
export const _paramsOk = (planParamsExt: PlanParamsExtended, type: _Type) => {
  const { valueForMonthRange, glidePath } = analyzeMonthsInParams(
    planParamsExt,
    { type: 'asVisible' },
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
  planParams: PlanParams,
  def: PlanParams,
) => {
  switch (type) {
    case 'expected-returns':
      return planParams.advanced.annualReturns.expected.type !== 'suggested'
    case 'inflation':
      return planParams.advanced.annualInflation.type !== 'suggested'
    case 'strategy':
      return planParams.advanced.strategy !== def.advanced.strategy
    case 'simulation':
      return (
        planParams.advanced.sampling !== def.advanced.sampling ||
        (planParams.advanced.sampling === 'monteCarlo' &&
          planParams.advanced.monteCarloSampling.blockSize !==
            def.advanced.monteCarloSampling.blockSize)
      )
    default:
      noCase(type)
  }
}

export namespace DevMode {
  const key = 'DevMode'
  export const readDevMode = () => {
    return !Config.client.production || localStorage.getItem(key) === 'true'
  }
  export const setDevMode = (devMode: boolean) => {
    return localStorage.setItem(key, devMode ? 'true' : 'false')
  }
}
