import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { getDefaultPlanParams, PlanParams } from '@tpaw/common'
import React, { useMemo, useState } from 'react'
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
import { noCase } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { Config } from '../../Config'
import { analyzeYearsInParams } from '../PlanInput/Helpers/AnalyzeYearsInParams'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import {
  PlanTransitionState,
  simplifyPlanTransitionState2,
} from '../PlanTransition'
import { PlanSummaryButton } from './PlanSummaryButton'
import { PlanSummaryRiskCopyFromAPreset } from './PlanSummaryRiskCopyFromAPreset'
import { PlanSummaryRiskSwitchMode } from './PlanSummaryRiskSwitchMode'
import { PlanSummarySave } from './PlanSummarySave/PlanSummarySave'

export type PlanSummaySizing = {
  dynamic: Record<
    PlanSummaryTransitionState,
    {
      origin: XY
      opacity: number
    }
  >
  fixed: {
    size: Size
    padding: ({ left: number; right: number } | { horz: number }) & {
      top: number
    }
    cardPadding: Padding
  }
}

const _toPlanSummaryTransitionState = simplifyPlanTransitionState2(
  { label: 'in', sections: [{ name: 'summary', dialogMode: 'any' }] },
  { label: 'out', sections: [{ name: 'rest', dialogMode: 'any' }] },
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
    const { params, paramsExt } = useSimulation()
    const { asYFN, withdrawalStartYear } = paramsExt
    const isRetired = asYFN(withdrawalStartYear) <= 0

    const [showAdvanced, setShowAdvanced] = useState(false)
    const advancedModifiedCount = _advancedInputs.filter((x) =>
      _isModified(x, params),
    ).length

    const targetSizing = useMemo(
      () =>
        sizing.dynamic[_toPlanSummaryTransitionState(planTransition.target)],
      [planTransition.target, sizing],
    )

    return (
      <NoDisplayOnOpacity0Transition
        // Don't destroy this because otherwise scroll position will be lost.
        noDisplayMeans="visibility:hidden"
        className={`absolute overflow-y-scroll`}
        style={{
          transitionProperty: 'transform, opacity',
          transitionDuration: `${planTransition.duration}ms`,
          transform: `translate(${targetSizing.origin.x}px,${targetSizing.origin.y}px)`,
          opacity: `${targetSizing.opacity}`,
          ...sizeCSSStyle(sizing.fixed.size),
          ...originCSSStyle({ x: 0, y: 0 }),
          ...paddingCSSStyleHorz(newPaddingHorz(sizing.fixed.padding)),
        }}
      >
        <div
          className="mb-4 w-full flex gap-x-4 justify-end items-center sticky top-0 z-10"
          style={{
            marginTop: `${sizing.fixed.padding.top}px`,
          }}
        >
          {/* <PlanSummaryReset />
          <PlanSummaryShare /> */}
          <PlanSummarySave className="" />
        </div>
        <div
          className={`flex flex-col gap-y-12 sm:gap-y-16 relative z-0 w-full mb-20`}
        >
          <div className="">
            <h2
              className="text-[20px] sm:text-[26px] font-bold mb-6"
              style={{ ...paddingCSSStyleHorz(sizing.fixed.cardPadding) }}
            >
              Basic Inputs
            </h2>
            <div className="flex flex-col gap-y-6 ">
              <PlanSummaryButton
                type="age"
                section={section}
                padding={sizing.fixed.cardPadding}
              />
              <PlanSummaryButton
                type="current-portfolio-balance"
                section={section}
                padding={sizing.fixed.cardPadding}
              />
              {!isRetired && (
                <PlanSummaryButton
                  type="future-savings"
                  section={section}
                  warn={!_paramsOk(paramsExt, 'future-savings')}
                  padding={sizing.fixed.cardPadding}
                  empty={params.futureSavings.length === 0}
                />
              )}
              <PlanSummaryButton
                type="income-during-retirement"
                section={section}
                warn={!_paramsOk(paramsExt, 'income-during-retirement')}
                padding={sizing.fixed.cardPadding}
                empty={params.retirementIncome.length === 0}
              />
            </div>
          </div>

          <div className="">
            <h2
              className="text-[20px] sm:text-[26px] font-bold mb-6"
              style={{ ...paddingCSSStyleHorz(sizing.fixed.cardPadding) }}
            >
              Spending Goals
            </h2>
            <div className="flex flex-col gap-y-6 ">
              <PlanSummaryButton
                type="extra-spending"
                section={section}
                warn={!_paramsOk(paramsExt, 'extra-spending')}
                padding={sizing.fixed.cardPadding}
                empty={
                  params.extraSpending.discretionary.length === 0 &&
                  params.extraSpending.essential.length === 0
                }
              />

              {params.strategy !== 'SWR' && (
                <PlanSummaryButton
                  type="legacy"
                  section={section}
                  padding={sizing.fixed.cardPadding}
                  empty={
                    params.legacy.tpawAndSPAW.external.length === 0 &&
                    params.legacy.tpawAndSPAW.total === 0
                  }
                />
              )}
            </div>
          </div>

          <div className="">
            <div
              className=""
              style={{ ...paddingCSSStyleHorz(sizing.fixed.cardPadding) }}
            >
              <h2 className="text-[20px] sm:text-[26px] font-bold  flex justify-between items-center">
                Risk
                <PlanSummaryRiskSwitchMode className="" />
              </h2>
              {!params.risk.useTPAWPreset && (
                <PlanSummaryRiskCopyFromAPreset className="pt-2 pb-4" />
              )}
            </div>
            <div className="flex flex-col gap-y-6 ">
              {params.risk.useTPAWPreset ? (
                <PlanSummaryButton
                  type="risk-level"
                  section={section}
                  padding={sizing.fixed.cardPadding}
                />
              ) : (
                <>
                  <PlanSummaryButton
                    type="stock-allocation"
                    section={section}
                    padding={sizing.fixed.cardPadding}
                  />
                  {params.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="spending-tilt"
                      section={section}
                      padding={sizing.fixed.cardPadding}
                    />
                  )}
                  {params.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="spending-ceiling-and-floor"
                      section={section}
                      padding={sizing.fixed.cardPadding}
                      empty={
                        params.risk.tpawAndSPAW.spendingCeiling === null &&
                        params.risk.tpawAndSPAW.spendingFloor === null
                      }
                    />
                  )}
                  {!Config.client.production && params.strategy !== 'SWR' && (
                    <PlanSummaryButton
                      type="lmp"
                      section={section}
                      padding={sizing.fixed.cardPadding}
                    />
                  )}
                  {params.strategy === 'SWR' && (
                    <PlanSummaryButton
                      type="withdrawal"
                      section={section}
                      padding={sizing.fixed.cardPadding}
                    />
                  )}
                </>
              )}
            </div>
          </div>

          <div className="">
            <button
              className=""
              style={{ ...paddingCSSStyleHorz(sizing.fixed.cardPadding) }}
              onClick={() => setShowAdvanced(!showAdvanced)}
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
              <div className="flex flex-col gap-y-6 mt-6">
                <PlanSummaryButton
                  type="expected-returns"
                  section={section}
                  padding={sizing.fixed.cardPadding}
                  flagAsModified={_isModified('expected-returns', params)}
                />
                <PlanSummaryButton
                  type="inflation"
                  section={section}
                  padding={sizing.fixed.cardPadding}
                  flagAsModified={_isModified('inflation', params)}
                />
                <PlanSummaryButton
                  type="simulation"
                  section={section}
                  padding={sizing.fixed.cardPadding}
                  flagAsModified={_isModified('simulation', params)}
                />
                <PlanSummaryButton
                  type="strategy"
                  section={section}
                  padding={sizing.fixed.cardPadding}
                  flagAsModified={_isModified('strategy', params)}
                />
                {!Config.client.production && (
                  <PlanSummaryButton
                    type="dev"
                    section={section}
                    padding={sizing.fixed.cardPadding}
                  />
                )}
              </div>
            )}
          </div>
        </div>
      </NoDisplayOnOpacity0Transition>
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
