import { Document } from '@contentful/rich-text-types'
import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, { useMemo, useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import {
  Size,
  XY,
  newPaddingHorz,
  originCSSStyle,
  paddingCSSStyleHorz,
  sizeCSSStyle,
} from '../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../Utils/NoDisplayOnOpacity0Transition'
import { fGet } from '../../../Utils/Utils'
import { usePlanContent } from '../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../PlanRootHelpers/WithSimulation'
import { PlanInputBodyHeader } from './PlanInput/PlanInputBody/PlanInputBodyHeader'
import {
  PlanTransitionState,
  simplifyPlanTransitionState2,
} from './PlanTransition'
import { CalendarMonthFns, block } from '@tpaw/common'
import { fWASM } from '../../../UseSimulator/Simulator/GetWASM'
import { formatPercentage } from '../../../Utils/FormatPercentage'
import { CallRust } from '../../../UseSimulator/PlanParamsProcessed/CallRust'

export type PlanHelpSizing = {
  dynamic: Record<_PlanHelpTransitionState, { origin: XY; opacity: number }>
  fixed: {
    size: Size
    padding: ({ left: number; right: number } | { horz: number }) & {
      top: number
    }
  }
}

const _toPlanHelpTransitionState = simplifyPlanTransitionState2(
  { label: 'in', sections: [{ section: 'help', dialogMode: 'any' }] },
  { label: 'out', sections: [{ section: 'rest', dialogMode: 'any' }] },
)
type _PlanHelpTransitionState = ReturnType<typeof _toPlanHelpTransitionState>

export const PlanHelp = React.memo(
  ({
    sizing,
    planTransition,
  }: {
    sizing: PlanHelpSizing
    planTransition: { target: PlanTransitionState; duration: number }
  }) => {
    const targetSizing = useMemo(
      () => sizing.dynamic[_toPlanHelpTransitionState(planTransition.target)],
      [planTransition.target, sizing],
    )
    return (
      <NoDisplayOnOpacity0Transition
        className="absolute overflow-scroll "
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
        {/* Moved to separate component to force recreation after every transition
         reset. */}
        <_Body sizing={sizing} />
      </NoDisplayOnOpacity0Transition>
    )
  },
)

const _Body = React.memo(({ sizing }: { sizing: PlanHelpSizing }) => {
  const { planParamsNorm, currentMarketData } = useSimulation()
  const contentBeforeVars =
    usePlanContent().help[planParamsNorm.advanced.strategy]
  const contentAfterVars = block(() => {
    const presetInfo =
      fWASM().process_market_data_for_expected_returns_for_planning_presets(
        CallRust.getPlanParamsRust(planParamsNorm).advanced.sampling,
        planParamsNorm.advanced.historicalMonthlyLogReturnsAdjustment
          .standardDeviation,
        currentMarketData,
      )
    const variables = {
      historicalExpectedStockReturn: formatPercentage(1)(
        presetInfo.stocks.historical,
      ),
      historicalExpectedBondReturn: formatPercentage(1)(
        presetInfo.bonds.historical,
      ),
      historicalReturnDataStartMonth: CalendarMonthFns.toStr(
        presetInfo.historicalReturnsMonthRange.start,
      ),
      historicalReturnDataEndMonth: CalendarMonthFns.toStr(
        presetInfo.historicalReturnsMonthRange.end,
      ),
      tipsYield20Year: formatPercentage(1)(
        currentMarketData.bondRates.twentyYear,
      ),
    }
    return Contentful.replaceVariables(variables, contentBeforeVars)
  })

  const content = Contentful.splitDocument(contentAfterVars, 'faq')

  return (
    <div
      className="mb-20"
      style={{
        paddingTop: `${sizing.fixed.padding.top}px`,
      }}
    >
      <PlanInputBodyHeader className="" type="help" />
      <_RichText className="mt-6" body={fGet(content.intro)} />
      <div className="mt-6">
        {content.sections.map((section, i) => (
          <_Collapsable key={i} className="" section={section} />
        ))}
      </div>
    </div>
  )
})

const _Collapsable = React.memo(
  ({
    className = '',
    section: { body, heading },
  }: {
    className?: string
    section: { heading: string; body: Document }
  }) => {
    const [show, setShow] = useState(false)
    return (
      <div className={`${className}`}>
        <button
          className={`font-semibold  text-start ${show ? 'mb-3' : 'mb-4'}`}
          onClick={() => setShow(!show)}
        >
          {heading}
          <FontAwesomeIcon
            className="ml-2 -mb-[1px]"
            icon={show ? faCaretDown : faCaretRight}
          />
        </button>
        {show && <_RichText className="mb-6" body={body} />}
      </div>
    )
  },
)

const _RichText = React.memo(
  ({ className = '', body }: { className?: string; body: Document }) => {
    return (
      <div className={`${className}`}>
        <Contentful.RichText
          body={body}
          ul="list-disc ml-5"
          ol="list-decimal ml-5"
          p="p-base mb-3"
          h2={([index]) => `font-bold text-2xl mb-3 mt-6`}
          h3={`font-bold text-lg mb-2 mt-4`}
        />
      </div>
    )
  },
)
