import { Document } from '@contentful/rich-text-types'
import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { CalendarMonthFns, block } from '@tpaw/common'
import React, { useMemo, useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { formatPercentage } from '../../../Utils/FormatPercentage'
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
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../PlanRootHelpers/WithSimulation'
import { PlanInputBodyHeader } from './PlanInput/PlanInputBody/PlanInputBodyHeader'
import {
  PlanTransitionState,
  simplifyPlanTransitionState2,
} from './PlanTransition'

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
  const { planParamsNormInstant } = useSimulationInfo()
  const { planParamsProcessed, numOfSimulationForMonteCarloSamplingOfResult } =
    useSimulationResultInfo().simulationResult

  const contentBeforeVars =
    usePlanContent().help[planParamsNormInstant.advanced.strategy]
  const contentAfterVars = block(() => {
    const variables = {
      numOfSimulationForMonteCarloSampling:
        numOfSimulationForMonteCarloSamplingOfResult.toString(),
      historicalExpectedStockReturn: formatPercentage(1)(
        planParamsProcessed.marketDataForPresets.expectedReturns.stocks
          .historical,
      ),
      historicalExpectedBondReturn: formatPercentage(1)(
        planParamsProcessed.marketDataForPresets.expectedReturns.bonds
          .historical,
      ),
      historicalReturnDataStartMonth: CalendarMonthFns.toStr(
        planParamsProcessed.historicalReturns.monthRange.start,
      ),
      historicalReturnDataEndMonth: CalendarMonthFns.toStr(
        planParamsProcessed.historicalReturns.monthRange.end,
      ),
      tipsYield20Year: formatPercentage(1)(
        planParamsProcessed.marketDataForPresets.expectedReturns.bonds
          .tipsYield20Year,
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
