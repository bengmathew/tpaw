import { Document } from '@contentful/rich-text-types'
import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, { useMemo, useState } from 'react'
import { Contentful } from '../../Utils/Contentful'
import {
  newPaddingHorz,
  originCSSStyle,
  paddingCSSStyleHorz,
  Size,
  sizeCSSStyle,
  XY,
} from '../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../Utils/NoDisplayOnOpacity0Transition'
import { fGet } from '../../Utils/Utils'
import { useSimulation } from '../App/WithSimulation'
import { usePlanContent } from './Plan'
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
  { label: 'in', sections: [{ name: 'help', dialogMode: 'any' }] },
  { label: 'out', sections: [{ name: 'rest', dialogMode: 'any' }] },
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
          paddingTop: `${sizing.fixed.padding.top}px`,
        }}
      >
        {/* Moved to separate component to force recreation after every transition
         reset. */}
        <_Body />
      </NoDisplayOnOpacity0Transition>
    )
  },
)

const _Body = React.memo(() => {
  const { params } = useSimulation()

  const content = Contentful.splitDocument(
    usePlanContent().help[params.strategy],
    'faq',
  )

  return (
    <div className="pb-20">
      <PlanInputBodyHeader type="help" />
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
          className={`font-semibold  text-start ${show ? 'mb-1' : 'mb-4'}`}
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