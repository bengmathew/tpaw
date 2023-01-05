import { Document } from '@contentful/rich-text-types'
import {
  faCaretDown,
  faCaretRight,
  faLongArrowRight,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import Link from 'next/link'
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
import { useGetSectionURL, usePlanContent } from './Plan'
import { PlanInputBodyHeader } from './PlanInput/PlanInputBody/PlanInputBodyHeader'
import {
  PlanTransitionState,
  simplifyPlanTransitionState3,
} from './PlanTransition'

export type PlanResultsSizing = {
  dynamic: Record<_PlanResultTransitionState, { origin: XY; opacity: number }>
  fixed: {
    size: Size
    padding: ({ left: number; right: number } | { horz: number }) & {
      top: number
    }
  }
}

const _toPlanResultsTransitionState = simplifyPlanTransitionState3(
  { label: 'in', sections: [{ name: 'results', dialogMode: 'any' }] },
  { label: 'outDialogMode', sections: [{ name: 'rest', dialogMode: true }] },
  {
    label: 'outNotDialogMode',
    sections: [{ name: 'rest', dialogMode: false }],
  },
)
type _PlanResultTransitionState = ReturnType<
  typeof _toPlanResultsTransitionState
>

export const PlanResults = React.memo(
  ({
    sizing,
    planTransition,
  }: {
    sizing: PlanResultsSizing
    planTransition: { target: PlanTransitionState; duration: number }
  }) => {
    const targetSizing = useMemo(
      () =>
        sizing.dynamic[_toPlanResultsTransitionState(planTransition.target)],
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
  const { params, setParams } = useSimulation()
  const [startingDialogMode] = useState(params.dialogMode)
  const getSectionURL = useGetSectionURL()

  // Use startingDialogMode and not params.dialogMode so content does not
  // flash on exiting to summary.
  const content = Contentful.splitDocument(
    usePlanContent().results[
      startingDialogMode ? 'dialogMode' : 'notDialogMode'
    ][params.strategy],
    'faq',
  )

  // const [showFAQ, setShowFAQ] = useState(!startingDialogMode)
  return (
    <div className="pb-20">
      {startingDialogMode ? (
        <h2 className="font-bold text-xl sm:text-3xl">Preliminary Results</h2>
      ) : (
        <PlanInputBodyHeader type="results" />
      )}
      <_RichText className="mt-6" body={fGet(content.intro)} />
      {/* <button
        className={`font-bold text-xl mb-4 mt-6`}
        disabled={!startingDialogMode}
        onClick={() => setShowFAQ(!showFAQ)}
      > */}
      {/* {startingDialogMode ? (
          <>
            Understanding the Graphs{' '}
            <FontAwesomeIcon className="ml-3 text-lg" icon={faChevronDown} />
          </>
        ) : ( */}
      {/* 'FAQ' */}
      {/* )} */}
      {/* </button> */}
      <div className="mt-6">
        {content.sections.map((section, i) => (
          <_Collapsable key={i} className="" section={section} />
        ))}
      </div>
      {startingDialogMode && (
        <div className="sticky bottom-5 flex justify-end mt-10">
          <Link href={getSectionURL('summary')} shallow>
            <a
              className=" right-0 btn-lg btn-dark mt-2 flex items-center gap-x-3"
              onClick={() => {
                setParams((params) => {
                  if (!params.dialogMode) return params
                  const clone = _.cloneDeep(params)
                  clone.dialogMode = false
                  return clone
                })
              }}
            >
              Continue to Full Planner{' '}
              <FontAwesomeIcon icon={faLongArrowRight} />
            </a>
          </Link>
        </div>
      )}
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
          className={`font-semibold  text-start ${
            show ? 'mb-1' : 'mb-4'
          }`}
          onClick={() => setShow(!show)}
        >
          {heading}
          <FontAwesomeIcon className="ml-2 -mb-[1px]" icon={show ? faCaretDown : faCaretRight} />
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
