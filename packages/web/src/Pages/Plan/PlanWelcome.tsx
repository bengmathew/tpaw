import {faLongArrowAltRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React, {useMemo} from 'react'
import {originCSSStyle, Size, sizeCSSStyle, XY} from '../../Utils/Geometry'
import {NoDisplayOnOpacity0Transition} from '../../Utils/NoDisplayOnOpacity0Transition'
import { useGetSectionURL } from './Plan'
import {
  PlanTransitionState,
  simplifyPlanTransitionState2,
} from './PlanTransition'

export type PlanWelcomeSizing = {
  dynamic: Record<
    PlanWelcomeTransitionState,
    {
      origin: XY
      opacity: number
    }
  >
  fixed: {size: Size}
}

const _toPlanWelcomeTransitionState = simplifyPlanTransitionState2(
  {label: 'in', sections: [{name: 'welcome', dialogMode: true}]},
  {label: 'out', sections: [{name: 'rest', dialogMode: 'any'}]}
)
export type PlanWelcomeTransitionState = ReturnType<
  typeof _toPlanWelcomeTransitionState
>

export const PlanWelcome = React.memo(
  ({
    sizing,
    planTransition,
  }: {
    sizing: PlanWelcomeSizing
    planTransition: {target: PlanTransitionState; duration: number}
  }) => {
    const getSectionURL = useGetSectionURL()
    const targetSizing = useMemo(
      () =>
        sizing.dynamic[_toPlanWelcomeTransitionState(planTransition.target)],
      [planTransition.target, sizing]
    )

    return (
      <NoDisplayOnOpacity0Transition
        className="absolute flex flex-col justify-center "
        style={{
          transitionProperty: 'transform, opacity',
          transitionDuration: `${planTransition.duration}ms`,
          transform: `translate(${targetSizing.origin.x}px,${targetSizing.origin.y}px)`,
          opacity: `${targetSizing.opacity}`,
          ...sizeCSSStyle(sizing.fixed.size),
          ...originCSSStyle({x: 0, y: 0}),
        }}
      >
        <div className="">
          <h2 className="font-bold text-3xl">Hi,</h2>
          <p className="p-base mt-10">{`Let's plan your retirement!`}</p>
          <p className="p-base mt-6">
            We need just{' '}
            <span className="text-2xl font-bold mx-1 whitespace-nowrap">
              4 basic inputs
            </span>{' '}
            from you to get started.
          </p>
          <div className="flex justify-end">
            <Link href={getSectionURL('age')} shallow>
              <a className="btn-dark btn-lg mt-14">
                {`Start`}
                <FontAwesomeIcon className="ml-3" icon={faLongArrowAltRight} />
              </a>
            </Link>
          </div>
        </div>
      </NoDisplayOnOpacity0Transition>
    )
  }
)
