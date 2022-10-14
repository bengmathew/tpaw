import {faClipboardList} from '@fortawesome/pro-regular-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {Power1, Power4} from 'gsap'
import Link from 'next/link'
import React, {useLayoutEffect, useMemo, useRef, useState} from 'react'
import {
  insetExt,
  originCSSStyle,
  Padding,
  RectExt,
  rectExt,
  regionCSSStyle,
  Size,
} from '../../../Utils/Geometry'
import {NoDisplayOnOpacity0Transition} from '../../../Utils/NoDisplayOnOpacity0Transition'
import {Record} from '../../../Utils/Record'
import {fGet} from '../../../Utils/Utils'
import {useChartData} from '../../App/WithChartData'
import {SimulationInfo, useSimulation} from '../../App/WithSimulation'
import {ChartAnimation} from '../../Common/Chart/Chart'
import {Config} from '../../Config'
import {PlanSectionName} from '../PlanInput/Helpers/PlanSectionName'
import {
  PlanTransitionState,
  simplifyPlanTransitionState4,
} from '../PlanTransition'
import {PlanChartLegacyCard} from './PlanChartLegacyCard'
import {PlanChartMainCard} from './PlanChartMainCard/PlanChartMainCard'
import {PlanChartRescale} from './PlanChartRescale'
import {PlanChartType} from './PlanChartType'
import {usePlanChartType} from './UsePlanChartType'

export const planChartMorphAnimation: ChartAnimation = {
  ease: Power4.easeOut,
  duration: 1.5,
}
export const planChartNormalAnimation: ChartAnimation = {
  ease: Power1.easeOut,
  duration: 0.5,
}

export type PlanChartStateful = {transitionUpdated: () => void}

type PlanChartTransitionState = ReturnType<typeof _toPlanChartTransitionState>
export type PlanChartInternalTransitionState = `${PlanChartTransitionState}${
  | 'With'
  | 'Without'}Legacy`

const _toPlanChartTransitionState = simplifyPlanTransitionState4(
  {label: 'results', sections: [{name: 'results', dialogMode: 'any'}]},
  {label: 'hidden', sections: [{name: 'rest', dialogMode: true}]},
  {label: 'summary', sections: [{name: 'summary', dialogMode: false}]},
  {label: 'input', sections: [{name: 'rest', dialogMode: false}]}
)

export type PlanChartSizing = {
  dynamic: Record<
    PlanChartTransitionState,
    {
      region: RectExt
      padding: Padding
      legacyWidth: number
      intraGap: number
      borderRadius: number
      tasksOpacity: number
      opacity: number
    }
  >
  fixed: {
    cardPadding: Padding
  }
}
export const PlanChart = React.memo(
  ({
    layout,
    sizing: sizingIn,
    planTransition,
    section,
  }: {
    layout: 'laptop' | 'desktop' | 'mobile'
    sizing: PlanChartSizing
    planTransition: {target: PlanTransitionState; duration: number}
    section: PlanSectionName
  }) => {
    const {paramSpace, setParamSpace, tpawResult} = useSimulation()
    const type = usePlanChartType()
    const shouldShowLegacy = _shouldShowLegacy(tpawResult, type)
    const tasksRef = useRef<HTMLAnchorElement>(null)

    const allChartData = useChartData()
    const chartMainData =
      type === 'reward-risk-ratio-comparison'
        ? fGet(allChartData.rewardRiskRatio)
        : fGet(allChartData.byYearsFromNowPercentiles.get(type))
    const chartLegacyData = allChartData.legacy

    const [mainYRange, setMainYRange] = useState(chartMainData.yDisplayRange)
    const [legacyYRange, setLegacyYRange] = useState(
      chartLegacyData.xyDisplayRange.y
    )

    const [measures, setMeasures] = useState({tasks: {height: 0, width: 0}})
    useLayoutEffect(() => {
      const observer = new ResizeObserver(() => {
        const tasks = fGet(tasksRef.current).getBoundingClientRect()
        setMeasures({
          tasks: {width: tasks.width, height: tasks.height},
        })
      })
      observer.observe(fGet(tasksRef.current))
      return () => observer.disconnect()
    }, [])

    const sizing = useMemo(
      () => _transformSizing(sizingIn, measures),
      [measures, sizingIn]
    )
    const transition = useMemo(
      () => ({
        target: `${_toPlanChartTransitionState(planTransition.target)}${
          shouldShowLegacy ? 'With' : 'Without'
        }Legacy` as const,
        duration: planTransition.duration,
      }),
      [planTransition, shouldShowLegacy]
    )
    const targetSizing = useMemo(
      () => sizing.dynamic[transition.target],
      [sizing, transition.target]
    )

    const transitionDuration = `${planTransition.duration}ms`
    return (
      <NoDisplayOnOpacity0Transition
        noDisplayMeans="visibility:hidden"
        className={`absolute z-10 bg-chartBG overflow-hidden
          ${layout == 'laptop' ? '' : ''}`}
        style={{
          transitionProperty: 'borderRadius, opacity, width, height, top, left',
          transitionDuration,
          borderRadius: `${targetSizing.borderRadius}px`,
          opacity: `${targetSizing.opacity}`,
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          ...regionCSSStyle(targetSizing.region),
        }}
      >
        {layout !== 'mobile' && (
          <h2
            className={`absolute uppercase font-bold text-white `}
            style={{
              transitionProperty: 'transform',
              transitionDuration,
              transform: `translate(${targetSizing.heading.origin.x}px,${targetSizing.heading.origin.y}px)`,
              ...originCSSStyle({x: 0, y: 0}),
            }}
            onClick={
              Config.client.production
                ? undefined
                : () => setParamSpace(paramSpace === 'a' ? 'b' : 'a')
            }
          >
            Results {paramSpace === 'a' ? '' : '(B)'}
          </h2>
        )}

        <PlanChartMainCard
          layout={layout}
          yRange={mainYRange}
          setYRange={setMainYRange}
          sizing={sizing}
          transition={transition}
        />
        <PlanChartLegacyCard
          yRange={legacyYRange}
          sizing={sizing}
          transition={transition}
        />
        <div
          className="absolute"
          style={{
            transitionProperty: 'transform',
            transitionDuration,
            transform: `translate(${targetSizing.rescale.origin.x}px,${targetSizing.rescale.origin.y}px)`,
            ...originCSSStyle({x: 0, y: 0}),
            height: `${targetSizing.rescale.height}px`,
          }}
        >
          <PlanChartRescale
            section={section}
            type={type}
            shouldShowLegacy={shouldShowLegacy}
            mainYRange={mainYRange}
            setMainYRange={setMainYRange}
            legacyYRange={legacyYRange}
            setLegacyYRange={setLegacyYRange}
          />
        </div>

        <Link href="/tasks-for-this-year">
          <a
            ref={tasksRef}
            className="absolute whitespace-nowrap bg-cardB py-1 rounded-xl flex items-center gap-x-2 text-white"
            style={{
              transitionProperty: 'opacity, transform',
              transitionDuration,
              transform: `translate(${targetSizing.tasks.x}px,${targetSizing.tasks.y}px)`,
              ...originCSSStyle({x: 0, y: 0}),
              opacity: `${targetSizing.tasks.opacity}`,
              pointerEvents: targetSizing.tasks.opacity === 0 ? 'none' : 'auto',
            }}
          >
            <h2 className="font-bold text-base sm:text-xl flex items-center gap-x-2">
              <FontAwesomeIcon icon={faClipboardList} />
              Tasks for this year
            </h2>
          </a>
        </Link>
      </NoDisplayOnOpacity0Transition>
    )
  }
)

const _transformSizing = (
  sizingIn: PlanChartSizing,
  measures: {tasks: Size}
) => {
  const {cardPadding} = sizingIn.fixed
  const _map = (
    state: PlanChartInternalTransitionState,
    orig: PlanChartSizing['dynamic']['hidden'],
    shouldShowLegacy: boolean
  ) => {
    const {padding, region, legacyWidth, intraGap, tasksOpacity} = orig
    const mainCard = (() => {
      const widthWithoutLegacy = region.width - padding.left - padding.right
      const mainRegion = rectExt({
        x: padding.left,
        y: padding.top,
        width: shouldShowLegacy
          ? widthWithoutLegacy - legacyWidth - intraGap
          : widthWithoutLegacy,
        height: region.height - padding.top - padding.bottom,
      })
      return {
        region: mainRegion,
        inset: insetExt(mainRegion, region),
      }
    })()
    const legacyCard = {
      inset: insetExt(
        rectExt({
          x: shouldShowLegacy
            ? mainCard.region.right + intraGap
            : region.width + 1,
          y: padding.top,
          width: legacyWidth,
          height: region.height - padding.top - padding.bottom,
        }),
        region
      ),
      visibility: shouldShowLegacy ? ('visible' as const) : ('hidden' as const),
    }

    const heading = {
      origin: {x: padding.left + cardPadding.left, y: mainCard.region.y - 30},
    }
    const rescale = (() => {
      const height = 25
      const origin = {
        x: padding.left,
        y:
          mainCard.region.bottom +
          Math.max(0, (region.height - mainCard.region.bottom - height) / 2),
      }
      return {height, origin}
    })()
    const tasks = {
      y:
        mainCard.region.bottom +
        Math.max(
          0,
          (region.height - mainCard.region.bottom - measures.tasks.height) / 2
        ),
      x: region.width - padding.right - measures.tasks.width,
      opacity: tasksOpacity,
    }

    return [
      state,
      {...orig, mainCard, legacyCard, heading, tasks, rescale},
    ] as const
  }

  return {
    dynamic: Record.merge(
      Record.map(sizingIn.dynamic, (state, orig) =>
        _map(`${state}WithLegacy`, orig, true)
      ),
      Record.map(sizingIn.dynamic, (state, orig) =>
        _map(`${state}WithoutLegacy`, orig, false)
      )
    ),
    fixed: {
      cardPadding: sizingIn.fixed.cardPadding,
    },
  }
}

const _shouldShowLegacy = (
  _: SimulationInfo['tpawResult'],
  __: PlanChartType
) => true // Always showing legacy now.
