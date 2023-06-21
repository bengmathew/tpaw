import { Power1, Power4 } from 'gsap'
import React, { useCallback, useMemo, useState } from 'react'
import {
  insetExt,
  originCSSStyle,
  Padding,
  RectExt,
  rectExt,
  regionCSSStyle,
} from '../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../Utils/NoDisplayOnOpacity0Transition'
import { Record } from '../../../Utils/Record'
import { fGet } from '../../../Utils/Utils'
import { useChartData } from '../../App/WithChartData'
import { ChartAnimation } from '../../Common/Chart/Chart'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import {
  PlanTransitionState,
  simplifyPlanTransitionState4,
} from '../PlanTransition'
import { PlanChartDialogCurtain } from './PlanChartDialogCurtain'
import { PlanChartHelp } from './PlanChartHelp'
import { PlanChartMainCard } from './PlanChartMainCard/PlanChartMainCard'
import { PlanChartRescale } from './PlanChartRescale'
import { PlanChartSidePanel } from './PlanChartSidePanel/PlanChartSidePanel'
import {
  planChartLegacyCardFormat,
  usePlanChartLegacyCardData,
} from './PlanChartSidePanel/PlanChartSidePanelLegacyCard'
import { usePlanChartType } from './UsePlanChartType'

export const planChartMorphAnimation: ChartAnimation = {
  ease: Power4.easeOut,
  duration: 1.5,
}
export const planChartNormalAnimation: ChartAnimation = {
  ease: Power1.easeOut,
  duration: 0.5,
}

export type PlanChartStateful = { transitionUpdated: () => void }

export type PlanChartTransitionState = ReturnType<
  typeof _toPlanChartTransitionState
>

const _toPlanChartTransitionState = simplifyPlanTransitionState4(
  { label: 'dialogSummary', sections: [{ name: 'summary', dialogMode: true }] },
  { label: 'dialogInput', sections: [{ name: 'rest', dialogMode: true }] },
  {
    label: 'notDialogSummary',
    sections: [{ name: 'summary', dialogMode: false }],
  },
  { label: 'notDialogInput', sections: [{ name: 'rest', dialogMode: false }] },
)

export type PlanChartSizing = {
  dynamic: Record<
    PlanChartTransitionState,
    {
      region: RectExt
      padding: Padding
      borderRadius: number
      opacity: number
    }
  >
  fixed: {
    intraGap: number
    cardPadding: Padding
  }
}

type _Props = {
  layout: 'laptop' | 'desktop' | 'mobile'
  sizing: PlanChartSizing
  planTransition: { target: PlanTransitionState; duration: number }
  section: PlanSectionName
}
export const PlanChart = React.forwardRef<HTMLDivElement, _Props>(
  ({ layout, sizing: sizingIn, planTransition, section }: _Props, ref) => {
    const type = usePlanChartType()

    const allChartData = useChartData()
    const chartMainData = fGet(allChartData.byYearsFromNowPercentiles.get(type))

    const [mainYRange, setMainYRange] = useState(chartMainData.yDisplayRange)

    const [measures, setMeasures] = useState({
      help: { height: 0 },
    })
    const handleHelpHeight = useCallback((height: number) => {
      setMeasures((x) => ({ help: { height } }))
    }, [])

    const maxLegacy = Math.max(
      ...usePlanChartLegacyCardData().map((x) => x.amount),
    )

    const sizing = useMemo(
      () => _transformSizing(sizingIn, measures, maxLegacy, layout),
      [sizingIn, maxLegacy, layout, measures],
    )
    const transition = useMemo(
      () => ({
        target: _toPlanChartTransitionState(planTransition.target),
        duration: planTransition.duration,
      }),
      [planTransition],
    )

    const targetSizing = useMemo(
      () => sizing.dynamic[transition.target],
      [sizing, transition.target],
    )

    const transitionDuration = `${planTransition.duration}ms`
    return (
      <NoDisplayOnOpacity0Transition
        ref={ref}
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
        <PlanChartDialogCurtain layout={layout} />
        <PlanChartMainCard
          layout={layout}
          yRange={mainYRange}
          setYRange={setMainYRange}
          sizing={sizing}
          transition={transition}
        />
        <PlanChartSidePanel
          layout={layout}
          targetDynamicSizing={targetSizing.sidePanel}
          duration={planTransition.duration}
          fixedSizing={sizing.fixed}
          section={section}
        />
        <div
          className="absolute"
          style={{
            transitionProperty: 'transform',
            transitionDuration,
            transform: `translate(${targetSizing.rescale.origin.x}px,${targetSizing.rescale.origin.y}px)`,
            ...originCSSStyle({ x: 0, y: 0 }),
            height: `${targetSizing.rescale.height}px`,
          }}
        >
          <PlanChartRescale
            section={section}
            type={type}
            mainYRange={mainYRange}
            setMainYRange={setMainYRange}
          />
        </div>
        <PlanChartHelp
          onHeight_const={handleHelpHeight}
          layout={layout}
          targetDynamicSizing={targetSizing.help}
          duration={planTransition.duration}
          section={section}
        />
      </NoDisplayOnOpacity0Transition>
    )
  },
)

const _transformSizing = (
  sizingIn: PlanChartSizing,
  measures: { help: { height: number } },
  maxLegacy: number,
  layout: 'laptop' | 'desktop' | 'mobile',
) => {
  const { cardPadding, intraGap } = sizingIn.fixed
  const _map = (
    state: PlanChartTransitionState,
    orig: PlanChartSizing['dynamic']['dialogInput'],
  ) => {
    const { padding, region } = orig
    const legacyWidth = Math.max(
      cardPadding.left +
        cardPadding.right +
        (planChartLegacyCardFormat(maxLegacy, layout).length + 2) * 7 +
        30,
      layout === 'mobile' ? 100 : 125,
    )

    const mainCard = (() => {
      const mainRegion = rectExt({
        x: padding.left,
        y: padding.top,
        width:
          region.width - padding.left - padding.right - legacyWidth - intraGap,
        height: region.height - padding.top - padding.bottom,
      })
      return {
        region: mainRegion,
        inset: insetExt(mainRegion, region),
      }
    })()
    const sidePanel = {
      inset: insetExt(
        rectExt({
          x: mainCard.region.right + intraGap,
          y: padding.top,
          width: legacyWidth,
          height: region.height - padding.top ,
        }),
        region,
      ),
    }

    const heading = {
      origin: { x: padding.left + cardPadding.left, y: mainCard.region.y - 30 },
    }
    const rescale = (() => {
      const height = 30
      const origin = {
        x: padding.left + 3,
        y: mainCard.region.bottom - 3 - height,
      }
      return { height, origin }
    })()
    const help = {
      inset: {
        bottom:
          (region.height - mainCard.region.bottom - measures.help.height) / 2,
        left: padding.left,
      },
    }

    return [
      state,
      {
        ...orig,
        mainCard,
        sidePanel,
        heading,
        help,
        rescale,
      },
    ] as const
  }

  return {
    dynamic: Record.map(sizingIn.dynamic, (state, orig) => _map(state, orig)),
    fixed: {
      intraGap: sizingIn.fixed.intraGap,
      cardPadding: sizingIn.fixed.cardPadding,
    },
  }
}
