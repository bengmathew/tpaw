import clix from 'clsx'
import { Power1, Power4 } from 'gsap'
import React, { useCallback, useMemo, useState } from 'react'

import getIsMobile from 'is-mobile'
import {
  Padding,
  RectExt,
  insetExt,
  originCSSStyle,
  rectExt,
  regionCSSStyle,
} from '../../../../Utils/Geometry'
import { NoDisplayOnOpacity0Transition } from '../../../../Utils/NoDisplayOnOpacity0Transition'
import { Record } from '../../../../Utils/Record'
import { ChartAnimation } from '../../../Common/Chart/Chart'
import { ChartPointerPortal } from '../../../Common/Chart/ChartComponent/ChartPointerPortal'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import {
  PlanTransitionState,
  simplifyPlanTransitionState4,
} from '../PlanTransition'
import { usePlanColors } from '../UsePlanColors'
import { useChartData } from '../WithPlanResultsChartData'
import { PlanResultsChartCard } from './PlanResultsChartCard/PlanResultsChartCard'
import { PlanResultsChartRescale } from './PlanResultsChartRescale'
import { PlanResultsDialogCurtain } from './PlanResultsDialogCurtain'
import { PlanResultsHelp } from './PlanResultsHelp'
import { PlanResultsSidePanel } from './PlanResultsSidePanel/PlanResultsSidePanel'
import {
  planResultsLegacyCardFormat,
  usePlanResultsLegacyCardData,
} from './PlanResultsSidePanel/PlanResultsSidePanelLegacyCard'
import { usePlanResultsChartType } from './UsePlanResultsChartType'

export const planResultsMorphAnimation: ChartAnimation = {
  ease: Power4.easeOut,
  duration: 1.5,
}
export const planResultsNormalAnimation: ChartAnimation = {
  ease: Power1.easeOut,
  duration: 0.5,
}

export type PlanResultsStateful = { transitionUpdated: () => void }

export type PlanResultsTransitionState = ReturnType<
  typeof _toPlanResultsTransitionState
>

const _toPlanResultsTransitionState = simplifyPlanTransitionState4(
  {
    label: 'dialogSummary',
    sections: [{ section: 'summary', dialogMode: true }],
  },
  {
    label: 'dialogInput',
    sections: [{ section: 'rest', dialogMode: true }],
  },
  {
    label: 'notDialogSummary',
    sections: [{ section: 'summary', dialogMode: false }],
  },
  {
    label: 'notDialogInput',
    sections: [{ section: 'rest', dialogMode: false }],
  },
)

export type PlanResultsSizing = {
  dynamic: Record<
    PlanResultsTransitionState,
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
  sizing: PlanResultsSizing
  planTransition: { target: PlanTransitionState; duration: number }
  section: PlanSectionName
  chartPointerPortal: ChartPointerPortal
  onChartHover: (hover: boolean) => void
  chartHover: boolean
}
export const PlanResults = React.forwardRef<HTMLDivElement, _Props>(
  (
    {
      layout,
      sizing: sizingIn,
      planTransition,
      section,
      chartPointerPortal,
      onChartHover,
      chartHover,
    }: _Props,
    ref,
  ) => {
    const chartType = usePlanResultsChartType()

    const chartData = useChartData(chartType)

    const [mainYRange, setMainYRange] = useState(chartData.displayRange.y)

    const [measures, setMeasures] = useState({
      help: { height: 0 },
    })
    const handleHelpHeight = useCallback((height: number) => {
      setMeasures((x) => ({ help: { height } }))
    }, [])

    const legacyData = usePlanResultsLegacyCardData()
    const maxLegacy = Math.max(...legacyData.map((x) => x.amount))

    const sizing = useMemo(
      () => _transformSizing(sizingIn, measures, maxLegacy, layout),
      [sizingIn, maxLegacy, layout, measures],
    )
    const transition = useMemo(
      () => ({
        target: _toPlanResultsTransitionState(planTransition.target),
        duration: planTransition.duration,
      }),
      [planTransition],
    )

    const targetSizing = useMemo(
      () => sizing.dynamic[transition.target],
      [sizing, transition.target],
    )

    const transitionDuration = `${planTransition.duration}ms`
    const planColors = usePlanColors()
    return (
      <NoDisplayOnOpacity0Transition
        ref={ref}
        noDisplayMeans="visibility:hidden"
        className={clix('absolute z-50  overflow-hidden')}
        style={{
          transitionProperty: 'borderRadius, opacity, width, height, top, left',
          transitionDuration,
          borderRadius: `${targetSizing.borderRadius}px`,
          opacity: `${targetSizing.opacity}`,
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          ...regionCSSStyle(targetSizing.region),
          backgroundColor: planColors.results.bg,

          color: planColors.results.fg,
        }}
      >
        <PlanResultsDialogCurtain layout={layout} />

        <PlanResultsSidePanel
          layout={layout}
          targetDynamicSizing={targetSizing.sidePanel}
          duration={planTransition.duration}
          fixedSizing={sizing.fixed}
          section={section}
        />

        <PlanResultsHelp
          onHeight_const={handleHelpHeight}
          layout={layout}
          targetDynamicSizing={targetSizing.help}
          duration={planTransition.duration}
          section={section}
        />
        <div
          className={'absolute inset-0 pointer-events-none bg-black/60'}
          style={{
            transitionProperty: 'opacity',
            transitionDuration: '300ms',
            opacity: `${getIsMobile() && chartHover ? 1 : 0}`,
          }}
        />
        <PlanResultsChartCard
          layout={layout}
          yRange={mainYRange}
          setYRange={setMainYRange}
          sizing={sizing}
          transition={transition}
          chartPointerPortal={chartPointerPortal}
          onChartHover={onChartHover}
          chartHover={chartHover}
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
          <PlanResultsChartRescale
            section={section}
            chartType={chartType}
            mainYRange={mainYRange}
            setMainYRange={setMainYRange}
          />
        </div>
      </NoDisplayOnOpacity0Transition>
    )
  },
)

const _transformSizing = (
  sizingIn: PlanResultsSizing,
  measures: { help: { height: number } },
  maxLegacy: number,
  layout: 'laptop' | 'desktop' | 'mobile',
) => {
  const { cardPadding, intraGap } = sizingIn.fixed
  const _map = (
    state: PlanResultsTransitionState,
    orig: PlanResultsSizing['dynamic']['dialogInput'],
  ) => {
    const { padding, region } = orig
    const legacyWidth = Math.max(
      cardPadding.left +
        cardPadding.right +
        (planResultsLegacyCardFormat(maxLegacy, layout).length + 2) * 7 +
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
          height: region.height - padding.top,
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
