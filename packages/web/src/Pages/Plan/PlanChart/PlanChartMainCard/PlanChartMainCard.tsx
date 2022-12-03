import React, { useEffect, useMemo, useRef, useState } from 'react'
import { cssEaseAsGSAPEase } from '../../../../Utils/CSSEaseAsGSAPEase'
import {
  insetCSSStyle,
  InsetExt,
  Padding,
  rectExt,
  regionCSSStyle,
} from '../../../../Utils/Geometry'
import { Record } from '../../../../Utils/Record'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { useAssertConst } from '../../../../Utils/UseAssertConst'
import { fGet } from '../../../../Utils/Utils'
import { useChartData } from '../../../App/WithChartData'
import { useSimulation } from '../../../App/WithSimulation'
import { ChartReactStatefull } from '../../../Common/Chart/ChartReact'
import {
  planChartMorphAnimation,
  planChartNormalAnimation,
  PlanChartTransitionState,
} from '../PlanChart'
import {
  TPAWChartDataMain,
  tpawChartDataScaled,
} from '../TPAWChart/TPAWChartDataMain'
import {
  TPAWChartMain,
  tpawChartMainXAxisSizing,
} from '../TPAWChart/TPAWChartMain'
import { usePlanChartType } from '../UsePlanChartType'
import { HeightsForWidths } from './HeightsForWidths'
import { PlanChartMainCardMenu } from './PlanChartMainCardMenu'
import { PlanChartMainCardMenuButton } from './PlanChartMainCardMenuButton'

export type PlanChartMainCardSizing = {
  dynamic: Record<PlanChartTransitionState, { mainCard: { inset: InsetExt } }>
  fixed: {
    cardPadding: Padding
  }
}
type _TransitionState = `${PlanChartTransitionState}${
  | 'With'
  | 'Without'}SuccessRate`

export const PlanChartMainCard = React.memo(
  ({
    layout,
    yRange,
    setYRange,
    sizing: sizingIn,
    transition,
  }: {
    layout: 'laptop' | 'desktop' | 'mobile'
    yRange: SimpleRange
    setYRange: (y: SimpleRange) => void
    sizing: PlanChartMainCardSizing
    transition: { target: PlanChartTransitionState; duration: number }
  }) => {
    const { params, tpawResult } = useSimulation()
    const type = usePlanChartType()

    const chartRef = useRef<ChartReactStatefull<TPAWChartDataMain> | null>(null)
    const [heights, setHeights] = useState<{
      menu: Record<_TransitionState, number> | null
    }>({ menu: null })
    const [handleMenuHeights] = useState(
      () => (heights: Record<_TransitionState, number>) =>
        setHeights((x) => ({ ...x, menu: heights })),
    )

    const showSuccessRate = params.strategy === 'SWR'
    const hasPartner = tpawResult.args.params.people.withPartner

    const sizing = useMemo(() => {
      const _map = (
        state: _TransitionState,
        dynamic: PlanChartMainCardSizing['dynamic']['hidden'],
        showSuccessRate: boolean,
      ) =>
        [
          state,
          _transformSizing(
            dynamic,
            sizingIn.fixed,
            {
              menu: heights.menu ? heights.menu[state] : 0,
            },
            hasPartner,
            showSuccessRate,
          ),
        ] as const

      return Record.merge(
        Record.map(sizingIn.dynamic, (state, dynamic) =>
          _map(`${state}WithSuccessRate`, dynamic, true),
        ),
        Record.map(sizingIn.dynamic, (state, dynamic) =>
          _map(`${state}WithoutSuccessRate`, dynamic, false),
        ),
      )
    }, [heights, sizingIn, hasPartner])
    const targetSizing = useMemo(
      () =>
        sizing[
          `${transition.target}${
            showSuccessRate ? 'With' : 'Without'
          }SuccessRate`
        ],
      [showSuccessRate, sizing, transition.target],
    )

    const allChartData = useChartData()
    const chartMainData =
      type === 'reward-risk-ratio-comparison'
        ? fGet(allChartData.rewardRiskRatio)
        : fGet(allChartData.byYearsFromNowPercentiles.get(type))

    useEffect(() => {
      const chart = chartRef.current
      if (!chart) return
      const data = chartMainData
      const prevState = chart.getState()
      if (type === prevState.data.type) {
        chart.setState(
          data,
          { x: data.years.displayRange, y: prevState.xyRange.y },
          planChartNormalAnimation,
        )
      } else {
        chart.setState(
          tpawChartDataScaled(prevState.data, data.yDisplayRange),
          { x: prevState.xyRange.x, y: data.yDisplayRange },
          null,
        )
        chart.setState(
          data,
          { x: data.years.displayRange, y: data.yDisplayRange },
          planChartMorphAnimation,
        )
      }
      setYRange(chart.getState().xyRange.y)
    }, [chartMainData, setYRange, type])
    useAssertConst([setYRange])

    useEffect(() => {
      const chart = chartRef.current
      if (!chart) return
      const prevState = chart.getState()
      chart.setState(
        prevState.data,
        { x: prevState.xyRange.x, y: yRange },
        planChartNormalAnimation,
      )
    }, [yRange])

    useEffect(() => {
      chartRef.current?.setSizing(targetSizing.chart, {
        duration: transition.duration,
        ease: cssEaseAsGSAPEase,
      })
    }, [targetSizing, transition.duration])

    const showHelp = transition.target === 'summary'
    return (
      <div
        className="absolute bg-cardBG   rounded-t-2xl rounded-b-2xl overflow-hidden "
        style={{
          transitionProperty: 'top, left, right, bottom',
          transitionDuration: `${transition.duration}ms`,
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          ...insetCSSStyle(targetSizing.inset),
        }}
      >
        <HeightsForWidths sizing={sizing} onHeights={handleMenuHeights}>
          {(sizing) => (
            <div style={{ width: `${sizing.menu.region.width}px` }}>
              <PlanChartMainCardMenuButton
                onClick={() => {}}
                transition={transition}
              />
            </div>
          )}
        </HeightsForWidths>
        <div
          className="absolute"
          style={{ ...regionCSSStyle(targetSizing.menu.region) }}
        >
          <PlanChartMainCardMenu transition={transition} />
        </div>

        <TPAWChartMain
          starting={{
            data: chartMainData,
            xyRange: {
              x: chartMainData.years.displayRange,
              y: chartMainData.yDisplayRange,
            },
            sizing: targetSizing.chart,
          }}
          ref={chartRef}
        />
      </div>
    )
  },
)

const _transformSizing = (
  { mainCard }: PlanChartMainCardSizing['dynamic']['hidden'],
  { cardPadding }: PlanChartMainCardSizing['fixed'],
  heights: { menu: number },
  hasPartner: boolean,
  showSuccessRate: boolean,
) => {
  // Scaling does not affect caluclated height, so scale it here.
  const { inset } = mainCard
  const menuHeight = heights.menu
  const menu = {
    region: rectExt({
      x: cardPadding.left,
      y: cardPadding.top,
      width: inset.width - cardPadding.right - cardPadding.left,
      height: menuHeight,
    }),
  }

  const chart = (() => {
    const position = rectExt({
      x: 0,
      y: menu.region.bottom,
      width: inset.width,
      bottom: inset.height,
    })
    const xAxisSizing = tpawChartMainXAxisSizing(position.width)
    return {
      position,
      padding: {
        left: cardPadding.left,
        right: cardPadding.right,
        top: 20, // 20 is needed for the max y label.
        bottom:
          4 +
          xAxisSizing.height +
          (hasPartner
            ? +3 * xAxisSizing.gap + xAxisSizing.height
            : 2 * xAxisSizing.gap),
      },
    }
  })()

  return { menu, chart, inset }
}
