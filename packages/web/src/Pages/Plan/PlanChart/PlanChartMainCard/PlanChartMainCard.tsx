import React, {useEffect, useMemo, useRef, useState} from 'react'
import {cssEaseAsGSAPEase} from '../../../../Utils/CSSEaseAsGSAPEase'
import {
  insetCSSStyle,
  InsetExt,
  Padding,
  rectExt,
  regionCSSStyle,
} from '../../../../Utils/Geometry'
import {Record} from '../../../../Utils/Record'
import {SimpleRange} from '../../../../Utils/SimpleRange'
import {useAssertConst} from '../../../../Utils/UseAssertConst'
import {fGet} from '../../../../Utils/Utils'
import {useChartData} from '../../../App/WithChartData'
import {useSimulation} from '../../../App/WithSimulation'
import {ChartReactStatefull} from '../../../Common/Chart/ChartReact'
import {
  PlanChartInternalTransitionState,
  planChartMorphAnimation,
  planChartNormalAnimation,
} from '../PlanChart'
import {
  TPAWChartDataMain,
  tpawChartDataScaled,
} from '../TPAWChart/TPAWChartDataMain'
import {TPAWChartMain} from '../TPAWChart/TPAWChartMain'
import {usePlanChartType} from '../UsePlanChartType'
import {HeightsForWidths} from './HeightsForWidths'
import {PlanChartMainCardMenu} from './PlanChartMainCardMenu'
import {PlanChartMainCardMenuButton} from './PlanChartMainCardMenuButton'
import {PlanInputMainCardHelp} from './PlanInputMainCardHelp'

export type PlanChartMainCardSizing = {
  dynamic: Record<
    PlanChartInternalTransitionState,
    {mainCard: {inset: InsetExt}}
  >
  fixed: {
    cardPadding: Padding
  }
}
type _TransitionState = `${PlanChartInternalTransitionState}${
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
    transition: {target: PlanChartInternalTransitionState; duration: number}
  }) => {
    const {params} = useSimulation()
    const type = usePlanChartType()

    const chartRef = useRef<ChartReactStatefull<TPAWChartDataMain> | null>(null)
    const [heights, setHeights] = useState<{
      menu: Record<_TransitionState, number> | null
      help: Record<_TransitionState, number> | null
    }>({menu: null, help: null})
    const [handleMenuHeights] = useState(
      () => (heights: Record<_TransitionState, number>) =>
        setHeights(x => ({...x, menu: heights}))
    )
    const [handleHelpHeights] = useState(
      () => (heights: Record<_TransitionState, number>) =>
        setHeights(x => ({...x, help: heights}))
    )
    const showSuccessRate = params.strategy === 'SWR'

    const sizing = useMemo(() => {
      const _map = (
        state: _TransitionState,
        dynamic: PlanChartMainCardSizing['dynamic']['hiddenWithLegacy'],
        showSuccessRate: boolean
      ) =>
        [
          state,
          _transformSizing(
            dynamic,
            sizingIn.fixed,
            {
              menu: heights.menu ? heights.menu[state] : 0,
              help: heights.help ? heights.help[state] : 0,
            },
            showSuccessRate
          ),
        ] as const

      return Record.merge(
        Record.map(sizingIn.dynamic, (state, dynamic) =>
          _map(`${state}WithSuccessRate`, dynamic, true)
        ),
        Record.map(sizingIn.dynamic, (state, dynamic) =>
          _map(`${state}WithoutSuccessRate`, dynamic, false)
        )
      )
    }, [heights, sizingIn])
    const targetSizing = useMemo(
      () =>
        sizing[
          `${transition.target}${
            showSuccessRate ? 'With' : 'Without'
          }SuccessRate`
        ],
      [showSuccessRate, sizing, transition.target]
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
          {x: data.years.displayRange, y: prevState.xyRange.y},
          planChartNormalAnimation
        )
      } else {
        chart.setState(
          tpawChartDataScaled(prevState.data, data.yDisplayRange),
          {x: prevState.xyRange.x, y: data.yDisplayRange},
          null
        )
        chart.setState(
          data,
          {x: data.years.displayRange, y: data.yDisplayRange},
          planChartMorphAnimation
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
        {x: prevState.xyRange.x, y: yRange},
        planChartNormalAnimation
      )
    }, [yRange])

    useEffect(() => {
      chartRef.current?.setSizing(targetSizing.chart, {
        duration: transition.duration,
        ease: cssEaseAsGSAPEase,
      })
    }, [targetSizing, transition.duration])

    const showHelp =
      transition.target === 'summaryWithLegacy' ||
      transition.target === 'summaryWithoutLegacy'
    return (
      <div
        className="absolute bg-cardBG   rounded-t-2xl rounded-b-2xl overflow-hidden border-gray-300 border"
        style={{
          transitionProperty: 'top, left, right, bottom',
          transitionDuration: `${transition.duration}ms`,
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          ...insetCSSStyle(targetSizing.inset),
        }}
      >
        <HeightsForWidths sizing={sizing} onHeights={handleMenuHeights}>
          {sizing => (
            <div style={{width: `${sizing.menu.region.width}px`}}>
              <PlanChartMainCardMenuButton
                layout={layout}
                onClick={() => {}}
                transition={transition}
              />
            </div>
          )}
        </HeightsForWidths>
        <div
          className="absolute"
          style={{...regionCSSStyle(targetSizing.menu.region)}}
        >
          <PlanChartMainCardMenu layout={layout} transition={transition} />
        </div>
        {layout !== 'mobile' && (
          <>
            <HeightsForWidths sizing={sizing} onHeights={handleHelpHeights}>
              {sizing => (
                <div className="" style={{width: sizing.help.width}}>
                  <PlanInputMainCardHelp />
                </div>
              )}
            </HeightsForWidths>
            <div
              className="absolute"
              style={{
                transitionProperty: 'opacity',
                transitionDuration: `${transition.duration}ms`,
                opacity: showHelp ? '1' : '0',
                pointerEvents: showHelp ? 'auto' : 'none',
                ...regionCSSStyle(targetSizing.help),
              }}
            >
              <PlanInputMainCardHelp />
            </div>
          </>
        )}
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
  }
)

const _transformSizing = (
  {mainCard}: PlanChartMainCardSizing['dynamic']['hiddenWithLegacy'],
  {cardPadding}: PlanChartMainCardSizing['fixed'],
  heights: {menu: number; help: number},
  showSuccessRate: boolean
) => {
  // Scaling does not affect caluclated height, so scale it here.
  const {inset} = mainCard
  const menuHeight = heights.menu
  const menu = {
    region: rectExt({
      x: cardPadding.left,
      y: cardPadding.top,
      width: inset.width - cardPadding.right - cardPadding.left,
      height: menuHeight,
    }),
  }
  const help = rectExt({
    x: cardPadding.left,
    y: menu.region.bottom,
    width: inset.width - cardPadding.right - cardPadding.left,
    height: heights.help,
  })
  const chart = {
    position: rectExt({
      x: 0,
      y: help.bottom,
      width: inset.width,
      bottom: inset.height,
    }),
    padding: {
      left: cardPadding.left,
      right: cardPadding.right,
      top: 20, // 20 is needed for the max y label.
      bottom: cardPadding.bottom + 30 + (showSuccessRate ? 30 : 0), // 30 is needed for the axis, which is in the padding.
    },
  }
  return {menu, help, chart, inset}
}
