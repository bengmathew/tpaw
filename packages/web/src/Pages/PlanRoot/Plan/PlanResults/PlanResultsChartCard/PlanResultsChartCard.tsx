import clix from 'clsx'
import getIsMobile from 'is-mobile'
import _ from 'lodash'
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { cssEaseAsGSAPEase } from '../../../../../Utils/CSSEaseAsGSAPEase'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import {
  InsetExt,
  Padding,
  insetCSSStyle,
  rectExt,
  regionCSSStyle,
} from '../../../../../Utils/Geometry'
import { Record } from '../../../../../Utils/Record'
import { SimpleRange } from '../../../../../Utils/SimpleRange'
import { useAssertConst } from '../../../../../Utils/UseAssertConst'
import { noCase } from '../../../../../Utils/Utils'
import { HeightsForWidths } from '../../../../../Utils/View/HeightsForWidths'
import { ChartPointerPortal } from '../../../../Common/Chart/ChartComponent/ChartPointerPortal'
import { ChartReactStatefull } from '../../../../Common/Chart/ChartReact'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import {
  useSimulationResultInfo
} from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'
import { useChartData } from '../../WithPlanResultsChartData'
import {
  PlanResultsTransitionState,
  planResultsMorphAnimation,
  planResultsNormalAnimation,
} from '../PlanResults'
import {
  PlanResultsChartType,
  isPlanResultsChartSpendingTotalFundingSourcesType,
} from '../PlanResultsChartType'
import { usePlanResultsChartType } from '../UsePlanResultsChartType'
import {
  PlanResultsChart,
  planResultChartOutsidePlotAreaSizing,
} from './PlanResultsChart/PlanResultsChart'
import { PlanResultsChartData } from './PlanResultsChart/PlanResultsChartData'
import { PlanResultsChartCardFundingSourcesToggle } from './PlanResultsChartCardFundingSourcesToggle'
import { PlanResultsChartCardMenu } from './PlanResultsChartCardMenu'
import { PlanResultsChartCardMenuButton } from './PlanResultsChartCardMenuButton'
import { planResultsChartLabel } from './PlanResultsChartLabel'

export type PlanResultsChartCardSizing = {
  dynamic: Record<PlanResultsTransitionState, { mainCard: { inset: InsetExt } }>
  fixed: {
    cardPadding: Padding
  }
}
type _TransitionState = `${PlanResultsTransitionState}${
  | 'With'
  | 'Without'}SuccessRate`

export const PlanResultsChartCard = React.memo(
  ({
    layout,
    yRange,
    setYRange,
    sizing: sizingIn,
    transition,
    chartPointerPortal,
    onChartHover,
    chartHover,
  }: {
    layout: 'laptop' | 'desktop' | 'mobile'
    yRange: SimpleRange
    setYRange: (y: SimpleRange) => void
    sizing: PlanResultsChartCardSizing
    transition: { target: PlanResultsTransitionState; duration: number }
    chartPointerPortal: ChartPointerPortal
    onChartHover: (hover: boolean) => void
    chartHover: boolean
  }) => {
    const simulationResult = useSimulationResultInfo().simulationResult
    const type = usePlanResultsChartType()
    const yAxisDescriptionByLayout = planResultsChartLabel(
      simulationResult.planParamsNormOfResult,
      type,
    ).yAxisDescription

    const showFundingSourcesToggle =
      type === 'spending-total' ||
      isPlanResultsChartSpendingTotalFundingSourcesType(type)

    const [showRealBlurbHelp, setShowRealBlurbHelp] = useState(false)
    const chartRef = useRef<ChartReactStatefull<{
      data: PlanResultsChartData
      yRange: SimpleRange
    }> | null>(null)
    const [heights, setHeights] = useState<{
      menu: Record<_TransitionState, number> | null
      yAxisDescription: number
      breakdownMenu: number
    }>({
      menu: null,
      // Hardcoding this because measuring interferes with transition animation.
      // See note below.
      yAxisDescription: layout === 'mobile' ? 22 : 26,
      breakdownMenu: layout === 'mobile' ? 22 : 26,
    })
    const [handleMenuHeights] = useState(
      () => (heights: Record<_TransitionState, number>) =>
        setHeights((x) => ({ ...x, menu: heights })),
    )

    // NOTE: This interferes with transition animation because it will be
    // fired at every frame. Avoid.
    //
    // useLayoutEffect(() => {
    //   const observer = new ResizeObserver(() => {
    //     const realBlurb = fGet(realBlurbRef.current).getBoundingClientRect()
    //     setHeights((x) => ({ ...x, realBlurb: realBlurb.height }))
    //   })
    //   observer.observe(fGet(realBlurbRef.current))
    //   return () => observer.disconnect()
    // }, [])

    const getYAxisDescription = (isMobile: boolean) =>
      yAxisDescriptionByLayout
        ? isMobile
          ? yAxisDescriptionByLayout.mobile
          : yAxisDescriptionByLayout.notMobile
        : null
    const showYAxisDescription = !!getYAxisDescription(layout === 'mobile')
    const showSuccessRate =
      simulationResult.planParamsNormOfResult.advanced.strategy === 'SWR'
    const hasPartner = !!simulationResult.planParamsNormOfResult.ages.person2

    const sizing = useMemo(() => {
      const _map = (
        state: _TransitionState,
        dynamic: PlanResultsChartCardSizing['dynamic']['dialogInput'],
        showSuccessRate: boolean,
      ) =>
        [
          state,
          _transformSizing(
            dynamic,
            sizingIn.fixed,
            {
              menu: heights.menu ? heights.menu[state] : 0,
              yAxisDescription: heights.yAxisDescription,
              breakdownMenu: heights.breakdownMenu,
            },
            hasPartner,
            showSuccessRate,
            showYAxisDescription,
            showFundingSourcesToggle,
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
    }, [
      heights,
      sizingIn,
      hasPartner,
      showYAxisDescription,
      showFundingSourcesToggle,
    ])
    const targetSizing = useMemo(
      () =>
        sizing[
          `${transition.target}${
            showSuccessRate ? 'With' : 'Without'
          }SuccessRate`
        ],
      [showSuccessRate, sizing, transition.target],
    )
    const yAxisDescription = getYAxisDescription(
      layout === 'mobile' || targetSizing.yAxisDescription.width < 400,
    )

    const chartData = useChartData(type)

    useEffect(() => {
      const chart = chartRef.current
      if (!chart) return
      const prevData = chart.getData()
      const typeChanged = chartData.chartType !== prevData.data.chartType
      const newData: Parameters<typeof chart.setData>[0] = {
        data: chartData,
        yRange:
          typeChanged &&
          _shouldResizeOnChartChange(
            prevData.data.chartType,
            chartData.chartType,
          )
            ? chartData.displayRange.y
            : yRange,
      }

      // We have to be careful not to call setData unnecessarily because it
      // will jump a running transition to the target. Normally this would be
      // fine because we are responding to changes in the source data, except
      // we are calling setYRange() in this function which will cause the
      // effect to run again since yRange is in our dependency list.
      if (
        newData.data === prevData.data &&
        _.isEqual(newData.yRange, prevData.yRange)
      )
        return

      chart.setData(
        newData,
        typeChanged ? planResultsMorphAnimation : planResultsNormalAnimation,
      )
      setYRange(chart.getData().yRange)
    }, [chartData, yRange, setYRange])
    useAssertConst([setYRange])

    useEffect(() => {
      chartRef.current?.setSizing(targetSizing.chart, {
        duration: transition.duration,
        ease: cssEaseAsGSAPEase,
      })
    }, [targetSizing, transition.duration])

    const numSuccessfullRuns =
      simulationResult.numSimulationsActual -
      simulationResult.numRunsWithInsufficientFunds
    const successRate =
      numSuccessfullRuns / simulationResult.numSimulationsActual
    const planColors = usePlanColors()

    const isMobile = useMemo(() => getIsMobile(), [])

    return (
      <div
        className={clix(
          'absolute   rounded-t-2xl rounded-b-2xl overflow-hidden ',
          chartHover && 'custom-shadow-md',
        )}
        style={{
          transitionProperty: 'top, left, right, bottom, box-shadow',
          transitionDuration: `${transition.duration}ms`,
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          ...insetCSSStyle(targetSizing.inset),
          backgroundColor: planColors.results.cardBG,
        }}
        onPointerMove={
          !isMobile ? (e) => chartRef.current?.onPointerMove(e) : undefined
        }
        onPointerEnter={
          !isMobile ? (e) => chartRef.current?.onPointerEnter(e) : undefined
        }
        onPointerLeave={
          !isMobile ? (e) => chartRef.current?.onPointerLeave(e) : undefined
        }
      >
        <HeightsForWidths sizing={sizing} onHeights={handleMenuHeights}>
          {(sizing) => (
            <div style={{ width: `${sizing.menu.region.width}px` }}>
              <PlanResultsChartCardMenuButton transition={transition} />
            </div>
          )}
        </HeightsForWidths>
        <div
          className="absolute"
          style={{ ...regionCSSStyle(targetSizing.menu.region) }}
        >
          <PlanResultsChartCardMenu transition={transition} />
        </div>
        {yAxisDescription && (
          <div
            className={clix(
              'absolute  lighten text-[14px] sm:text-[16px] whitespace-nowrap ',
            )}
            style={{
              left: `${targetSizing.yAxisDescription.x}px`,
              top: `${targetSizing.yAxisDescription.y}px`,
              width: `${targetSizing.yAxisDescription.width}px`,
            }}
          >
            <div>
              {yAxisDescription.map((x, i) =>
                x.type === 'plain' ? (
                  <span key={i}>{x.value} </span>
                ) : x.type === 'inflation' ? (
                  <span
                    key={i}
                    className="underline cursor-pointer"
                    onClick={() => setShowRealBlurbHelp(true)}
                  >
                    {x.value}{' '}
                  </span>
                ) : (
                  noCase(x)
                ),
              )}
            </div>
          </div>
        )}

        {showFundingSourcesToggle && (
          <PlanResultsChartCardFundingSourcesToggle
            className="absolute"
            layout={layout}
            style={{
              left: `${targetSizing.breakdownMenu.x}px`,
              top: `${targetSizing.breakdownMenu.y}px`,
              width: `${targetSizing.breakdownMenu.width}px`,
            }}
          />
        )}

        <PlanResultsChart
          starting={{
            data: { data: chartData, yRange },
            sizing: targetSizing.chart,
          }}
          ref={chartRef}
          chartPointerPortal={chartPointerPortal}
          onHover={onChartHover}
          capturePointerEvents={isMobile}
        />
        {showSuccessRate && (
          <div
            className="absolute flex items-center font-bold"
            style={{
              right: `${targetSizing.successRate.inset.right}px`,
              height: `${targetSizing.successRate.inset.height}px`,
              bottom: `${0}px`,
            }}
          >
            Success Rate:{' '}
            {successRate < 0.95
              ? formatPercentage(0)(successRate)
              : successRate < 0.99
                ? formatPercentage(1)(successRate)
                : formatPercentage(2)(successRate)}
            {simulationResult.planParamsNormOfResult.advanced.sampling.type ===
              'historical' && (
              <span className="sm:inline hidden ml-2 font-normal text-sm lighten">
                {numSuccessfullRuns} of {simulationResult.numSimulationsActual}
              </span>
            )}
          </div>
        )}
        <CenteredModal
          className=" dialog-outer-div"
          show={showRealBlurbHelp}
          onOutsideClickOrEscape={() => setShowRealBlurbHelp(false)}
          style={{
            backgroundColor: planColors.results.cardBG,
            color: planColors.results.fg,
          }}
        >
          <div className="">
            <p className="font-font2 text-lg ">
              {`Future dollars are adjusted for inflation and expressed as today's
          dollars. For example, thirty years from now, $10 might buy only what
          $5 buys today. In that case, $10 thirty years from now is expressed as
          just $5 on the graph. So a dollar at any age on the graph has the same
          buying power as a dollar does today.`}
            </p>
            <p className="font-font2 text-lg  mt-3">{`When dollars are adjusted for
          inflation in this way, they are called “real” dollars.`}</p>
          </div>
        </CenteredModal>
      </div>
    )
  },
)

const _transformSizing = (
  { mainCard }: PlanResultsChartCardSizing['dynamic']['dialogInput'],
  { cardPadding }: PlanResultsChartCardSizing['fixed'],
  heights: { menu: number; yAxisDescription: number; breakdownMenu: number },
  hasPartner: boolean,
  showSuccessRate: boolean,
  showYAxisDescription: boolean,
  showBreakdownMenu: boolean,
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

  const yAxisDescription = {
    x: menu.region.x,
    y: menu.region.bottom,
    width: menu.region.width,
  }
  const breakdownMenu = {
    x: menu.region.x,
    y: yAxisDescription.y + heights.yAxisDescription,
    width: menu.region.width,
  }

  const successRateHeight = 30
  const chart = (() => {
    const position = rectExt({
      x: 0,
      y:
        menu.region.bottom +
        (showYAxisDescription ? heights.yAxisDescription : 0) +
        (showBreakdownMenu ? heights.breakdownMenu : 0),
      width: inset.width,
      bottom: inset.height - (showSuccessRate ? successRateHeight : 0),
    })

    return {
      position,
      padding: {
        left: cardPadding.left,
        right: cardPadding.right,
        top: 20, // 20 is needed for the max y label.
        bottom:
          planResultChartOutsidePlotAreaSizing.below.gapToLine +
          planResultChartOutsidePlotAreaSizing.below.lineWidth +
          planResultChartOutsidePlotAreaSizing.below.xAxis.height +
          (hasPartner
            ? planResultChartOutsidePlotAreaSizing.below.xAxis
                .vertGapBetweenPeople +
              planResultChartOutsidePlotAreaSizing.below.xAxis.height
            : 0) +
          planResultChartOutsidePlotAreaSizing.below.xAxis.vertGapBetweenPeople,
      },
    }
  })()

  const successRate = {
    inset: { right: cardPadding.right, height: successRateHeight },
  }

  return { menu, chart, yAxisDescription, breakdownMenu, successRate, inset }
}

const _shouldResizeOnChartChange = (
  prev: PlanResultsChartType,
  next: PlanResultsChartType,
) => {
  if (isPlanResultsChartSpendingTotalFundingSourcesType(next)) return false
  // if (
  //   prev === 'spending-total' &&
  //   isPlanResultsChartSpendingTotalBreakdownType(next)
  // )
  //   return false
  // if (
  //   isPlanResultsChartSpendingTotalBreakdownType(prev) &&
  //   isPlanResultsChartSpendingTotalBreakdownType(next) &&
  //   parseInt(planResultsChartSpendingTotalBreakdownPercentile(prev)) >
  //     parseInt(planResultsChartSpendingTotalBreakdownPercentile(next))
  // )
  //   return false
  return true
}
