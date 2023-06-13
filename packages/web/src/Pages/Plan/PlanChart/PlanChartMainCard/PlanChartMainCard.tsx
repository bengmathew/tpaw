import React, { useEffect, useMemo, useRef, useState } from 'react'
import { cssEaseAsGSAPEase } from '../../../../Utils/CSSEaseAsGSAPEase'
import { Contentful } from '../../../../Utils/Contentful'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import {
  InsetExt,
  Padding,
  insetCSSStyle,
  rectExt,
  regionCSSStyle,
} from '../../../../Utils/Geometry'
import { Record } from '../../../../Utils/Record'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { useAssertConst } from '../../../../Utils/UseAssertConst'
import { fGet, noCase } from '../../../../Utils/Utils'
import { useChartData } from '../../../App/WithChartData'
import { useSimulation } from '../../../App/WithSimulation'
import { ChartReactStatefull } from '../../../Common/Chart/ChartReact'
import { ModalBase } from '../../../Common/Modal/ModalBase'
import { usePlanContent } from '../../Plan'
import {
  PlanChartTransitionState,
  planChartMorphAnimation,
  planChartNormalAnimation,
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
import { planChartLabel } from './PlanChartLabel'
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
    const yAxisDescriptionByLayout = planChartLabel(
      params,
      type,
      'full',
    ).yAxisDescription
    const yAxisDescription = yAxisDescriptionByLayout
      ? layout === 'mobile'
        ? yAxisDescriptionByLayout.mobile
        : yAxisDescriptionByLayout.notMobile
      : null

    const [showRealBlurbHelp, setShowRealBlurbHelp] = useState(false)
    const chartRef = useRef<ChartReactStatefull<TPAWChartDataMain> | null>(null)
    const [heights, setHeights] = useState<{
      menu: Record<_TransitionState, number> | null
      yAxisDescription: number | null
    }>({
      menu: null,
      // Hardcoding this because measuring interferes with transition animation.
      // See not below.
      yAxisDescription: layout === 'mobile' ? 20 : 25,
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

    const showSuccessRate = params.plan.advanced.strategy === 'SWR'
    const hasPartner = tpawResult.params.people.withPartner

    const sizing = useMemo(() => {
      const _map = (
        state: _TransitionState,
        dynamic: PlanChartMainCardSizing['dynamic']['dialogInput'],
        showSuccessRate: boolean,
      ) =>
        [
          state,
          _transformSizing(
            dynamic,
            sizingIn.fixed,
            {
              menu: heights.menu ? heights.menu[state] : 0,
              yAxisDescription: heights.yAxisDescription ?? 0,
            },
            hasPartner,
            showSuccessRate,
            yAxisDescription !== null,
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
    }, [heights, sizingIn, hasPartner, yAxisDescription])
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
    const chartMainData = fGet(allChartData.byYearsFromNowPercentiles.get(type))

    useEffect(() => {
      const chart = chartRef.current
      if (!chart) return
      const data = chartMainData
      const prevState = chart.getState()
      if (type === prevState.data.type) {
        chart.setState(
          data,
          { x: data.months.displayRange, y: prevState.xyRange.y },
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
          { x: data.months.displayRange, y: data.yDisplayRange },
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

    const planContent = usePlanContent()

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
        {yAxisDescription !== null && (
          <div
            className={`absolute  lighten
          ${layout === 'mobile' ? 'text-[13px]' : ' text-[16px]'}`}
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
            {/* {yAxisDescription === 'realDollarsExplanation' ? (
              <div>
                {' '}
                {layout === 'mobile' ? 'Dollars ' : 'These dollars are '}
                <span
                  className="underline cursor-pointer"
                  onClick={() => setShowRealBlurbHelp(true)}
                >
                  adjusted for inflation
                </span>
              </div>
            ) : yAxisDescription === 'assetAllocationExplanation' ? (
              <>{`Percentage of portfolio in stocks`}</>
            ) : (
              noCase(yAxisDescription)
            )} */}
          </div>
        )}

        <TPAWChartMain
          starting={{
            data: chartMainData,
            xyRange: {
              x: chartMainData.months.displayRange,
              y: chartMainData.yDisplayRange,
            },
            sizing: targetSizing.chart,
          }}
          ref={chartRef}
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
            {formatPercentage(0)(
              1 - tpawResult.percentageOfRunsWithInsufficientFunds,
            )}
          </div>
        )}
        {showRealBlurbHelp && (
          <ModalBase onClose={() => setShowRealBlurbHelp(false)}>
            {(transitionOut) => (
              <div className="-mt-4">
                <Contentful.RichText
                  body={
                    planContent.chart.realBlurb[params.plan.advanced.strategy]
                  }
                  p={`p-base mt-4`}
                />
              </div>
            )}
          </ModalBase>
        )}
      </div>
    )
  },
)

const _transformSizing = (
  { mainCard }: PlanChartMainCardSizing['dynamic']['dialogInput'],
  { cardPadding }: PlanChartMainCardSizing['fixed'],
  heights: { menu: number; yAxisDescription: number },
  hasPartner: boolean,
  showSuccessRate: boolean,
  showYAxisDescription: boolean,
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

  const successRateHeight = 30
  const chart = (() => {
    const position = rectExt({
      x: 0,
      y:
        menu.region.bottom +
        (showYAxisDescription ? heights.yAxisDescription : 0),
      width: inset.width,
      bottom: inset.height - (showSuccessRate ? successRateHeight : 0),
    })
    const xAxisSizing = tpawChartMainXAxisSizing(position.width)
    return {
      position,
      padding: {
        left: cardPadding.left,
        right: cardPadding.right,
        top: 20, // 20 is needed for the max y label.
        bottom:
          2 + // gap below plotArea.
          4 + // line at bottom of plotArea.
          xAxisSizing.height +
          (hasPartner
            ? +3 * xAxisSizing.gap + xAxisSizing.height
            : 2 * xAxisSizing.gap),
      },
    }
  })()

  const successRate = {
    inset: { right: cardPadding.right, height: successRateHeight },
  }

  return { menu, chart, yAxisDescription, successRate, inset }
}
